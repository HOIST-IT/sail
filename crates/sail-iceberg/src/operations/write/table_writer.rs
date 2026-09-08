// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, new_null_array};
use datafusion::arrow::datatypes::{FieldRef, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use sail_common_datafusion::array::record_batch::cast_record_batch_relaxed_tz;
use url::Url;

use crate::operations::write::arrow_parquet::ArrowParquetWriter;
use crate::operations::write::base_writer::DataFileWriter;
use crate::operations::write::config::WriterConfig;
use crate::operations::write::file_writer::location_generator::DefaultLocationGenerator;
use crate::operations::write::partition::split_record_batch_by_partition;
use crate::operations::write::variant_shredding::{
    VariantShreddingPlan, apply_variant_shredding_plan, build_variant_shredding_plan,
    unshred_shredded_variants_for_write,
};
use crate::spec::DataFile;
use crate::spec::schema::Schema as IcebergSchema;
use crate::spec::types::NestedField;
use crate::spec::types::values::Literal;
use crate::utils::conversions::to_scalar;

enum PartitionWriterState {
    Pending {
        batches: Vec<RecordBatch>,
        num_rows: usize,
    },
    Open {
        writer: Box<ArrowParquetWriter>,
        variant_shredding_plan: Option<VariantShreddingPlan>,
    },
}

struct PartitionWriter {
    partition_dir: String,
    state: PartitionWriterState,
}

pub struct IcebergTableWriter {
    pub store: Arc<dyn object_store::ObjectStore>,
    pub config: WriterConfig,
    pub generator: DefaultLocationGenerator,
    pub data_url: Url,
    // Typed partition tuple -> writer.
    // TODO: Roll each partition writer using the `target-file-size-bytes` write option or
    // `write.target-file-size-bytes` table property.
    writers: HashMap<Vec<Option<Literal>>, PartitionWriter>,
    // Keys of `writers`, least recently written first. Bounds how many partition writers
    // are open at once so memory does not scale with the partition cardinality one task
    // happens to see.
    open_order: Vec<Vec<Option<Literal>>>,
    written: Vec<DataFile>,
    pub partition_spec_id: i32,
}

impl IcebergTableWriter {
    pub fn new(
        store: Arc<dyn object_store::ObjectStore>,
        root: ObjectPath,
        config: WriterConfig,
        partition_spec_id: i32,
        data_url: Url,
    ) -> Self {
        Self {
            generator: DefaultLocationGenerator::new(root),
            store,
            config,
            data_url,
            writers: HashMap::new(),
            open_order: Vec::new(),
            written: Vec::new(),
            partition_spec_id,
        }
    }

    pub async fn write(&mut self, batch: &RecordBatch) -> Result<(), String> {
        let spec = &self.config.partition_spec;
        let iceberg_schema = &self.config.iceberg_schema;
        let parts = split_record_batch_by_partition(batch, spec, iceberg_schema)?;
        for p in parts {
            let partition_dir = p.partition_dir;
            let partition_values = p.partition_values;
            let padded = Self::align_batch_with_table_schema(
                &p.record_batch,
                &self.config.table_schema,
                self.config.iceberg_schema.as_ref(),
            )
            .map_err(|e| e.to_string())?;
            let normalized =
                unshred_shredded_variants_for_write(&padded, &self.config.table_schema)?;
            let aligned = cast_record_batch_relaxed_tz(&normalized, &self.config.table_schema)
                .map_err(|e| e.to_string())?;
            self.write_aligned_batch(partition_values, partition_dir, aligned)
                .await?;
        }

        Ok(())
    }

    async fn write_aligned_batch(
        &mut self,
        partition_values: Vec<Option<Literal>>,
        partition_dir: String,
        batch: RecordBatch,
    ) -> Result<(), String> {
        let (partition_dir, state) = match self.writers.remove(&partition_values) {
            Some(writer) => (writer.partition_dir, writer.state),
            None => (partition_dir, self.new_partition_writer_state()?),
        };
        let state = self.write_partition_state(state, batch).await?;
        self.writers.insert(
            partition_values.clone(),
            PartitionWriter {
                partition_dir,
                state,
            },
        );
        self.touch_open(&partition_values);
        self.enforce_open_writer_bound().await?;
        Ok(())
    }

    /// Move `key` to the most recently written end of `open_order`.
    fn touch_open(&mut self, key: &[Option<Literal>]) {
        match self.open_order.iter().position(|k| k.as_slice() == key) {
            Some(pos) => {
                let existing = self.open_order.remove(pos);
                self.open_order.push(existing);
            }
            None => self.open_order.push(key.to_vec()),
        }
    }

    /// Finish least recently written partitions until the open-writer bound holds.
    ///
    /// Eviction emits an additional data file for that partition, which Iceberg allows, so
    /// this trades file count for a memory ceiling rather than failing the write. When the
    /// plan sorts by partition key the bound is never reached, because one partition is
    /// live at a time.
    async fn enforce_open_writer_bound(&mut self) -> Result<(), String> {
        let max_open = self.config.max_open_writers.max(1);
        while self.writers.len() > max_open {
            let Some(evicted) = self.open_order.first().cloned() else {
                break;
            };
            self.open_order.remove(0);
            let Some(writer) = self.writers.remove(&evicted) else {
                continue;
            };
            self.flush_partition(writer.state, &writer.partition_dir, evicted)
                .await?;
        }
        Ok(())
    }

    fn new_partition_writer_state(&self) -> Result<PartitionWriterState, String> {
        if self.config.variant_shredding.enabled {
            Ok(PartitionWriterState::Pending {
                batches: Vec::new(),
                num_rows: 0,
            })
        } else {
            Ok(PartitionWriterState::Open {
                writer: Box::new(self.new_arrow_writer(self.config.table_schema.clone())?),
                variant_shredding_plan: None,
            })
        }
    }

    async fn write_partition_state(
        &mut self,
        state: PartitionWriterState,
        batch: RecordBatch,
    ) -> Result<PartitionWriterState, String> {
        match state {
            PartitionWriterState::Pending {
                mut batches,
                mut num_rows,
            } => {
                num_rows += batch.num_rows();
                batches.push(batch);
                if num_rows >= self.config.variant_shredding.inference_buffer_size.max(1) {
                    self.open_and_write_pending_batches(batches).await
                } else {
                    Ok(PartitionWriterState::Pending { batches, num_rows })
                }
            }
            PartitionWriterState::Open {
                mut writer,
                variant_shredding_plan,
            } => {
                let batch = if let Some(plan) = variant_shredding_plan.as_ref() {
                    apply_variant_shredding_plan(&batch, plan)?
                } else {
                    batch
                };
                writer.write_batch(&batch).await?;
                Ok(PartitionWriterState::Open {
                    writer,
                    variant_shredding_plan,
                })
            }
        }
    }

    async fn open_and_write_pending_batches(
        &mut self,
        batches: Vec<RecordBatch>,
    ) -> Result<PartitionWriterState, String> {
        let plan = build_variant_shredding_plan(
            &self.config.table_schema,
            &batches,
            self.config.variant_shredding.inference_buffer_size,
            self.config.variant_shredding.inference_node_budget,
        )?;
        let plan = (!plan.is_noop()).then_some(plan);
        let physical_batches = batches
            .into_iter()
            .map(|batch| {
                if let Some(plan) = plan.as_ref() {
                    apply_variant_shredding_plan(&batch, plan)
                } else {
                    Ok(batch)
                }
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;

        let schema = physical_batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(|| self.config.table_schema.clone());
        let mut writer = self.new_arrow_writer(schema)?;
        for batch in physical_batches {
            writer.write_batch(&batch).await?;
        }
        Ok(PartitionWriterState::Open {
            writer: Box::new(writer),
            variant_shredding_plan: plan,
        })
    }

    fn new_arrow_writer(&self, schema: SchemaRef) -> Result<ArrowParquetWriter, String> {
        for (i, f) in schema.fields().iter().enumerate() {
            log::trace!(
                "iceberg.table_writer.writer_schema: field[{}]='{}' type={:?} field_id_meta={:?}",
                i,
                f.name(),
                f.data_type(),
                f.metadata().get(PARQUET_FIELD_ID_META_KEY)
            );
        }
        ArrowParquetWriter::try_new(schema.as_ref(), self.config.writer_properties.clone())
    }

    async fn finish_partition_state(
        &mut self,
        state: PartitionWriterState,
    ) -> Result<ArrowParquetWriter, String> {
        match state {
            PartitionWriterState::Pending { batches, .. } => {
                let PartitionWriterState::Open { writer, .. } =
                    self.open_and_write_pending_batches(batches).await?
                else {
                    return Err("failed to open pending Iceberg partition writer".to_string());
                };
                Ok(*writer)
            }
            PartitionWriterState::Open { writer, .. } => Ok(*writer),
        }
    }

    async fn flush_partition(
        &mut self,
        state: PartitionWriterState,
        partition_dir: &str,
        partition_values: Vec<Option<Literal>>,
    ) -> Result<(), String> {
        let writer = self.finish_partition_state(state).await?;
        let (bytes, meta) = writer.close().await?;
        let (rel, full) = self.generator.next_data_path(Some(partition_dir))?;
        log::trace!("iceberg.table_writer.flush_partition.writing: {}", full);
        self.store
            .put(&full, object_store::PutPayload::from(bytes))
            .await
            .map_err(|e| e.to_string())?;
        log::trace!(
            "iceberg.table_writer.flush_partition.written: rel={} full={}",
            rel,
            full
        );
        // Prevent a leading partition segment containing ':' from being parsed as a URI scheme.
        let file_path = match self.data_url.join(&format!("./{rel}")) {
            Ok(u) => u.to_string(),
            Err(_) => {
                format!("{}{}", self.data_url.as_str(), rel)
            }
        };
        let df = DataFileWriter::new(self.partition_spec_id, file_path, partition_values)
            .finish_with_schema(meta, self.config.iceberg_schema.as_ref())?
            .data_file;
        self.written.push(df);
        Ok(())
    }

    pub async fn close(mut self) -> Result<Vec<DataFile>, String> {
        for (partition_values, writer) in std::mem::take(&mut self.writers) {
            self.flush_partition(writer.state, &writer.partition_dir, partition_values)
                .await?;
        }
        Ok(self.written)
    }

    fn align_batch_with_table_schema(
        batch: &RecordBatch,
        table_schema: &SchemaRef,
        iceberg_schema: &IcebergSchema,
    ) -> Result<RecordBatch, DataFusionError> {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(table_schema.fields().len());
        let mut schema_fields: Vec<FieldRef> = Vec::with_capacity(table_schema.fields().len());

        for field in table_schema.fields() {
            match batch.schema().index_of(field.name()) {
                Ok(idx) => {
                    columns.push(batch.column(idx).clone());
                    schema_fields.push(Arc::new(batch.schema().field(idx).clone()));
                }
                Err(_) => {
                    let array =
                        Self::build_missing_column_array(field, iceberg_schema, batch.num_rows())?;
                    columns.push(array);
                    schema_fields.push(field.clone());
                }
            }
        }

        let aligned_schema = Arc::new(Schema::new(schema_fields));
        Ok(RecordBatch::try_new(aligned_schema, columns)?)
    }

    fn build_missing_column_array(
        field: &FieldRef,
        iceberg_schema: &IcebergSchema,
        num_rows: usize,
    ) -> Result<ArrayRef, DataFusionError> {
        let iceberg_field = iceberg_schema.field_by_name(field.name()).ok_or_else(|| {
            DataFusionError::Plan(format!(
                "Column '{}' missing from Iceberg schema during alignment",
                field.name()
            ))
        })?;

        if let Some(array) = Self::default_array_for_field(iceberg_field.as_ref(), num_rows)? {
            return Ok(array);
        }

        if field.is_nullable() {
            return Ok(new_null_array(field.data_type(), num_rows));
        }

        Err(DataFusionError::Plan(format!(
            "Column '{}' is required but missing in input batch and has no default value",
            field.name()
        )))
    }

    fn default_array_for_field(
        field: &NestedField,
        num_rows: usize,
    ) -> Result<Option<ArrayRef>, DataFusionError> {
        let literal = field
            .write_default
            .as_ref()
            .or(field.initial_default.as_ref());
        if let Some(lit) = literal {
            let scalar = to_scalar(lit, field.field_type.as_ref())?;
            let array = scalar
                .to_array_of_size(num_rows)
                .map_err(|e| DataFusionError::Plan(e.to_string()))?;
            return Ok(Some(array));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use object_store::memory::InMemory;

    use super::*;
    use crate::operations::write::config::WriterConfig;
    use crate::spec::partition::{UnboundPartitionField, UnboundPartitionSpec};
    use crate::spec::types::{PrimitiveType, Type};
    use crate::spec::{Schema as SpecSchema, Transform};

    fn field_with_id(name: &str, data_type: DataType, id: &str) -> Field {
        Field::new(name, data_type, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    /// One writer per partition tuple used to be held open until `close`, so memory grew
    /// with the partition cardinality a single task happened to see. Writing more
    /// partitions than the bound must finish the oldest rather than accumulate.
    #[test]
    fn open_partition_writers_stay_within_the_configured_bound() -> Result<(), String> {
        const MAX_OPEN: usize = 2;
        const PARTITIONS: i32 = 6;

        futures::executor::block_on(async {
            let arrow_schema = Arc::new(ArrowSchema::new(vec![
                field_with_id("id", DataType::Int32, "1"),
                field_with_id("part", DataType::Utf8, "2"),
            ]));
            let iceberg_schema = SpecSchema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    Arc::new(NestedField::optional(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::optional(
                        2,
                        "part",
                        Type::Primitive(PrimitiveType::String),
                    )),
                ])
                .build()
                .map_err(|error| error.to_string())?;
            let config = WriterConfig {
                table_schema: arrow_schema.clone(),
                writer_properties: parquet::file::properties::WriterProperties::builder().build(),
                iceberg_schema: Arc::new(iceberg_schema),
                partition_spec: UnboundPartitionSpec {
                    fields: vec![UnboundPartitionField {
                        source_id: 2,
                        name: "part".to_string(),
                        transform: Transform::Identity,
                    }],
                },
                variant_shredding: Default::default(),
                max_open_writers: MAX_OPEN,
            };
            let mut writer = IcebergTableWriter::new(
                Arc::new(InMemory::new()),
                ObjectPath::from("t"),
                config,
                0,
                Url::parse("memory:///t/data/").map_err(|error| error.to_string())?,
            );

            // Each batch carries one distinct partition, so an unbounded map would hold six.
            for partition in 0..PARTITIONS {
                let batch = RecordBatch::try_new(
                    arrow_schema.clone(),
                    vec![
                        Arc::new(Int32Array::from(vec![partition])),
                        Arc::new(StringArray::from(vec![format!("p{partition}")])),
                    ],
                )
                .map_err(|error| error.to_string())?;
                writer.write(&batch).await?;
                assert!(
                    writer.writers.len() <= MAX_OPEN,
                    "held {} open writers, bound is {MAX_OPEN}",
                    writer.writers.len()
                );
            }

            // Eviction must flush rather than drop: every partition still produces a file.
            let files = writer.close().await?;
            assert_eq!(files.len(), PARTITIONS as usize);
            Ok(())
        })
    }

    /// The allow pin for the bound: a write that stays under it keeps its writer open, so
    /// the common sorted case is not split into one file per batch.
    #[test]
    fn a_partition_under_the_bound_is_not_evicted_between_batches() -> Result<(), String> {
        futures::executor::block_on(async {
            let arrow_schema = Arc::new(ArrowSchema::new(vec![
                field_with_id("id", DataType::Int32, "1"),
                field_with_id("part", DataType::Utf8, "2"),
            ]));
            let iceberg_schema = SpecSchema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    Arc::new(NestedField::optional(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::optional(
                        2,
                        "part",
                        Type::Primitive(PrimitiveType::String),
                    )),
                ])
                .build()
                .map_err(|error| error.to_string())?;
            let config = WriterConfig {
                table_schema: arrow_schema.clone(),
                writer_properties: parquet::file::properties::WriterProperties::builder().build(),
                iceberg_schema: Arc::new(iceberg_schema),
                partition_spec: UnboundPartitionSpec {
                    fields: vec![UnboundPartitionField {
                        source_id: 2,
                        name: "part".to_string(),
                        transform: Transform::Identity,
                    }],
                },
                variant_shredding: Default::default(),
                max_open_writers: 4,
            };
            let mut writer = IcebergTableWriter::new(
                Arc::new(InMemory::new()),
                ObjectPath::from("t"),
                config,
                0,
                Url::parse("memory:///t/data/").map_err(|error| error.to_string())?,
            );

            for id in 0..3 {
                let batch = RecordBatch::try_new(
                    arrow_schema.clone(),
                    vec![
                        Arc::new(Int32Array::from(vec![id])),
                        Arc::new(StringArray::from(vec!["p0"])),
                    ],
                )
                .map_err(|error| error.to_string())?;
                writer.write(&batch).await?;
            }
            assert_eq!(writer.writers.len(), 1);

            let files = writer.close().await?;
            assert_eq!(files.len(), 1, "three batches of one partition must not split");
            Ok(())
        })
    }
}
