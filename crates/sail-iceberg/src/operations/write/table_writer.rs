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

use datafusion::arrow::array::ArrayRef;
use datafusion::arrow::datatypes::{FieldRef, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use object_store::buffered::BufWriter;
use object_store::path::Path as ObjectPath;
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
use sail_common_datafusion::schema_evolution::{
    StructFieldMatching, cast_array_with_schema_evolution_relaxed_tz,
};
use url::Url;

use crate::operations::write::arrow_parquet::{ArrowParquetWriter, ParquetObjectSink};
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
use crate::spec::types::values::Literal;

enum PartitionWriterState {
    Pending {
        batches: Vec<RecordBatch>,
        num_rows: usize,
    },
    Open {
        writer: Box<ArrowParquetWriter<ParquetObjectSink>>,
        file_path: String,
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
        let padded =
            Self::align_batch_with_table_schema(batch, &self.config.table_schema, iceberg_schema)
                .map_err(|e| e.to_string())?;
        let normalized = unshred_shredded_variants_for_write(&padded, &self.config.table_schema)?;
        let columns = normalized
            .columns()
            .iter()
            .zip(self.config.table_schema.fields())
            .map(|(column, field)| {
                cast_array_with_schema_evolution_relaxed_tz(
                    column,
                    field,
                    &Default::default(),
                    StructFieldMatching::Name,
                )
            })
            .collect::<Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        let aligned = RecordBatch::try_new(self.config.table_schema.clone(), columns)
            .map_err(|e| e.to_string())?;
        let parts = split_record_batch_by_partition(&aligned, spec, iceberg_schema)?;
        for p in parts {
            self.write_aligned_batch(p.partition_values, p.partition_dir, p.record_batch)
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
        // Check encoded size at a bounded row interval even for large input batches.
        for offset in (0..batch.num_rows()).step_by(1000) {
            let chunk = batch.slice(offset, (batch.num_rows() - offset).min(1000));
            let state = match self.writers.remove(&partition_values) {
                Some(writer) => writer.state,
                None => self.new_partition_writer_state(&partition_dir)?,
            };
            let state = self
                .write_partition_state(state, chunk, &partition_dir)
                .await?;
            if matches!(&state, PartitionWriterState::Open { writer, .. }
                if writer.estimated_size() >= self.config.target_file_size_bytes)
            {
                self.close_open(&partition_values);
                self.flush_partition(state, &partition_dir, partition_values.clone())
                    .await?;
            } else {
                self.writers.insert(
                    partition_values.clone(),
                    PartitionWriter {
                        partition_dir: partition_dir.clone(),
                        state,
                    },
                );
                self.touch_open(&partition_values);
                self.enforce_open_writer_bound().await?;
            }
        }
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

    /// Forget `key` once its writer has left `writers`, so the two stay in step.
    fn close_open(&mut self, key: &[Option<Literal>]) {
        if let Some(pos) = self.open_order.iter().position(|k| k.as_slice() == key) {
            self.open_order.remove(pos);
        }
    }

    /// Finish least recently written partitions until the open-writer bound holds.
    ///
    /// Eviction trades file count for a memory ceiling rather than failing the write, which
    /// Iceberg allows. The writer's required input ordering leads with the partition keys, so
    /// a task writes one partition at a time and the one evicted first is the one it finished
    /// longest ago. Evicting a finished partition costs no extra file. The extra file shows up
    /// where ordering cannot reach, such as a merge-on-read input distributed by data file
    /// rather than by partition.
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

    fn new_partition_writer_state(
        &mut self,
        partition_dir: &str,
    ) -> Result<PartitionWriterState, String> {
        if self.config.variant_shredding.enabled {
            Ok(PartitionWriterState::Pending {
                batches: Vec::new(),
                num_rows: 0,
            })
        } else {
            let (writer, file_path) =
                self.new_arrow_writer(self.config.table_schema.clone(), partition_dir)?;
            Ok(PartitionWriterState::Open {
                writer: Box::new(writer),
                file_path,
                variant_shredding_plan: None,
            })
        }
    }

    async fn write_partition_state(
        &mut self,
        state: PartitionWriterState,
        batch: RecordBatch,
        partition_dir: &str,
    ) -> Result<PartitionWriterState, String> {
        match state {
            PartitionWriterState::Pending {
                mut batches,
                mut num_rows,
            } => {
                num_rows += batch.num_rows();
                batches.push(batch);
                if num_rows >= self.config.variant_shredding.inference_buffer_size.max(1) {
                    self.open_and_write_pending_batches(batches, partition_dir)
                        .await
                } else {
                    Ok(PartitionWriterState::Pending { batches, num_rows })
                }
            }
            PartitionWriterState::Open {
                mut writer,
                file_path,
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
                    file_path,
                    variant_shredding_plan,
                })
            }
        }
    }

    async fn open_and_write_pending_batches(
        &mut self,
        batches: Vec<RecordBatch>,
        partition_dir: &str,
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
        let (mut writer, file_path) = self.new_arrow_writer(schema, partition_dir)?;
        for batch in physical_batches {
            writer.write_batch(&batch).await?;
        }
        Ok(PartitionWriterState::Open {
            writer: Box::new(writer),
            file_path,
            variant_shredding_plan: plan,
        })
    }

    fn new_arrow_writer(
        &mut self,
        schema: SchemaRef,
        partition_dir: &str,
    ) -> Result<(ArrowParquetWriter<ParquetObjectSink>, String), String> {
        for (i, f) in schema.fields().iter().enumerate() {
            log::trace!(
                "iceberg.table_writer.writer_schema: field[{}]='{}' type={:?} field_id_meta={:?}",
                i,
                f.name(),
                f.data_type(),
                f.metadata().get(PARQUET_FIELD_ID_META_KEY)
            );
        }
        let (relative, path) = self.generator.next_data_path(Some(partition_dir))?;
        let file_path = self
            .data_url
            .join(&format!("./{relative}"))
            .map_err(|error| error.to_string())?
            .to_string();
        let output = ParquetObjectSink::new(BufWriter::new(Arc::clone(&self.store), path));
        let writer = ArrowParquetWriter::try_new(
            schema.as_ref(),
            self.config.writer_properties.clone(),
            output,
        )?;
        Ok((writer, file_path))
    }

    async fn finish_partition_state(
        &mut self,
        state: PartitionWriterState,
        partition_dir: &str,
    ) -> Result<(ArrowParquetWriter<ParquetObjectSink>, String), String> {
        match state {
            PartitionWriterState::Pending { batches, .. } => {
                let PartitionWriterState::Open {
                    writer, file_path, ..
                } = self
                    .open_and_write_pending_batches(batches, partition_dir)
                    .await?
                else {
                    return Err("failed to open pending Iceberg partition writer".to_string());
                };
                Ok((*writer, file_path))
            }
            PartitionWriterState::Open {
                writer, file_path, ..
            } => Ok((*writer, file_path)),
        }
    }

    async fn flush_partition(
        &mut self,
        state: PartitionWriterState,
        partition_dir: &str,
        partition_values: Vec<Option<Literal>>,
    ) -> Result<(), String> {
        let (writer, file_path) = self.finish_partition_state(state, partition_dir).await?;
        let (_, meta) = writer.close().await?;
        let mut df = DataFileWriter::new(self.partition_spec_id, file_path, partition_values)
            .finish_with_schema(meta, self.config.iceberg_schema.as_ref())?
            .data_file;
        df.sort_order_id = self.config.sort_order_id;
        self.written.push(df);
        Ok(())
    }

    pub async fn close(mut self) -> Result<Vec<DataFile>, String> {
        // FIXME: Retain ownership of uploaded files across partial close failures and task cancellation.
        // Cleanup must wait for the job outcome so retries can safely reuse successful task output.
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
                    schema_fields.push(Arc::new(
                        field
                            .as_ref()
                            .clone()
                            .with_data_type(array.data_type().clone()),
                    ));
                    columns.push(array);
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

        crate::schema_defaults::missing_write_value(iceberg_field.as_ref(), num_rows)
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use object_store::memory::InMemory;

    use super::*;
    use crate::spec::Transform;
    use crate::spec::partition::{UnboundPartitionField, UnboundPartitionSpec};
    use crate::spec::types::{NestedField, PrimitiveType, Type};

    /// Large enough that nothing rolls on size, so the tests only observe the open-writer bound.
    const NO_FILE_ROLLING: u64 = u64::MAX;

    fn arrow_schema() -> SchemaRef {
        let field = |name: &str, data_type: DataType, id: &str| {
            Field::new(name, data_type, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                id.to_string(),
            )]))
        };
        Arc::new(ArrowSchema::new(vec![
            field("id", DataType::Int32, "1"),
            field("part", DataType::Utf8, "2"),
        ]))
    }

    fn partitioned_writer(max_open_writers: usize) -> Result<IcebergTableWriter, String> {
        let table_schema = arrow_schema();
        let iceberg_schema = IcebergSchema::builder()
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
            table_schema,
            writer_properties: parquet::file::properties::WriterProperties::builder().build(),
            target_file_size_bytes: NO_FILE_ROLLING,
            sort_order_id: None,
            iceberg_schema: Arc::new(iceberg_schema),
            partition_spec: UnboundPartitionSpec {
                fields: vec![UnboundPartitionField {
                    source_id: 2,
                    name: "part".to_string(),
                    transform: Transform::Identity,
                }],
            },
            variant_shredding: Default::default(),
            max_open_writers,
        };
        Ok(IcebergTableWriter::new(
            Arc::new(InMemory::new()),
            ObjectPath::from("t"),
            config,
            0,
            Url::parse("memory:///t/data/").map_err(|error| error.to_string())?,
        ))
    }

    fn row(schema: &SchemaRef, id: i32, part: &str) -> Result<RecordBatch, String> {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![id])),
                Arc::new(StringArray::from(vec![part.to_string()])),
            ],
        )
        .map_err(|error| error.to_string())
    }

    /// One writer per partition tuple used to be held open until `close`, so memory grew
    /// with the partition cardinality a single task happened to see. Writing more
    /// partitions than the bound must finish the oldest rather than accumulate.
    #[test]
    fn open_partition_writers_stay_within_the_configured_bound() -> Result<(), String> {
        const MAX_OPEN: usize = 2;
        const PARTITIONS: i32 = 6;

        futures::executor::block_on(async {
            let mut writer = partitioned_writer(MAX_OPEN)?;
            let schema = writer.config.table_schema.clone();

            // Each batch carries one distinct partition, so an unbounded map would hold six.
            for partition in 0..PARTITIONS {
                writer
                    .write(&row(&schema, partition, &format!("p{partition}"))?)
                    .await?;
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
    /// the common grouped case is not split into one file per batch.
    #[test]
    fn a_partition_under_the_bound_is_not_evicted_between_batches() -> Result<(), String> {
        futures::executor::block_on(async {
            let mut writer = partitioned_writer(4)?;
            let schema = writer.config.table_schema.clone();

            for id in 0..3 {
                writer.write(&row(&schema, id, "p0")?).await?;
            }
            assert_eq!(writer.writers.len(), 1);

            let files = writer.close().await?;
            assert_eq!(
                files.len(),
                1,
                "three batches of one partition must not split"
            );
            Ok(())
        })
    }
}
