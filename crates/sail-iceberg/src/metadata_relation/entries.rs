// SPDX-FileCopyrightText: 2026 HOIST IT B.V.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, Int32Array, Int64Array, RecordBatch, RecordBatchOptions, StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::Result;

use super::files::{self, FilesProjection};
use crate::spec::{ManifestEntry, ManifestStatus, TableMetadata};

/// Entry lineage wraps the same typed file fields as the files relation.
pub(crate) fn schema(metadata: &TableMetadata) -> Result<SchemaRef> {
    // Unsupported evolution must fail rather than omit historical partition values.
    let current = metadata
        .current_schema()
        .ok_or_else(|| datafusion::common::plan_datafusion_err!("Missing current schema"))?;
    for spec in &metadata.partition_specs {
        for field in spec.fields() {
            let source_id = field
                .source_id()
                .map_err(datafusion::common::DataFusionError::Plan)?;
            if current.field_by_id(source_id).is_none() {
                return datafusion::common::not_impl_err!(
                    "Partition spec {} references source field {source_id} absent from the current schema",
                    spec.spec_id()
                );
            }
        }
    }
    let files = files::schema(metadata)?;
    let mut fields = vec![
        Field::new("status", DataType::Int32, false),
        Field::new("snapshot_id", DataType::Int64, true),
        Field::new("sequence_number", DataType::Int64, true),
        Field::new("file_sequence_number", DataType::Int64, true),
        Field::new(
            "data_file",
            DataType::Struct(
                files
                    .fields()
                    .iter()
                    .filter(|field| field.name() != "readable_metrics")
                    .cloned()
                    .collect(),
            ),
            false,
        ),
    ];
    fields.push(files.field_with_name("readable_metrics")?.clone());
    Ok(Arc::new(Schema::new(fields)))
}

pub(crate) struct EntriesProjection<'a> {
    schema: SchemaRef,
    files: Option<FilesProjection<'a>>,
    metrics: Option<FilesProjection<'a>>,
}

impl<'a> EntriesProjection<'a> {
    pub(crate) fn try_new(metadata: &'a TableMetadata, projection: &[usize]) -> Result<Self> {
        let schema = Arc::new(schema(metadata)?.project(projection)?);
        let file_schema = files::schema(metadata)?;
        let files = if schema.column_with_name("data_file").is_some() {
            let columns = file_schema
                .fields()
                .iter()
                .enumerate()
                .filter_map(|(index, field)| (field.name() != "readable_metrics").then_some(index))
                .collect::<Vec<_>>();
            Some(FilesProjection::try_new(metadata, &columns)?)
        } else {
            None
        };
        let metrics = if schema.column_with_name("readable_metrics").is_some() {
            Some(FilesProjection::try_new(
                metadata,
                &[file_schema.index_of("readable_metrics")?],
            )?)
        } else {
            None
        };
        Ok(Self {
            schema,
            files,
            metrics,
        })
    }

    pub(crate) fn batch(&self, entries: &[ManifestEntry]) -> Result<RecordBatch> {
        if entries.is_empty() {
            return Ok(RecordBatch::new_empty(self.schema.clone()));
        }
        let files = if self.files.is_some() || self.metrics.is_some() {
            entries
                .iter()
                .map(|entry| entry.data_file.clone())
                .collect::<Vec<_>>()
        } else {
            vec![]
        };
        let data_file = self
            .files
            .as_ref()
            .map(|projection| {
                let batch = projection.batch(&files)?;
                Ok::<ArrayRef, datafusion::common::DataFusionError>(Arc::new(StructArray::try_new(
                    batch.schema().fields().clone(),
                    batch.columns().to_vec(),
                    None,
                )?))
            })
            .transpose()?;
        let metrics = self
            .metrics
            .as_ref()
            .map(|projection| projection.batch(&files))
            .transpose()?;
        let mut columns: Vec<ArrayRef> = vec![];
        for field in self.schema.fields() {
            columns.push(match field.name().as_str() {
                "status" => Arc::new(Int32Array::from_iter_values(entries.iter().map(|entry| {
                    match entry.status {
                        ManifestStatus::Existing => 0,
                        ManifestStatus::Added => 1,
                        ManifestStatus::Deleted => 2,
                    }
                }))),
                "snapshot_id" => Arc::new(Int64Array::from_iter(
                    entries.iter().map(|entry| entry.snapshot_id),
                )),
                "sequence_number" => Arc::new(Int64Array::from_iter(
                    entries.iter().map(|entry| entry.sequence_number),
                )),
                "file_sequence_number" => Arc::new(Int64Array::from_iter(
                    entries.iter().map(|entry| entry.file_sequence_number),
                )),
                "data_file" => data_file.as_ref().cloned().ok_or_else(|| {
                    datafusion::common::internal_datafusion_err!("Missing entry file projection")
                })?,
                "readable_metrics" => metrics
                    .as_ref()
                    .map(|batch| batch.column(0).clone())
                    .ok_or_else(|| {
                        datafusion::common::internal_datafusion_err!(
                            "Missing entry metrics projection"
                        )
                    })?,
                name => {
                    return datafusion::common::internal_err!(
                        "Unknown manifest entry field: {name}"
                    );
                }
            });
        }
        RecordBatch::try_new_with_options(
            self.schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(entries.len())),
        )
        .map_err(Into::into)
    }
}
