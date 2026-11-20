// SPDX-FileCopyrightText: 2026 HOIST IT B.V.
// SPDX-License-Identifier: Apache-2.0

#![expect(clippy::expect_used)]

use std::sync::Arc;

use bytes::Bytes;
use datafusion::arrow::array::{
    Array, Int32Array, Int64Array, StringArray, StructArray, TimestampMicrosecondArray,
};
use datafusion::common::Result;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::TryStreamExt;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use object_store::path::Path;
use sail_common_datafusion::datasource::{OptionLayer, SourceInfo};
use sail_common_datafusion::lakerelation::{
    LakeRelation, LakeRelationAccess, LakeRelationProvider, LakeRelationResolution,
    LakeRelationTimeTravel,
};
use url::Url;

use crate::io::StoreContext;
use crate::lake_source::IcebergLakeSource;
use crate::physical_plan::IcebergMetadataRelationExec;
use crate::spec::manifest_list::ManifestListWriter;
use crate::spec::types::values::{Literal, PrimitiveLiteral};
use crate::spec::types::{NestedField, PrimitiveType, Type};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, ManifestContentType, ManifestEntry, ManifestMetadata,
    ManifestStatus, ManifestWriter, PartitionSpec, Schema,
};

async fn fixture(all_statuses: bool) -> Result<(SessionContext, Arc<InMemory>, Url)> {
    let url = Url::parse("memory://bucket/table").expect("URL");
    let store = Arc::new(InMemory::new());
    let storage = StoreContext::new(store.clone(), &url)?;
    let mut metadata = super::tests::table_metadata();
    metadata.location = url.to_string();
    metadata.schemas = vec![
        Schema::builder()
            .with_schema_id(7)
            .with_fields(vec![
                NestedField::optional(1, "week", Type::Primitive(PrimitiveType::Timestamp)).into(),
            ])
            .build()
            .expect("schema"),
    ];
    let spec = PartitionSpec::builder()
        .with_spec_id(7)
        .add_field_with_id(1, 1001, "week", crate::spec::Transform::Identity)
        .build();
    metadata.partition_specs = vec![spec.clone()];
    metadata.default_spec_id = 7;
    for snapshot in &mut metadata.snapshots {
        let id = snapshot.snapshot_id();
        snapshot.manifest_list = format!("metadata/list-{id}.avro");
        let mut list = ManifestListWriter::new();
        let rows = if all_statuses { 3 } else { 1 };
        for index in 0..rows {
            let (status, content, manifest_content) = match index {
                0 => (
                    ManifestStatus::Added,
                    DataContentType::Data,
                    ManifestContentType::Data,
                ),
                1 => (
                    ManifestStatus::Existing,
                    DataContentType::PositionDeletes,
                    ManifestContentType::Deletes,
                ),
                _ => (
                    ManifestStatus::Deleted,
                    DataContentType::EqualityDeletes,
                    ManifestContentType::Deletes,
                ),
            };
            let mut file: DataFile = serde_json::from_value(serde_json::json!({
                "content": "DATA", "file_path": format!("data/{id}-{index}.parquet"),
                "file_format": "PARQUET", "partition": [], "record_count": 10,
                "file_size_in_bytes": 100, "partition_spec_id": 7
            }))
            .expect("file");
            file.content = content;
            file.partition = vec![Some(Literal::Primitive(PrimitiveLiteral::Long(
                id * 1_000_000,
            )))];
            let mut writer = ManifestWriter::new(
                Some(id),
                None,
                ManifestMetadata::new(
                    Arc::new(metadata.schemas[0].clone()),
                    7,
                    spec.clone(),
                    FormatVersion::V2,
                    manifest_content,
                ),
            );
            writer.add_entry(ManifestEntry::new(
                status,
                Some(id),
                Some(index + 1),
                Some(index + 2),
                file,
            ));
            let path = format!("metadata/m-{id}-{index}.avro");
            list.append(
                writer
                    .clone()
                    .into_manifest_file(path.clone(), 3, id)
                    .expect("descriptor"),
            );
            storage
                .prefixed
                .put(
                    &Path::from(path),
                    Bytes::from(writer.finish().to_avro_bytes_v2().expect("manifest")).into(),
                )
                .await?;
        }
        storage
            .prefixed
            .put(
                &Path::from(snapshot.manifest_list()),
                Bytes::from(list.to_bytes(FormatVersion::V2).expect("list")).into(),
            )
            .await?;
    }
    storage
        .prefixed
        .put(
            &Path::from("metadata/v1.metadata.json"),
            Bytes::from(metadata.to_json().expect("metadata")).into(),
        )
        .await?;
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(2)
            .with_batch_size(1),
    );
    ctx.register_object_store(&url, store.clone());
    Ok((ctx, store, url))
}

async fn plan(ctx: &SessionContext, url: &Url, id: i64) -> Result<Arc<dyn ExecutionPlan>> {
    let source = IcebergLakeSource
        .create_relation(
            &ctx.state(),
            SourceInfo {
                paths: vec![url.to_string()],
                options: vec![OptionLayer::OptionList {
                    items: vec![
                        (
                            "metadata-location".into(),
                            "memory://bucket/table/metadata/v1.metadata.json".into(),
                        ),
                        ("snapshot-id".into(), id.to_string()),
                    ],
                }],
                lakehouse_table: None,
                schema: None,
                constraints: Default::default(),
                partition_by: vec![],
                bucket_by: None,
                sort_order: vec![],
                read_case_sensitive: false,
            },
            LakeRelation::new(
                "entries",
                LakeRelationAccess::MetadataRead,
                LakeRelationTimeTravel::Supported,
            ),
        )
        .await?;
    let provider = datafusion::datasource::source_as_provider(&source)?;
    provider.scan(&ctx.state(), None, &[], None).await
}

#[tokio::test]
async fn entries_snapshot_selection_survives_worker_serialization() -> Result<()> {
    let (ctx, store, url) = fixture(false).await?;
    assert!(matches!(IcebergLakeSource.resolve_relation("entries"),
        LakeRelationResolution::Supported(relation) if relation.time_travel() == LakeRelationTimeTravel::Supported));
    let plan = plan(&ctx, &url, 10).await?;
    let exec = plan
        .downcast_ref::<IcebergMetadataRelationExec>()
        .expect("metadata exec");
    let exec = IcebergMetadataRelationExec::try_from_serialized(
        exec.original_schema().clone(),
        &exec.serialized_scan()?,
    )?;
    // A worker has its own context. It must not rediscover the current snapshot.
    let worker = SessionContext::new();
    worker.register_object_store(&url, store);
    let batches: Vec<_> = exec.execute(0, worker.task_ctx())?.try_collect().await?;
    assert_eq!(batches.len(), 1);
    let files = batches[0]
        .column_by_name("data_file")
        .expect("files")
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct");
    let paths = files
        .column_by_name("file_path")
        .expect("paths")
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("strings");
    assert_eq!(paths.value(0), "data/10-0.parquet");
    let missing = self::plan(&ctx, &url, 999)
        .await
        .expect_err("missing snapshot must fail");
    assert!(missing.to_string().contains("999"));
    Ok(())
}

#[tokio::test]
async fn entries_preserve_status_content_lineage_and_typed_partition() -> Result<()> {
    let (ctx, _, url) = fixture(true).await?;
    let plan = plan(&ctx, &url, 20).await?;
    let exec = plan
        .downcast_ref::<IcebergMetadataRelationExec>()
        .expect("metadata exec");
    let mut rows = vec![];
    for partition in 0..plan.output_partitioning().partition_count() {
        let batches: Vec<_> = plan
            .execute(partition, ctx.task_ctx())?
            .try_collect()
            .await?;
        for batch in batches {
            let integer = |name: &str| {
                batch
                    .column_by_name(name)
                    .expect("column")
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("int64")
                    .value(0)
            };
            let status = batch
                .column_by_name("status")
                .expect("status")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("int32")
                .value(0);
            let files = batch
                .column_by_name("data_file")
                .expect("files")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("struct");
            let content = files
                .column_by_name("content")
                .expect("content")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("int32")
                .value(0);
            let spec = files
                .column_by_name("spec_id")
                .expect("spec")
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("int32")
                .value(0);
            let parts = files
                .column_by_name("partition")
                .expect("partition")
                .as_any()
                .downcast_ref::<StructArray>()
                .expect("struct");
            let week = parts
                .column_by_name("week")
                .expect("week")
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("timestamp")
                .value(0);
            rows.push((
                status,
                integer("snapshot_id"),
                integer("sequence_number"),
                integer("file_sequence_number"),
                content,
                spec,
                week,
            ));
        }
    }
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (0, 20, 2, 3, 1, 7, 20_000_000),
            (1, 20, 1, 2, 0, 7, 20_000_000),
            (2, 20, 3, 4, 2, 7, 20_000_000)
        ]
    );
    let mut scan: serde_json::Value =
        serde_json::from_str(&exec.serialized_scan()?).expect("scan JSON");
    scan["projection"] = serde_json::json!([0, 1]);
    scan["limit"] = serde_json::json!(1);
    let projected = IcebergMetadataRelationExec::try_from_serialized(
        exec.original_schema().clone(),
        &scan.to_string(),
    )?;
    let batches: Vec<_> = projected.execute(0, ctx.task_ctx())?.try_collect().await?;
    assert_eq!((batches[0].num_columns(), batches[0].num_rows()), (2, 1));
    scan["projection"] = serde_json::json!([]);
    scan["limit"] = serde_json::Value::Null;
    let projected = IcebergMetadataRelationExec::try_from_serialized(
        exec.original_schema().clone(),
        &scan.to_string(),
    )?;
    let mut count = 0;
    for partition in 0..projected.properties().partitioning.partition_count() {
        let batches: Vec<_> = projected
            .execute(partition, ctx.task_ctx())?
            .try_collect()
            .await?;
        for batch in batches {
            assert_eq!(batch.num_columns(), 0);
            count += batch.num_rows();
        }
    }
    assert_eq!(count, 3);
    let mut unsupported = super::tests::table_metadata();
    unsupported.partition_specs = vec![
        PartitionSpec::builder()
            .with_spec_id(7)
            .add_field_with_id(99, 1001, "missing_source", crate::spec::Transform::Identity)
            .build(),
    ];
    assert!(
        super::entries::schema(&unsupported).is_err(),
        "a partition value must not silently disappear from the schema"
    );
    Ok(())
}
