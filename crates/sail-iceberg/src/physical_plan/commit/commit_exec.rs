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

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::{Distribution, EquivalenceProperties};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};
use datafusion_common::{DataFusionError, Result, internal_err, plan_err};
use futures::StreamExt;
use futures::stream::once;
use object_store::ObjectStoreExt;
use sail_catalog::error::CatalogError;
use sail_common_datafusion::catalog::LakehouseExecutionContext;
use url::Url;

use crate::catalog_support::commit::{
    CatalogCommitOutcome, CatalogTableInfo, IcebergCatalogCommitCoordinator,
    IcebergCatalogCommitMode, catalog_requirements, table_metadata_location,
};
use crate::io::{StoreContext, load_manifest, load_manifest_list};
use crate::operations::bootstrap::{
    BootstrapResult, NewTableMetadataStyle, PersistStrategy, bootstrap_first_snapshot,
    bootstrap_new_table_with_style, prepare_bootstrap_snapshot,
};
use crate::operations::helpers::format_version_for_schema;
use crate::operations::{
    PreparedSnapshotCommit, SnapshotProducer, SnapshotUpdateKind, Transaction,
};
use crate::physical_plan::action_schema::{CommitMeta, decode_actions_and_meta_from_batch};
use crate::physical_plan::commit::IcebergCommitInfo;
use crate::snapshot_properties::validate_snapshot_properties;
use crate::spec::catalog::TableUpdate;
use crate::spec::manifest::ManifestStatus;
use crate::spec::metadata::table_metadata::SnapshotLog;
use crate::spec::partition::{UnboundPartitionField, UnboundPartitionSpec};
use crate::spec::snapshots::MAIN_BRANCH;
use crate::spec::{
    DataContentType, DataFile, FormatVersion, PartitionSpec, Schema as IcebergSchema,
    TableMetadata, TableRequirement,
};
use crate::table::metadata_loader::{
    encode_metadata_file, load_metadata_file_bytes, metadata_file_extension_from_properties,
    metadata_file_version_from_path, metadata_location_to_object_path_string, write_version_hint,
};
use crate::table_format::metadata_location_from_properties;
use crate::utils::get_object_store_from_context;
use crate::utils::metadata::metadata_files_for_version;
const MAX_COMMIT_RETRIES: usize = 5;

#[derive(Debug)]
struct PublicationStateUnknownError(String);

impl std::fmt::Display for PublicationStateUnknownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Iceberg publication state is unknown: {}", self.0)
    }
}

impl std::error::Error for PublicationStateUnknownError {}

fn publication_state_unknown_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::External(Box::new(PublicationStateUnknownError(message.into())))
}

fn operation_with_cleanup_failure(
    operation_error: DataFusionError,
    cleanup_error: DataFusionError,
) -> DataFusionError {
    DataFusionError::Execution(format!(
        "{operation_error}; uncommitted Iceberg artifact cleanup also failed: {cleanup_error}"
    ))
}

async fn cleanup_uncommitted_task_files(
    store_ctx: &StoreContext,
    file_paths: &[String],
) -> Result<()> {
    let mut base_paths = Vec::new();
    let mut prefixed_paths = Vec::new();
    let mut failures = Vec::new();
    for file_path in file_paths.iter().collect::<BTreeSet<_>>() {
        let (store, path) = match store_ctx.resolve(file_path) {
            Ok(resolved) => resolved,
            Err(error) => {
                failures.push(format!("failed to resolve {file_path}: {error}"));
                continue;
            }
        };
        if Arc::ptr_eq(store, &store_ctx.base) {
            base_paths.push(path);
        } else {
            prefixed_paths.push(path);
        }
    }

    if let Err(error) = delete_task_files(&store_ctx.base, base_paths).await {
        failures.push(error.to_string());
    }
    if let Err(error) = delete_task_files(&store_ctx.prefixed, prefixed_paths).await {
        failures.push(error.to_string());
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(DataFusionError::Execution(format!(
            "failed to clean uncommitted Iceberg task files: {}",
            failures.join("; ")
        )))
    }
}

async fn delete_task_files(
    store: &Arc<dyn object_store::ObjectStore>,
    paths: Vec<object_store::path::Path>,
) -> Result<()> {
    let locations = futures::stream::iter(paths.into_iter().map(Ok));
    let mut deletions = store.delete_stream(Box::pin(locations));
    let mut failures = Vec::new();
    while let Some(result) = deletions.next().await {
        match result {
            Ok(_) | Err(object_store::Error::NotFound { .. }) => {}
            Err(error) => failures.push(error.to_string()),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(DataFusionError::Execution(failures.join("; ")))
    }
}

async fn cleanup_uncommitted_snapshot_attempt(
    store_ctx: &StoreContext,
    metadata_file: &str,
    prepared_snapshot: PreparedSnapshotCommit,
    task_files_may_be_committed: &mut bool,
) -> Result<()> {
    let metadata_path = object_store::path::Path::from(metadata_file);
    match store_ctx.prefixed.delete(&metadata_path).await {
        Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
        Err(error) => {
            *task_files_may_be_committed = true;
            drop(prepared_snapshot);
            return Err(publication_state_unknown_error(format!(
                "failed to remove candidate metadata {metadata_file}; metadata, manifests and task data were retained: {error}"
            )));
        }
    }
    prepared_snapshot.cleanup().await
}

async fn error_after_snapshot_attempt_cleanup(
    store_ctx: &StoreContext,
    metadata_file: &str,
    prepared_snapshot: PreparedSnapshotCommit,
    task_files_may_be_committed: &mut bool,
    operation_error: DataFusionError,
) -> DataFusionError {
    match cleanup_uncommitted_snapshot_attempt(
        store_ctx,
        metadata_file,
        prepared_snapshot,
        task_files_may_be_committed,
    )
    .await
    {
        Ok(()) => operation_error,
        Err(cleanup_error) if publication_state_is_unknown(&cleanup_error) => cleanup_error,
        Err(cleanup_error) => operation_with_cleanup_failure(operation_error, cleanup_error),
    }
}

async fn cleanup_uncommitted_bootstrap_attempt(
    bootstrap_result: BootstrapResult,
    store_ctx: &StoreContext,
    task_files_may_be_committed: &mut bool,
) -> Result<()> {
    match bootstrap_result.cleanup(store_ctx).await {
        Err(error) if publication_state_is_unknown(&error) => {
            *task_files_may_be_committed = true;
            Err(error)
        }
        result => result,
    }
}

async fn error_after_bootstrap_cleanup(
    bootstrap_result: BootstrapResult,
    store_ctx: &StoreContext,
    task_files_may_be_committed: &mut bool,
    operation_error: DataFusionError,
) -> DataFusionError {
    match cleanup_uncommitted_bootstrap_attempt(
        bootstrap_result,
        store_ctx,
        task_files_may_be_committed,
    )
    .await
    {
        Ok(()) => operation_error,
        Err(cleanup_error) if publication_state_is_unknown(&cleanup_error) => cleanup_error,
        Err(cleanup_error) => operation_with_cleanup_failure(operation_error, cleanup_error),
    }
}

fn is_retryable_publication_conflict(error: &DataFusionError) -> bool {
    let DataFusionError::External(source) = error else {
        return false;
    };
    source.downcast_ref::<CatalogError>().is_some_and(|error| {
        matches!(
            error,
            CatalogError::Conflict(_) | CatalogError::StaleMetadata(_)
        )
    }) || source
        .downcast_ref::<object_store::Error>()
        .is_some_and(|error| matches!(error, object_store::Error::AlreadyExists { .. }))
}

fn publication_state_is_unknown(error: &DataFusionError) -> bool {
    matches!(
        error,
        DataFusionError::Execution(message)
            if message.contains("Iceberg catalog commit state is unknown")
    ) || matches!(
        error,
        DataFusionError::External(source)
            if source.downcast_ref::<PublicationStateUnknownError>().is_some()
                || source
                    .downcast_ref::<CatalogError>()
                    .is_some_and(|error| matches!(error, CatalogError::CommitStateUnknown(_)))
    )
}

fn task_file_paths(data_files: &[DataFile], delete_files: &[DataFile]) -> Vec<String> {
    data_files
        .iter()
        .chain(delete_files.iter())
        .map(|file| file.file_path.clone())
        .collect()
}

fn commit_count_batch(schema: SchemaRef, row_count: u64) -> Result<RecordBatch> {
    let row_count = i64::try_from(row_count).map_err(|e| {
        DataFusionError::Execution(format!("Iceberg commit row count overflow: {e}"))
    })?;
    let array = Arc::new(Int64Array::from(vec![row_count]));
    RecordBatch::try_new(schema, vec![array])
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

fn expected_snapshot_requirement(
    expected_snapshot_id: Option<Option<i64>>,
) -> Option<TableRequirement> {
    expected_snapshot_id.map(|snapshot_id| TableRequirement::RefSnapshotIdMatch {
        r#ref: MAIN_BRANCH.to_string(),
        snapshot_id,
    })
}

fn caller_expected_snapshot_requirement(
    expected_snapshot_id: Option<i64>,
) -> Option<TableRequirement> {
    expected_snapshot_id.map(|snapshot_id| TableRequirement::RefSnapshotIdMatch {
        r#ref: MAIN_BRANCH.to_string(),
        snapshot_id: Some(snapshot_id),
    })
}

fn validate_scoped_overwrite_format(
    snapshot_update_kind: SnapshotUpdateKind,
    format_version: FormatVersion,
) -> Result<()> {
    if matches!(
        snapshot_update_kind,
        SnapshotUpdateKind::CopyOnWrite | SnapshotUpdateKind::CopyOnWriteDelete
    ) && matches!(format_version, FormatVersion::V3)
    {
        return Err(DataFusionError::NotImplemented(
            "Iceberg v3 scoped overwrite is not supported until row lineage is preserved"
                .to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub struct IcebergCommitExec {
    input: Arc<dyn ExecutionPlan>,
    table_url: Url,
    lakehouse_table: Option<LakehouseExecutionContext>,
    snapshot_update_kind: SnapshotUpdateKind,
    expected_snapshot_id: Option<Option<i64>>,
    caller_expected_snapshot_id: Option<i64>,
    snapshot_properties: Vec<(String, String)>,
    removed_data_file_paths: Vec<String>,
    dynamic_partition_overwrite: bool,
    cache: Arc<PlanProperties>,
}

impl IcebergCommitExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        table_url: Url,
        lakehouse_table: Option<LakehouseExecutionContext>,
        snapshot_update_kind: SnapshotUpdateKind,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "count",
            DataType::Int64,
            true,
        )]));
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            input,
            table_url,
            lakehouse_table,
            snapshot_update_kind,
            expected_snapshot_id: None,
            caller_expected_snapshot_id: None,
            snapshot_properties: Vec::new(),
            removed_data_file_paths: Vec::new(),
            dynamic_partition_overwrite: false,
            cache,
        }
    }

    pub fn with_expected_snapshot_id(mut self, expected_snapshot_id: Option<Option<i64>>) -> Self {
        self.expected_snapshot_id = expected_snapshot_id;
        self
    }

    pub fn with_caller_expected_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.caller_expected_snapshot_id = snapshot_id;
        self
    }

    pub fn with_snapshot_properties(mut self, properties: Vec<(String, String)>) -> Self {
        self.snapshot_properties = properties;
        self
    }

    pub fn with_removed_data_file_paths(mut self, paths: Vec<String>) -> Self {
        self.removed_data_file_paths = paths;
        self
    }

    pub fn with_dynamic_partition_overwrite(mut self, enabled: bool) -> Self {
        self.dynamic_partition_overwrite = enabled;
        self
    }

    pub fn removed_data_file_paths(&self) -> &[String] {
        &self.removed_data_file_paths
    }

    pub fn dynamic_partition_overwrite(&self) -> bool {
        self.dynamic_partition_overwrite
    }

    pub fn table_url(&self) -> &Url {
        &self.table_url
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn lakehouse_table(&self) -> Option<&LakehouseExecutionContext> {
        self.lakehouse_table.as_ref()
    }

    pub fn snapshot_update_kind(&self) -> SnapshotUpdateKind {
        self.snapshot_update_kind
    }

    pub fn expected_snapshot_id(&self) -> Option<Option<i64>> {
        self.expected_snapshot_id
    }

    pub fn caller_expected_snapshot_id(&self) -> Option<i64> {
        self.caller_expected_snapshot_id
    }

    pub fn snapshot_properties(&self) -> &[(String, String)] {
        &self.snapshot_properties
    }

    pub fn validate_publication_controls(&self) -> Result<()> {
        validate_snapshot_properties(&self.snapshot_properties)?;
        if self
            .caller_expected_snapshot_id
            .is_some_and(|snapshot_id| snapshot_id <= 0)
        {
            return plan_err!("Iceberg expected snapshot ID must be a positive integer");
        }
        Ok(())
    }

    fn apply_schema_update(table_meta: &mut TableMetadata, new_schema: IcebergSchema) {
        let schema_id = new_schema.schema_id();
        let highest_field_id = new_schema.highest_field_id();

        let mut replaced = false;
        for schema in table_meta.schemas.iter_mut() {
            if schema.schema_id() == schema_id {
                *schema = new_schema.clone();
                replaced = true;
                break;
            }
        }
        if !replaced {
            table_meta.schemas.push(new_schema.clone());
        }

        table_meta.current_schema_id = schema_id;
        table_meta.last_column_id = table_meta.last_column_id.max(highest_field_id);
        table_meta.format_version = table_meta
            .format_version
            .max(format_version_for_schema(&new_schema));
    }

    fn apply_partition_spec_update(table_meta: &mut TableMetadata, new_spec: PartitionSpec) {
        let spec_id = new_spec.spec_id();
        let mut replaced = false;
        for spec in table_meta.partition_specs.iter_mut() {
            if spec.spec_id() == spec_id {
                *spec = new_spec.clone();
                replaced = true;
                break;
            }
        }
        if !replaced {
            table_meta.partition_specs.push(new_spec.clone());
        }
        table_meta.default_spec_id = spec_id;
        table_meta.last_partition_id = table_meta
            .last_partition_id
            .max(new_spec.last_assigned_field_id());
    }

    fn validate_requirements(
        table_meta: Option<&TableMetadata>,
        requirements: &[TableRequirement],
    ) -> Result<()> {
        for requirement in requirements {
            match requirement {
                TableRequirement::NotExist => {
                    if table_meta.is_some() {
                        return Err(DataFusionError::Plan(
                            "Iceberg table already exists but commit asserted non-existence."
                                .to_string(),
                        ));
                    }
                }
                TableRequirement::LastAssignedFieldIdMatch {
                    last_assigned_field_id,
                } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating field id requirement"
                                .to_string(),
                        )
                    })?;
                    if &meta.last_column_id != last_assigned_field_id {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: expected last assigned field id {} but found {}. Reload table metadata and retry.",
                            last_assigned_field_id, meta.last_column_id
                        )));
                    }
                }
                TableRequirement::CurrentSchemaIdMatch { current_schema_id } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating schema requirement"
                                .to_string(),
                        )
                    })?;
                    if &meta.current_schema_id != current_schema_id {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: expected current schema id {} but found {}. Reload table metadata and retry.",
                            current_schema_id, meta.current_schema_id
                        )));
                    }
                }
                TableRequirement::RefSnapshotIdMatch {
                    r#ref: reference,
                    snapshot_id,
                } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating snapshot requirement"
                                .to_string(),
                        )
                    })?;
                    let actual = if reference == MAIN_BRANCH {
                        meta.current_snapshot_id
                    } else {
                        meta.refs
                            .get(reference)
                            .map(|ref_entry| ref_entry.snapshot_id)
                    };
                    let actual = actual.filter(|snapshot_id| *snapshot_id >= 0);
                    if &actual != snapshot_id {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: reference '{}' expected snapshot {:?} but found {:?}",
                            reference, snapshot_id, actual
                        )));
                    }
                }
                TableRequirement::UuidMatch { uuid } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating UUID requirement"
                                .to_string(),
                        )
                    })?;
                    if meta.table_uuid.as_ref() != Some(uuid) {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: expected table UUID {} but found {:?}. Reload table metadata and retry.",
                            uuid, meta.table_uuid
                        )));
                    }
                }
                TableRequirement::LastAssignedPartitionIdMatch {
                    last_assigned_partition_id,
                } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating partition id requirement"
                                .to_string(),
                        )
                    })?;
                    if &meta.last_partition_id != last_assigned_partition_id {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: expected last assigned partition id {} but found {}. Reload table metadata and retry.",
                            last_assigned_partition_id, meta.last_partition_id
                        )));
                    }
                }
                TableRequirement::DefaultSpecIdMatch { default_spec_id } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating partition spec requirement"
                                .to_string(),
                        )
                    })?;
                    if &meta.default_spec_id != default_spec_id {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: expected default partition spec id {} but found {}. Reload table metadata and retry.",
                            default_spec_id, meta.default_spec_id
                        )));
                    }
                }
                TableRequirement::DefaultSortOrderIdMatch {
                    default_sort_order_id,
                } => {
                    let meta = table_meta.ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata missing while validating sort order requirement"
                                .to_string(),
                        )
                    })?;
                    let actual = meta.default_sort_order_id.map(i64::from).unwrap_or(0);
                    if &actual != default_sort_order_id {
                        return Err(DataFusionError::Plan(format!(
                            "Iceberg commit failed: expected default sort order id {} but found {}. Reload table metadata and retry.",
                            default_sort_order_id, actual
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn unbound_partition_spec(spec: &PartitionSpec) -> UnboundPartitionSpec {
        let fields = spec
            .fields()
            .iter()
            .map(|field| UnboundPartitionField {
                source_id: field.source_id,
                name: field.name.clone(),
                transform: field.transform,
            })
            .collect();
        UnboundPartitionSpec { fields }
    }

    async fn load_authoritative_table_metadata(
        context: &Arc<TaskContext>,
        object_store: &Arc<dyn object_store::ObjectStore>,
        table_url: &Url,
        catalog_table: Option<&[String]>,
    ) -> Result<(String, TableMetadata)> {
        let metadata_file = if let Some(catalog_table) = catalog_table {
            let metadata_location = Self::load_catalog_metadata_location(context, catalog_table)
                .await?
                .ok_or_else(|| {
                    DataFusionError::Plan(
                        "catalog-authoritative Iceberg table has no metadata location".to_string(),
                    )
                })?;
            metadata_location_to_object_path_string(&metadata_location)?
        } else {
            crate::table::find_latest_metadata_file(object_store, table_url).await?
        };
        let bytes = load_metadata_file_bytes(object_store, &metadata_file).await?;
        let table_metadata = TableMetadata::from_json(&bytes)
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        Ok((metadata_file, table_metadata))
    }

    async fn load_catalog_table_info(
        context: &Arc<TaskContext>,
        catalog_table: &[String],
    ) -> Result<CatalogTableInfo> {
        IcebergCatalogCommitCoordinator::load_table_info(context.as_ref(), catalog_table).await
    }

    async fn load_catalog_metadata_location(
        context: &Arc<TaskContext>,
        catalog_table: &[String],
    ) -> Result<Option<String>> {
        IcebergCatalogCommitCoordinator::load_metadata_location(context.as_ref(), catalog_table)
            .await
    }

    async fn try_commit_to_catalog(
        context: &Arc<TaskContext>,
        catalog_table: &[String],
        lakehouse_table: &LakehouseExecutionContext,
        requirements: Vec<TableRequirement>,
        updates: Vec<TableUpdate>,
    ) -> Result<CatalogCommitOutcome> {
        IcebergCatalogCommitCoordinator::new(context.as_ref(), catalog_table)
            .commit(lakehouse_table, requirements, updates)
            .await
    }

    async fn update_catalog_metadata_location(
        context: &Arc<TaskContext>,
        catalog_table: &[String],
        existing_properties: &[(String, String)],
        previous_metadata_location: Option<&str>,
        new_metadata_location: &str,
    ) -> Result<()> {
        IcebergCatalogCommitCoordinator::new(context.as_ref(), catalog_table)
            .update_metadata_location(
                existing_properties,
                previous_metadata_location,
                new_metadata_location,
            )
            .await
    }

    fn table_metadata_location(table_url: &Url, metadata_file: &str) -> Result<String> {
        table_metadata_location(table_url, metadata_file)
    }

    async fn current_live_data_files(
        store_ctx: &StoreContext,
        table_metadata: &TableMetadata,
    ) -> Result<Vec<DataFile>> {
        let Some(snapshot) = table_metadata.current_snapshot() else {
            return Ok(Vec::new());
        };
        let manifest_list = load_manifest_list(store_ctx, snapshot.manifest_list()).await?;
        let mut live_data_files = Vec::new();
        for manifest_file in manifest_list.entries() {
            let manifest = load_manifest(store_ctx, &manifest_file.manifest_path).await?;
            for entry in manifest.entries().iter().filter(|entry| {
                matches!(
                    entry.status,
                    ManifestStatus::Added | ManifestStatus::Existing
                )
            }) {
                if !matches!(entry.data_file.content, DataContentType::Data) {
                    return Err(DataFusionError::Plan(
                        "copy-on-write scoped overwrite is not supported for Iceberg tables with active delete files"
                            .to_string(),
                    ));
                }
                let mut file = entry.data_file.clone();
                file.partition_spec_id = manifest_file.partition_spec_id;
                live_data_files.push(file);
            }
        }
        Ok(live_data_files)
    }

    fn dynamic_partition_overwrite_paths(
        added_data_files: &[DataFile],
        live_data_files: &[DataFile],
        default_spec: &PartitionSpec,
    ) -> Result<Vec<String>> {
        if added_data_files.is_empty() {
            return Ok(Vec::new());
        }
        let default_spec_id = default_spec.spec_id();
        let partition_field_count = default_spec.fields().len();
        if added_data_files.iter().any(|file| {
            file.partition_spec_id != default_spec_id
                || file.partition.len() != partition_field_count
        }) {
            return Err(DataFusionError::Plan(format!(
                "dynamic partition overwrite produced files that do not match the default Iceberg partition spec {default_spec_id}"
            )));
        }
        if live_data_files.iter().any(|file| {
            file.partition_spec_id != default_spec_id
                || file.partition.len() != partition_field_count
        }) {
            return Err(DataFusionError::NotImplemented(
                "dynamic partition overwrite is not supported for Iceberg tables with incomparable live partition specs"
                    .to_string(),
            ));
        }
        let touched_partitions = added_data_files
            .iter()
            .map(|file| file.partition.clone())
            .collect::<HashSet<_>>();
        let mut paths = live_data_files
            .iter()
            .filter(|file| touched_partitions.contains(&file.partition))
            .map(|file| file.file_path.clone())
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();
        Ok(paths)
    }

    fn merge_writer_commit_meta(
        accumulated: &mut Option<CommitMeta>,
        mut incoming: CommitMeta,
    ) -> Result<()> {
        let Some(existing) = accumulated.as_mut() else {
            *accumulated = Some(incoming);
            return Ok(());
        };

        let incoming_row_count = incoming.row_count;
        incoming.row_count = existing.row_count;
        if existing != &incoming {
            return Err(DataFusionError::Internal(
                "inconsistent commit_meta actions from Iceberg writer partitions".to_string(),
            ));
        }
        existing.row_count = existing
            .row_count
            .checked_add(incoming_row_count)
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "Iceberg writer row count overflow across partitions".to_string(),
                )
            })?;
        Ok(())
    }
}

#[async_trait]
impl ExecutionPlan for IcebergCommitExec {
    fn name(&self) -> &'static str {
        "IcebergCommitExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("IcebergCommitExec requires exactly one child");
        }
        Ok(Arc::new(
            Self::new(
                Arc::clone(&children[0]),
                self.table_url.clone(),
                self.lakehouse_table.clone(),
                self.snapshot_update_kind,
            )
            .with_expected_snapshot_id(self.expected_snapshot_id)
            .with_caller_expected_snapshot_id(self.caller_expected_snapshot_id)
            .with_snapshot_properties(self.snapshot_properties.clone())
            .with_removed_data_file_paths(self.removed_data_file_paths.clone())
            .with_dynamic_partition_overwrite(self.dynamic_partition_overwrite),
        ))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!("IcebergCommitExec can only be executed in a single partition");
        }

        let input_partitions = self.input.output_partitioning().partition_count();
        if input_partitions != 1 {
            return internal_err!(
                "IcebergCommitExec requires exactly one input partition, got {input_partitions}"
            );
        }

        self.validate_publication_controls()?;
        let input_stream = self.input.execute(0, Arc::clone(&context))?;

        let table_url = self.table_url.clone();
        let lakehouse_table = self.lakehouse_table.clone();
        let snapshot_update_kind = self.snapshot_update_kind;
        let expected_snapshot_id = self.expected_snapshot_id;
        let caller_expected_snapshot_id = self.caller_expected_snapshot_id;
        let snapshot_properties = self.snapshot_properties.clone();
        let planned_removed_data_file_paths = self.removed_data_file_paths.clone();
        let dynamic_partition_overwrite = self.dynamic_partition_overwrite;
        let schema = self.schema();
        let future = async move {
            let object_store = get_object_store_from_context(&context, &table_url)?;
            let store_ctx = StoreContext::new(object_store.clone(), &table_url)?;

            // Read writer result as Arrow-native action batches (may be empty for IgnoreIfExists).
            let mut data = input_stream;
            let mut added_data_files: Vec<DataFile> = Vec::new();
            let mut added_delete_files: Vec<DataFile> = Vec::new();
            let mut commit_meta = None;
            let input_result: Result<()> = async {
                while let Some(batch_result) = data.next().await {
                    let batch = batch_result?;
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let (adds, deletes, meta) = decode_actions_and_meta_from_batch(&batch)?;
                    added_data_files.extend(adds);
                    added_delete_files.extend(deletes);
                    if let Some(meta) = meta {
                        Self::merge_writer_commit_meta(&mut commit_meta, meta)?;
                    }
                }
                Ok(())
            }
            .await;
            if let Err(error) = input_result {
                let paths = task_file_paths(&added_data_files, &added_delete_files);
                return match cleanup_uncommitted_task_files(&store_ctx, &paths).await {
                    Ok(()) => Err(error),
                    Err(cleanup_error) => Err(operation_with_cleanup_failure(error, cleanup_error)),
                };
            }

            let task_file_paths = task_file_paths(&added_data_files, &added_delete_files);
            let mut task_files_may_be_committed = false;
            let commit_result: Result<RecordBatch> = async {

            // No-op path (e.g. IgnoreIfExists on existing table): no rows, no meta.
            if commit_meta.is_none() && added_data_files.is_empty() && added_delete_files.is_empty()
            {
                if expected_snapshot_id.is_some()
                    || caller_expected_snapshot_id.is_some()
                    || !snapshot_properties.is_empty()
                {
                    return Err(DataFusionError::Internal(
                        "controlled Iceberg mutation produced no commit metadata".to_string(),
                    ));
                }
                return commit_count_batch(schema, 0);
            }

            let commit_meta = commit_meta.ok_or_else(|| {
                DataFusionError::Internal(
                    "missing commit_meta action from writer output".to_string(),
                )
            })?;

            let mut commit_info = IcebergCommitInfo {
                table_uri: commit_meta.table_uri,
                row_count: commit_meta.row_count,
                data_files: added_data_files,
                delete_files: added_delete_files,
                manifest_path: String::new(),
                manifest_list_path: String::new(),
                updates: vec![],
                requirements: commit_meta.requirements,
                table_properties: commit_meta.table_properties,
                snapshot_properties,
                caller_expected_snapshot_id,
                lakehouse_table: commit_meta.lakehouse_table.or(lakehouse_table),
                snapshot_update_kind,
                schema: commit_meta.schema,
                partition_spec: commit_meta.partition_spec,
            };
            if let Some(requirement) = expected_snapshot_requirement(expected_snapshot_id)
                && !commit_info.requirements.contains(&requirement)
            {
                commit_info.requirements.push(requirement);
            }
            if let Some(requirement) = caller_expected_snapshot_requirement(
                commit_info.caller_expected_snapshot_id,
            ) && !commit_info.requirements.contains(&requirement)
            {
                commit_info.requirements.push(requirement);
            }
            if !matches!(
                snapshot_update_kind,
                SnapshotUpdateKind::CopyOnWrite | SnapshotUpdateKind::CopyOnWriteDelete
            ) && (dynamic_partition_overwrite || !planned_removed_data_file_paths.is_empty())
            {
                return Err(DataFusionError::Internal(
                    "scoped overwrite requires a copy-on-write snapshot update".to_string(),
                ));
            }
            if dynamic_partition_overwrite && !planned_removed_data_file_paths.is_empty() {
                return Err(DataFusionError::Internal(
                    "dynamic partition overwrite cannot carry planned removal paths".to_string(),
                ));
            }
            let validation_only_copy_on_write = matches!(
                snapshot_update_kind,
                SnapshotUpdateKind::CopyOnWrite | SnapshotUpdateKind::CopyOnWriteDelete
            ) && commit_info.data_files.is_empty()
                && planned_removed_data_file_paths.is_empty();
            let mut removed_data_file_paths = planned_removed_data_file_paths;

            let catalog_table = commit_info
                .lakehouse_table
                .as_ref()
                .map(|context| context.catalog_table().to_vec());
            let CatalogTableInfo {
                metadata_location: catalog_status_metadata_location,
                is_catalog_managed_iceberg_table: is_catalog_status_managed_iceberg_table,
            } = match catalog_table.as_ref() {
                Some(table) => Self::load_catalog_table_info(&context, table).await?,
                None => CatalogTableInfo::default(),
            };
            let catalog_table_info = CatalogTableInfo {
                metadata_location: catalog_status_metadata_location,
                is_catalog_managed_iceberg_table: is_catalog_status_managed_iceberg_table,
            };
            let catalog_commit_mode = IcebergCatalogCommitMode::resolve(
                commit_info.lakehouse_table.as_ref(),
                &catalog_table_info,
                &commit_info.table_properties,
            );
            let table_property_metadata_location =
                metadata_location_from_properties(&commit_info.table_properties);
            let catalog_recorded_metadata_location = table_property_metadata_location
                .clone()
                .or(catalog_table_info.metadata_location.clone());
            let catalog_metadata_location = catalog_commit_mode
                .uses_catalog_metadata()
                .then(|| catalog_recorded_metadata_location.clone())
                .flatten();

            // Managed external catalogs use the authoritative metadata-location.
            // Path tables may record metadata-location in the session catalog for display, but
            // their current state is discovered from the metadata directory and version hint.
            let latest_meta_res = match catalog_metadata_location.as_deref() {
                Some(location) => Ok(metadata_location_to_object_path_string(location)?),
                None => crate::table::find_latest_metadata_file(&object_store, &table_url).await,
            };
            let catalog_metadata_table = catalog_table
                .as_ref()
                .filter(|_| catalog_commit_mode.uses_catalog_metadata());
            let catalog_commit_table = catalog_table
                .as_ref()
                .filter(|_| catalog_commit_mode.uses_catalog_commit());
            let catalog_metadata_update_table = catalog_table
                .as_ref()
                .filter(|_| catalog_commit_mode.uses_metadata_location_update());
            let catalog_registered_metadata_table = catalog_table
                .as_ref()
                .filter(|_| matches!(catalog_commit_mode, IcebergCatalogCommitMode::Filesystem));
            log::debug!(
                "Iceberg catalog commit context: table={:?}, metadata_location={:?}, mode={:?}",
                catalog_table,
                catalog_metadata_location,
                catalog_commit_mode
            );

            if latest_meta_res.is_err() {
                Self::validate_requirements(None, &commit_info.requirements)?;
                if validation_only_copy_on_write {
                    return commit_count_batch(schema, 0);
                }
                if let Some(catalog_table) = catalog_metadata_update_table {
                    let bootstrap_result = bootstrap_new_table_with_style(
                        &table_url,
                        &store_ctx,
                        &commit_info,
                        NewTableMetadataStyle::Uuid,
                    )
                    .await?;
                    let new_metadata_location = match Self::table_metadata_location(
                        &table_url,
                        &bootstrap_result.metadata_file,
                    ) {
                        Ok(location) => location,
                        Err(error) => {
                            return Err(
                                error_after_bootstrap_cleanup(
                                    bootstrap_result,
                                    &store_ctx,
                                    &mut task_files_may_be_committed,
                                    error,
                                )
                                .await,
                            );
                        }
                    };
                    match Self::update_catalog_metadata_location(
                        &context,
                        catalog_table,
                        &commit_info.table_properties,
                        None,
                        &new_metadata_location,
                    )
                    .await
                    {
                        Ok(()) => {
                            task_files_may_be_committed = true;
                        }
                        Err(error) if is_retryable_publication_conflict(&error) => {
                            return Err(
                                error_after_bootstrap_cleanup(
                                    bootstrap_result,
                                    &store_ctx,
                                    &mut task_files_may_be_committed,
                                    commit_conflict_error(),
                                )
                                .await,
                            );
                        }
                        Err(error) if publication_state_is_unknown(&error) => {
                            task_files_may_be_committed = true;
                            return Err(error);
                        }
                        Err(error) => {
                            return Err(
                                error_after_bootstrap_cleanup(
                                    bootstrap_result,
                                    &store_ctx,
                                    &mut task_files_may_be_committed,
                                    error,
                                )
                                .await,
                            );
                        }
                    }
                } else if catalog_commit_mode.uses_catalog_commit() {
                    return Err(DataFusionError::Plan(
                        "missing Iceberg metadata for catalog-authoritative table".to_string(),
                    ));
                } else {
                    // Bootstrap a new table using the Hadoop/path-table convention.
                    let bootstrap_result = bootstrap_new_table_with_style(
                        &table_url,
                        &store_ctx,
                        &commit_info,
                        NewTableMetadataStyle::Hadoop,
                    )
                    .await?;
                    task_files_may_be_committed = true;
                    if let Some(catalog_table) = catalog_registered_metadata_table {
                        let new_metadata_location = Self::table_metadata_location(
                            &table_url,
                            &bootstrap_result.metadata_file,
                        )?;
                        Self::update_catalog_metadata_location(
                            &context,
                            catalog_table,
                            &commit_info.table_properties,
                            catalog_recorded_metadata_location.as_deref(),
                            &new_metadata_location,
                        )
                        .await?;
                    }
                }

                return commit_count_batch(schema, commit_info.row_count);
            }

            let initial_latest_meta = latest_meta_res?;

            let mut attempt = 0;
            loop {
                attempt += 1;
                let catalog_metadata_location = if attempt == 1 {
                    catalog_metadata_location.clone()
                } else if let Some(catalog_table) = catalog_metadata_table {
                    Self::load_catalog_metadata_location(&context, catalog_table).await?
                } else {
                    catalog_metadata_location.clone()
                };
                let latest_meta = if attempt == 1 {
                    initial_latest_meta.clone()
                } else if let Some(location) = catalog_metadata_location.as_deref() {
                    metadata_location_to_object_path_string(location)?
                } else {
                    crate::table::find_latest_metadata_file(&object_store, &table_url).await?
                };

                let bytes = load_metadata_file_bytes(&object_store, &latest_meta).await?;
                let mut table_meta = TableMetadata::from_json(&bytes)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                Self::validate_requirements(Some(&table_meta), &commit_info.requirements)?;
                validate_scoped_overwrite_format(
                    snapshot_update_kind,
                    table_meta.format_version,
                )?;
                if validation_only_copy_on_write {
                    let (authoritative_metadata_file, authoritative_table_meta) =
                        Self::load_authoritative_table_metadata(
                            &context,
                            &object_store,
                            &table_url,
                            catalog_metadata_table.map(Vec::as_slice),
                        )
                        .await?;
                    Self::validate_requirements(
                        Some(&authoritative_table_meta),
                        &commit_info.requirements,
                    )?;
                    if authoritative_metadata_file != latest_meta {
                        if attempt >= MAX_COMMIT_RETRIES {
                            return Err(commit_conflict_error());
                        }
                        continue;
                    }
                    return commit_count_batch(schema, 0);
                }
                if dynamic_partition_overwrite {
                    let default_spec = table_meta.default_partition_spec().ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata has no default partition spec".to_string(),
                        )
                    })?;
                    let live_data_files =
                        Self::current_live_data_files(&store_ctx, &table_meta).await?;
                    removed_data_file_paths = Self::dynamic_partition_overwrite_paths(
                        &commit_info.data_files,
                        &live_data_files,
                        default_spec,
                    )?;
                }
                let original_format_version = table_meta.format_version;
                let mut metadata_updates = Vec::new();
                if let Some(new_schema) = commit_info.schema.clone() {
                    let schema_id = new_schema.schema_id();
                    let should_add_schema = !table_meta
                        .schemas
                        .iter()
                        .any(|schema| schema.schema_id() == schema_id);
                    let should_set_current_schema = table_meta.current_schema_id != schema_id;
                    Self::apply_schema_update(&mut table_meta, new_schema.clone());
                    if should_add_schema {
                        metadata_updates.push(TableUpdate::AddSchema {
                            schema: Box::new(new_schema),
                        });
                    }
                    if should_set_current_schema {
                        metadata_updates.push(TableUpdate::SetCurrentSchema { schema_id });
                    }
                }
                let mut partition_spec_for_commit = table_meta
                    .default_partition_spec()
                    .cloned()
                    .unwrap_or_else(PartitionSpec::unpartitioned_spec);
                if let Some(new_spec) = commit_info.partition_spec.clone() {
                    let spec = if new_spec.spec_id() == 0 && table_meta.default_spec_id != 0 {
                        new_spec.with_spec_id(table_meta.default_spec_id)
                    } else {
                        new_spec
                    };
                    let spec_id = spec.spec_id();
                    let should_add_spec = !table_meta
                        .partition_specs
                        .iter()
                        .any(|partition_spec| partition_spec.spec_id() == spec_id);
                    let should_set_default_spec = table_meta.default_spec_id != spec_id;
                    Self::apply_partition_spec_update(&mut table_meta, spec.clone());
                    partition_spec_for_commit = spec;
                    if should_add_spec {
                        metadata_updates.push(TableUpdate::AddSpec {
                            spec: Self::unbound_partition_spec(&partition_spec_for_commit),
                        });
                    }
                    if should_set_default_spec {
                        metadata_updates.push(TableUpdate::SetDefaultSpec { spec_id });
                    }
                }
                let maybe_snapshot = table_meta.current_snapshot().cloned();
                let schema_iceberg = table_meta.current_schema().cloned().ok_or_else(|| {
                    DataFusionError::Plan("No current schema in table metadata".to_string())
                })?;
                table_meta.format_version = table_meta
                    .format_version
                    .max(format_version_for_schema(&schema_iceberg));
                if table_meta.format_version > original_format_version {
                    metadata_updates.insert(
                        0,
                        TableUpdate::UpgradeFormatVersion {
                            format_version: table_meta.format_version,
                        },
                    );
                }
                // Schema evolution can raise a loaded v2 table to v3. Re-check the effective
                // version before row-lineage resolution or any manifest work.
                validate_scoped_overwrite_format(
                    snapshot_update_kind,
                    table_meta.format_version,
                )?;
                let row_lineage_start_row_id = table_meta.row_lineage_start_row_id();

                // If metadata exists but there is no current snapshot (e.g. from a CREATE TABLE),
                // bootstrap the first snapshot as a normal metadata version.
                if maybe_snapshot.is_none() {
                    let mut catalog_fallback_table = catalog_metadata_update_table;
                    if let Some(catalog_table) = catalog_commit_table {
                        let prepared_snapshot = prepare_bootstrap_snapshot(
                            &table_url,
                            &store_ctx,
                            &commit_info,
                            &table_meta,
                        )
                        .await?;
                        let action_requirements = prepared_snapshot
                            .action_commit()
                            .requirements()
                            .to_vec();
                        if let Err(error) =
                            Self::validate_requirements(Some(&table_meta), &action_requirements)
                        {
                            return Err(prepared_snapshot.cleanup_after(error).await);
                        }
                        let requirements = catalog_requirements(
                            &table_meta,
                            &commit_info.requirements,
                            &action_requirements,
                        );
                        let mut updates = metadata_updates.clone();
                        updates.extend(prepared_snapshot.action_commit().updates().to_vec());
                        let lakehouse_table = match commit_info.lakehouse_table.as_ref() {
                            Some(table) => table,
                            None => {
                                let error = DataFusionError::Internal(
                                    "missing lakehouse context for Iceberg catalog commit"
                                        .to_string(),
                                );
                                return Err(prepared_snapshot.cleanup_after(error).await);
                            }
                        };
                        let catalog_outcome = match Self::try_commit_to_catalog(
                            &context,
                            catalog_table,
                            lakehouse_table,
                            requirements,
                            updates,
                        )
                        .await
                        {
                            Ok(outcome) => outcome,
                            Err(error) if publication_state_is_unknown(&error) => {
                                task_files_may_be_committed = true;
                                return Err(error);
                            }
                            Err(error) => {
                                return Err(prepared_snapshot.cleanup_after(error).await);
                            }
                        };
                        match catalog_outcome {
                            CatalogCommitOutcome::Committed(committed) => {
                                task_files_may_be_committed = true;
                                if let Some(metadata_location) = committed.metadata_location() {
                                    log::debug!(
                                        "Iceberg catalog commit returned metadata-location={metadata_location}"
                                    );
                                }
                                if committed.payload().is_some() {
                                    log::trace!("Iceberg catalog commit returned a payload");
                                }
                                return commit_count_batch(schema, commit_info.row_count);
                            }
                            CatalogCommitOutcome::NotSupported => {
                                if matches!(
                                    catalog_commit_mode,
                                    IcebergCatalogCommitMode::CompatibilityCatalogCommit
                                ) {
                                    prepared_snapshot.cleanup().await?;
                                    catalog_fallback_table = Some(catalog_table);
                                } else {
                                    let error = DataFusionError::Plan(
                                        "Iceberg catalog commit is not supported by the resolved catalog authority"
                                            .to_string(),
                                    );
                                    return Err(prepared_snapshot.cleanup_after(error).await);
                                }
                            }
                            CatalogCommitOutcome::Conflict => {
                                prepared_snapshot.cleanup().await?;
                                if attempt >= MAX_COMMIT_RETRIES {
                                    return Err(commit_conflict_error());
                                }
                                continue;
                            }
                        }
                    }

                    let (authoritative_metadata_file, authoritative_table_meta) =
                        Self::load_authoritative_table_metadata(
                            &context,
                            &object_store,
                            &table_url,
                            catalog_metadata_table.map(Vec::as_slice),
                        )
                        .await?;
                    Self::validate_requirements(
                        Some(&authoritative_table_meta),
                        &commit_info.requirements,
                    )?;
                    if authoritative_metadata_file != latest_meta {
                        if attempt >= MAX_COMMIT_RETRIES {
                            return Err(commit_conflict_error());
                        }
                        continue;
                    }

                    let persist_strategy = if catalog_fallback_table.is_some() {
                        PersistStrategy::NewUuidVersion
                    } else {
                        PersistStrategy::NewVersion
                    };
                    let previous_metadata_file = catalog_fallback_table
                        .is_some()
                        .then_some(catalog_metadata_location.as_deref())
                        .flatten();
                    let bootstrap_result = match bootstrap_first_snapshot(
                        &table_url,
                        &store_ctx,
                        &commit_info,
                        table_meta,
                        &latest_meta,
                        previous_metadata_file,
                        persist_strategy,
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(error) if is_retryable_publication_conflict(&error) => {
                            if attempt >= MAX_COMMIT_RETRIES {
                                return Err(commit_conflict_error());
                            }
                            continue;
                        }
                        Err(error) => return Err(error),
                    };
                    if let (Some(catalog_table), Some(previous_metadata_location)) =
                        (catalog_fallback_table, catalog_metadata_location.as_deref())
                    {
                        let new_metadata_location = match Self::table_metadata_location(
                            &table_url,
                            &bootstrap_result.metadata_file,
                        ) {
                            Ok(location) => location,
                            Err(error) => {
                                return Err(
                                    error_after_bootstrap_cleanup(
                                        bootstrap_result,
                                        &store_ctx,
                                        &mut task_files_may_be_committed,
                                        error,
                                    )
                                    .await,
                                );
                            }
                        };
                        match Self::update_catalog_metadata_location(
                            &context,
                            catalog_table,
                            &commit_info.table_properties,
                            Some(previous_metadata_location),
                            &new_metadata_location,
                        )
                        .await
                        {
                            Ok(()) => {
                                task_files_may_be_committed = true;
                            }
                            Err(error) if is_retryable_publication_conflict(&error) => {
                                cleanup_uncommitted_bootstrap_attempt(
                                    bootstrap_result,
                                    &store_ctx,
                                    &mut task_files_may_be_committed,
                                )
                                .await?;
                                if attempt >= MAX_COMMIT_RETRIES {
                                    return Err(commit_conflict_error());
                                }
                                continue;
                            }
                            Err(error) if publication_state_is_unknown(&error) => {
                                task_files_may_be_committed = true;
                                return Err(error);
                            }
                            Err(error) => {
                                return Err(
                                    error_after_bootstrap_cleanup(
                                        bootstrap_result,
                                        &store_ctx,
                                        &mut task_files_may_be_committed,
                                        error,
                                    )
                                    .await,
                                );
                            }
                        }
                    } else {
                        task_files_may_be_committed = true;
                        if let Some(catalog_table) = catalog_registered_metadata_table {
                            let new_metadata_location = Self::table_metadata_location(
                                &table_url,
                                &bootstrap_result.metadata_file,
                            )?;
                            Self::update_catalog_metadata_location(
                                &context,
                                catalog_table,
                                &commit_info.table_properties,
                                catalog_recorded_metadata_location.as_deref(),
                                &new_metadata_location,
                            )
                            .await?;
                        }
                    }

                    return commit_count_batch(schema, commit_info.row_count);
                }

                let snapshot = maybe_snapshot.ok_or_else(|| {
                    DataFusionError::Plan("No current snapshot in table metadata".to_string())
                })?;

                let current_version = metadata_file_version_from_path(&latest_meta).unwrap_or(0);
                let next_version = current_version + 1;

                let existing_for_next =
                    metadata_files_for_version(&store_ctx, next_version).await?;
                if !existing_for_next.is_empty() {
                    log::warn!(
                        "Detected existing metadata files for version {}: {:?}. Retrying attempt {}",
                        next_version,
                        existing_for_next,
                        attempt
                    );
                    if attempt >= MAX_COMMIT_RETRIES {
                        return Err(commit_conflict_error());
                    }
                    continue;
                }

                // Build transaction and action based on the snapshot update algorithm.
                let tx = Transaction::new(
                    table_url.to_string(),
                    snapshot,
                    table_meta.last_sequence_number,
                );
                let manifest_meta = tx.default_manifest_metadata(
                    &schema_iceberg,
                    &partition_spec_for_commit,
                    table_meta.format_version,
                );
                let prepared_snapshot = SnapshotProducer::new(
                    &tx,
                    commit_info.data_files.clone(),
                    Some(store_ctx.clone()),
                    Some(manifest_meta),
                )
                .with_snapshot_properties(commit_info.snapshot_properties.clone())
                .with_added_delete_files(commit_info.delete_files.clone())
                .with_removed_data_file_paths(removed_data_file_paths.clone())
                .with_partition_specs(table_meta.partition_specs.clone())
                .with_row_lineage_start_row_id(row_lineage_start_row_id)
                .prepare(commit_info.snapshot_update_kind)
                .await
                .map_err(DataFusionError::Execution)?;

                // Apply updates (only handle the ones we emit: AddSnapshot, SetSnapshotRef)
                let action_requirements = prepared_snapshot.action_commit().requirements().to_vec();
                if let Err(error) =
                    Self::validate_requirements(Some(&table_meta), &action_requirements)
                {
                    return Err(prepared_snapshot.cleanup_after(error).await);
                }
                let action_updates = prepared_snapshot.action_commit().updates().to_vec();
                if let Some(catalog_table) = catalog_commit_table {
                    let requirements = catalog_requirements(
                        &table_meta,
                        &commit_info.requirements,
                        &action_requirements,
                    );
                    let mut updates = metadata_updates.clone();
                    updates.extend(action_updates.clone());
                    let lakehouse_table = match commit_info.lakehouse_table.as_ref() {
                        Some(table) => table,
                        None => {
                            let error = DataFusionError::Internal(
                                "missing lakehouse context for Iceberg catalog commit".to_string(),
                            );
                            return Err(prepared_snapshot.cleanup_after(error).await);
                        }
                    };
                    let catalog_outcome = match Self::try_commit_to_catalog(
                        &context,
                        catalog_table,
                        lakehouse_table,
                        requirements,
                        updates,
                    )
                    .await
                    {
                        Ok(outcome) => outcome,
                        Err(error) if publication_state_is_unknown(&error) => {
                            task_files_may_be_committed = true;
                            return Err(error);
                        }
                        Err(error) => {
                            return Err(prepared_snapshot.cleanup_after(error).await);
                        }
                    };
                    match catalog_outcome {
                        CatalogCommitOutcome::Committed(committed) => {
                            task_files_may_be_committed = true;
                            if let Some(metadata_location) = committed.metadata_location() {
                                log::debug!(
                                    "Iceberg catalog commit returned metadata-location={metadata_location}"
                                );
                            }
                            if committed.payload().is_some() {
                                log::trace!("Iceberg catalog commit returned a payload");
                            }
                            return commit_count_batch(schema, commit_info.row_count);
                        }
                        CatalogCommitOutcome::NotSupported
                            if matches!(
                                catalog_commit_mode,
                                IcebergCatalogCommitMode::CompatibilityCatalogCommit
                            ) => {}
                        CatalogCommitOutcome::NotSupported => {
                            let error = DataFusionError::Plan(
                                "Iceberg catalog commit is not supported by the resolved catalog authority"
                                    .to_string(),
                            );
                            return Err(prepared_snapshot.cleanup_after(error).await);
                        }
                        CatalogCommitOutcome::Conflict => {
                            prepared_snapshot.cleanup().await?;
                            if attempt >= MAX_COMMIT_RETRIES {
                                return Err(commit_conflict_error());
                            }
                            continue;
                        }
                    }
                }

                log::trace!("commit_exec: applying updates: {:?}", action_updates);
                let mut newest_snapshot_seq: Option<i64> = None;
                let mut newest_snapshot_added_rows: Option<i64> = None;
                let previous_metadata_timestamp_ms = table_meta.last_updated_ms;
                let timestamp_ms = crate::utils::timestamp::monotonic_timestamp_ms();
                for upd in action_updates {
                    match upd {
                        TableUpdate::AddSnapshot { snapshot } => {
                            newest_snapshot_seq = Some(snapshot.sequence_number());
                            newest_snapshot_added_rows = snapshot.added_rows;
                            table_meta.snapshots.push(snapshot.clone());
                            table_meta.current_snapshot_id = Some(snapshot.snapshot_id());
                            table_meta.snapshot_log.push(SnapshotLog {
                                timestamp_ms,
                                snapshot_id: snapshot.snapshot_id(),
                            });
                        }
                        TableUpdate::SetSnapshotRef {
                            ref_name,
                            reference,
                        } => {
                            table_meta.refs.insert(ref_name, reference);
                        }
                        _ => {}
                    }
                }
                if let Some(seq) = newest_snapshot_seq
                    && seq > table_meta.last_sequence_number
                {
                    table_meta.last_sequence_number = seq;
                }
                table_meta.last_updated_ms = timestamp_ms;
                if let Some(added_rows) = newest_snapshot_added_rows {
                    table_meta.advance_next_row_id(added_rows);
                }

                // Add metadata_log entry referencing previous metadata file
                table_meta
                    .metadata_log
                    .push(crate::spec::metadata::table_metadata::MetadataLog {
                        timestamp_ms: previous_metadata_timestamp_ms,
                        metadata_file: catalog_metadata_location
                            .clone()
                            .unwrap_or_else(|| latest_meta.clone()),
                    });

                let use_uuid_metadata_file = catalog_metadata_update_table.is_some();
                let encoded_metadata: Result<(String, String, Vec<u8>)> = (|| {
                    let metadata_json = table_meta
                        .to_json()
                        .map_err(|error| DataFusionError::External(Box::new(error)))?;
                    let file_extension =
                        metadata_file_extension_from_properties(&table_meta.properties)?;
                    let metadata_file = if use_uuid_metadata_file {
                        format!(
                            "metadata/{next_version:05}-{}{file_extension}",
                            uuid::Uuid::new_v4()
                        )
                    } else {
                        format!("metadata/v{next_version}{file_extension}")
                    };
                    let metadata_location =
                        Self::table_metadata_location(&table_url, &metadata_file)?;
                    let metadata_bytes = encode_metadata_file(&metadata_file, &metadata_json)
                        .map_err(|error| DataFusionError::External(Box::new(error)))?;
                    Ok((metadata_file, metadata_location, metadata_bytes))
                })();
                let (metadata_file, metadata_location, metadata_bytes) = match encoded_metadata {
                    Ok(encoded_metadata) => encoded_metadata,
                    Err(error) => {
                        return Err(prepared_snapshot.cleanup_after(error).await);
                    }
                };

                log::trace!(
                    "Writing metadata: {} snapshot_id={:?} table_url={}",
                    metadata_file,
                    table_meta.current_snapshot_id,
                    table_url
                );

                let authoritative_base = Self::load_authoritative_table_metadata(
                    &context,
                    &object_store,
                    &table_url,
                    catalog_metadata_table.map(Vec::as_slice),
                )
                .await;
                let (authoritative_metadata_file, authoritative_table_meta) =
                    match authoritative_base {
                        Ok(base) => base,
                        Err(error) => {
                            return Err(prepared_snapshot.cleanup_after(error).await);
                        }
                    };
                if let Err(error) = Self::validate_requirements(
                    Some(&authoritative_table_meta),
                    &commit_info.requirements,
                ) {
                    return Err(prepared_snapshot.cleanup_after(error).await);
                }
                if authoritative_metadata_file != latest_meta {
                    prepared_snapshot.cleanup().await?;
                    if attempt >= MAX_COMMIT_RETRIES {
                        return Err(commit_conflict_error());
                    }
                    continue;
                }

                let metadata_path = object_store::path::Path::from(metadata_file.as_str());
                let put_opts = object_store::PutOptions {
                    mode: object_store::PutMode::Create,
                    ..Default::default()
                };
                let payload = object_store::PutPayload::from(Bytes::from(metadata_bytes));
                match store_ctx
                    .prefixed
                    .put_opts(&metadata_path, payload, put_opts)
                    .await
                {
                    Ok(_) => {}
                    Err(object_store::Error::AlreadyExists { .. }) => {
                        log::warn!(
                            "Metadata file {} already exists for version {}. Retrying attempt {}",
                            metadata_file,
                            next_version,
                            attempt
                        );
                        prepared_snapshot.cleanup().await?;
                        if attempt >= MAX_COMMIT_RETRIES {
                            return Err(commit_conflict_error());
                        }
                        continue;
                    }
                    Err(error) => {
                        let error = DataFusionError::External(Box::new(error));
                        return Err(prepared_snapshot.cleanup_after(error).await);
                    }
                }
                let version_files = match metadata_files_for_version(&store_ctx, next_version).await {
                    Ok(files) => files,
                    Err(error) => {
                        return Err(
                            error_after_snapshot_attempt_cleanup(
                                &store_ctx,
                                &metadata_file,
                                prepared_snapshot,
                                &mut task_files_may_be_committed,
                                error,
                            )
                            .await,
                        );
                    }
                };
                let conflict_after_write = version_files.iter().any(|path| path != &metadata_file);
                if conflict_after_write {
                    log::warn!(
                        "Concurrent metadata writes detected for version {}: {:?}. Retrying attempt {}",
                        next_version,
                        version_files,
                        attempt
                    );
                    cleanup_uncommitted_snapshot_attempt(
                        &store_ctx,
                        &metadata_file,
                        prepared_snapshot,
                        &mut task_files_may_be_committed,
                    )
                    .await?;
                    if attempt >= MAX_COMMIT_RETRIES {
                        return Err(commit_conflict_error());
                    }
                    continue;
                }

                if let Some(catalog_table) = catalog_metadata_update_table {
                    match Self::update_catalog_metadata_location(
                        &context,
                        catalog_table,
                        &commit_info.table_properties,
                        catalog_metadata_location.as_deref(),
                        &metadata_location,
                    )
                    .await
                    {
                        Ok(()) => {}
                        Err(error) if is_retryable_publication_conflict(&error) => {
                            cleanup_uncommitted_snapshot_attempt(
                                &store_ctx,
                                &metadata_file,
                                prepared_snapshot,
                                &mut task_files_may_be_committed,
                            )
                            .await?;
                            if attempt >= MAX_COMMIT_RETRIES {
                                return Err(commit_conflict_error());
                            }
                            continue;
                        }
                        Err(error) if publication_state_is_unknown(&error) => {
                            task_files_may_be_committed = true;
                            drop(prepared_snapshot);
                            return Err(error);
                        }
                        Err(error) => {
                            return Err(
                                error_after_snapshot_attempt_cleanup(
                                    &store_ctx,
                                    &metadata_file,
                                    prepared_snapshot,
                                    &mut task_files_may_be_committed,
                                    error,
                                )
                                .await,
                            );
                        }
                    }
                }

                task_files_may_be_committed = true;
                drop(prepared_snapshot);
                log::trace!("Metadata published successfully");

                let version_hint = if use_uuid_metadata_file {
                    metadata_file
                        .rsplit('/')
                        .next()
                        .unwrap_or(metadata_file.as_str())
                        .to_string()
                } else {
                    next_version.to_string()
                };
                write_version_hint(&store_ctx.prefixed, &version_hint).await;

                if catalog_metadata_update_table.is_none()
                    && let Some(catalog_table) = catalog_registered_metadata_table
                {
                    Self::update_catalog_metadata_location(
                        &context,
                        catalog_table,
                        &commit_info.table_properties,
                        catalog_recorded_metadata_location.as_deref(),
                        &metadata_location,
                    )
                    .await?;
                }

                return commit_count_batch(schema, commit_info.row_count);
            }
            }
            .await;

            match commit_result {
                Err(error) if !task_files_may_be_committed => {
                    match cleanup_uncommitted_task_files(&store_ctx, &task_file_paths).await {
                        Ok(()) => Err(error),
                        Err(cleanup_error) => {
                            Err(operation_with_cleanup_failure(error, cleanup_error))
                        }
                    }
                }
                result => result,
            }
        };

        let stream = once(future);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}

impl DisplayAs for IcebergCommitExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "IcebergCommitExec(table_path={})", self.table_url)
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "format: iceberg")?;
                write!(f, "table_path={}", self.table_url)
            }
        }
    }
}

fn commit_conflict_error() -> DataFusionError {
    DataFusionError::Execution(format!(
        "Iceberg commit failed after {MAX_COMMIT_RETRIES} retries due to concurrent metadata updates"
    ))
}

#[cfg(test)]
#[expect(clippy::expect_used)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier, Mutex};

    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionContext;
    use futures::stream::BoxStream;
    use futures::{StreamExt, TryStreamExt};
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
        ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use sail_catalog::manager::{CatalogManager, CatalogManagerOptions};
    use sail_catalog::provider::{
        AlterTableOptions, CatalogProvider, CreateTableMode, CreateTableOptions, Namespace,
    };
    use sail_catalog_memory::MemoryCatalogProvider;
    use sail_common_datafusion::catalog::{
        CatalogProviderId, CatalogTableIdentity, CommitAuthority, LakehouseAuthority,
        LakehouseFormat, LakehouseOperation, MetadataPointerAuthority, ScanAuthority,
        TableLifecycle,
    };

    use super::*;
    use crate::physical_plan::action_schema::{
        encode_add_data_files, encode_commit_meta, iceberg_action_schema,
    };
    use crate::spec::transform::Transform;
    use crate::spec::types::values::{Literal, PrimitiveLiteral};
    use crate::spec::types::{NestedField, PrimitiveType, Type};
    use crate::spec::{
        DataContentType, DataFileFormat, FormatVersion, Operation, SnapshotBuilder,
        SnapshotReference, SnapshotRetention,
    };

    #[test]
    fn scoped_overwrite_rejects_effective_v3_after_schema_evolution() {
        let schema = IcebergSchema::builder()
            .with_schema_id(1)
            .with_fields(vec![Arc::new(NestedField::required(
                1,
                "event_time",
                Type::Primitive(PrimitiveType::TimestampNs),
            ))])
            .build()
            .expect("v3 schema");
        let effective = FormatVersion::V2.max(format_version_for_schema(&schema));
        assert_eq!(effective, FormatVersion::V3);

        for mode in ["predicate", "dynamic"] {
            let error =
                validate_scoped_overwrite_format(SnapshotUpdateKind::CopyOnWrite, effective)
                    .expect_err(mode);
            assert!(error.to_string().contains("v3 scoped overwrite"), "{mode}");
        }
    }

    #[test]
    fn scoped_overwrite_execute_rechecks_format_after_schema_evolution() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/scoped-overwrite-v3/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(store, &table_url).expect("store context");
            let initial_schema = IcebergSchema::builder()
                .with_schema_id(0)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("v2 schema");
            let table_properties = vec![("format-version".to_string(), "2".to_string())];
            crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                initial_schema,
                PartitionSpec::unpartitioned_spec(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");

            let evolved_schema = IcebergSchema::builder()
                .with_schema_id(1)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "event_time",
                    Type::Primitive(PrimitiveType::TimestampNs),
                ))])
                .build()
                .expect("v3 schema");
            let action_schema = iceberg_action_schema().expect("action schema");
            let action_batch = encode_commit_meta(CommitMeta {
                table_uri: table_url.to_string(),
                row_count: 0,
                requirements: vec![],
                table_properties,
                lakehouse_table: None,
                schema: Some(evolved_schema),
                partition_spec: None,
            })
            .expect("commit metadata action");
            let input = MemorySourceConfig::try_new_exec(
                &[vec![action_batch]],
                Arc::clone(&action_schema),
                None,
            )
            .expect("memory input");
            let commit =
                IcebergCommitExec::new(input, table_url, None, SnapshotUpdateKind::CopyOnWrite)
                    .with_removed_data_file_paths(vec!["old.parquet".to_string()]);
            let context = SessionContext::new();
            context
                .runtime_env()
                .register_object_store(&Url::parse("file:///").expect("file store URL"), memory);

            let mut output = commit
                .execute(0, context.task_ctx())
                .expect("commit stream");
            let error = output
                .next()
                .await
                .expect("commit result")
                .expect_err("effective v3 scoped overwrite must fail");
            assert!(error.to_string().contains("v3 scoped overwrite"));
        });
    }

    fn partitioned_data_file(path: &str, spec_id: i32, value: i32) -> DataFile {
        DataFile {
            content: DataContentType::Data,
            file_path: path.to_string(),
            file_format: DataFileFormat::Parquet,
            partition: vec![Some(Literal::Primitive(PrimitiveLiteral::Int(value)))],
            record_count: 1,
            file_size_in_bytes: 1,
            column_sizes: HashMap::new(),
            value_counts: HashMap::new(),
            null_value_counts: HashMap::new(),
            nan_value_counts: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            block_size_in_bytes: None,
            key_metadata: None,
            split_offsets: Vec::new(),
            equality_ids: Vec::new(),
            sort_order_id: None,
            first_row_id: None,
            partition_spec_id: spec_id,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        }
    }

    fn identity_partition_spec() -> PartitionSpec {
        PartitionSpec::builder()
            .with_spec_id(3)
            .add_field(2, "part", Transform::Identity)
            .build()
    }

    #[test]
    fn dynamic_partition_overwrite_removes_only_touched_live_partitions() {
        let spec = identity_partition_spec();
        let added = vec![partitioned_data_file("new-2.parquet", 3, 2)];
        let live = vec![
            partitioned_data_file("old-1.parquet", 3, 1),
            partitioned_data_file("old-2.parquet", 3, 2),
            partitioned_data_file("old-3.parquet", 3, 3),
        ];
        let paths = IcebergCommitExec::dynamic_partition_overwrite_paths(&added, &live, &spec)
            .expect("dynamic overwrite paths");
        assert_eq!(paths, vec!["old-2.parquet"]);
    }

    #[test]
    fn dynamic_partition_overwrite_rejects_mismatched_partition_spec() {
        let spec = identity_partition_spec();
        let added = vec![partitioned_data_file("new.parquet", 4, 2)];
        let error = IcebergCommitExec::dynamic_partition_overwrite_paths(&added, &[], &spec)
            .expect_err("mismatched spec must fail");
        assert!(
            error
                .to_string()
                .contains("default Iceberg partition spec 3")
        );
    }

    #[test]
    fn empty_dynamic_overwrite_produces_no_removal_paths() {
        let spec = identity_partition_spec();
        let paths = IcebergCommitExec::dynamic_partition_overwrite_paths(
            &[],
            &[partitioned_data_file("old.parquet", 3, 1)],
            &spec,
        )
        .expect("empty dynamic overwrite");
        assert!(paths.is_empty());
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FilesystemRacePoint {
        BeforeCandidateCreate,
        AfterCandidateCreate,
        MetadataRead,
    }

    #[derive(Debug)]
    struct FilesystemRaceStore {
        memory_store: Arc<object_store::memory::InMemory>,
        race_point: FilesystemRacePoint,
        trigger_path: Path,
        concurrent_metadata_path: Path,
        concurrent_metadata: Bytes,
        version_hint_path: Path,
        concurrent_version_hint: Bytes,
        failed_delete_path: Option<Path>,
        conflict_injected: AtomicBool,
    }

    impl FilesystemRaceStore {
        async fn inject_concurrent_commit(&self) -> object_store::Result<()> {
            self.memory_store
                .put(
                    &self.concurrent_metadata_path,
                    PutPayload::from(self.concurrent_metadata.clone()),
                )
                .await?;
            self.memory_store
                .put(
                    &self.version_hint_path,
                    PutPayload::from(self.concurrent_version_hint.clone()),
                )
                .await?;
            Ok(())
        }

        async fn inject_once_if_triggered(&self, location: &Path) -> object_store::Result<()> {
            if location == &self.trigger_path
                && !self.conflict_injected.swap(true, Ordering::SeqCst)
            {
                self.inject_concurrent_commit().await?;
            }
            Ok(())
        }
    }

    impl std::fmt::Display for FilesystemRaceStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FilesystemRaceStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FilesystemRaceStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            if matches!(self.race_point, FilesystemRacePoint::BeforeCandidateCreate) {
                self.inject_once_if_triggered(location).await?;
            }
            let result = self.memory_store.put_opts(location, payload, opts).await?;
            if matches!(self.race_point, FilesystemRacePoint::AfterCandidateCreate) {
                self.inject_once_if_triggered(location).await?;
            }
            Ok(result)
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.memory_store.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let result = self.memory_store.get_opts(location, options).await?;
            if matches!(self.race_point, FilesystemRacePoint::MetadataRead) {
                self.inject_once_if_triggered(location).await?;
            }
            Ok(result)
        }

        async fn get_ranges(
            &self,
            location: &Path,
            ranges: &[Range<u64>],
        ) -> object_store::Result<Vec<Bytes>> {
            self.memory_store.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            let memory_store = Arc::clone(&self.memory_store);
            let failed_delete_path = self.failed_delete_path.clone();
            Box::pin(locations.then(move |location| {
                let memory_store = Arc::clone(&memory_store);
                let failed_delete_path = failed_delete_path.clone();
                async move {
                    let location = location?;
                    if failed_delete_path.as_ref() == Some(&location) {
                        return Err(object_store::Error::Generic {
                            store: "filesystem-race-store",
                            source: Box::new(std::io::Error::other(
                                "injected metadata deletion failure",
                            )),
                        });
                    }
                    memory_store.delete(&location).await?;
                    Ok(location)
                }
            }))
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.memory_store.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.memory_store.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.memory_store.copy_opts(from, to, options).await
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CatalogRacePoint {
        CandidateMetadataWrite,
        MetadataRead,
    }

    struct CatalogAdvancingStore {
        memory_store: Arc<object_store::memory::InMemory>,
        catalog: Arc<MemoryCatalogProvider>,
        database: Namespace,
        table: String,
        concurrent_metadata_location: String,
        race_point: CatalogRacePoint,
        trigger_read_path: Option<Path>,
        conflict_injected: AtomicBool,
    }

    impl CatalogAdvancingStore {
        async fn advance_catalog_once(&self) -> object_store::Result<()> {
            if self.conflict_injected.swap(true, Ordering::SeqCst) {
                return Ok(());
            }
            self.catalog
                .alter_table(
                    &self.database,
                    &self.table,
                    AlterTableOptions::SetTableProperties {
                        properties: vec![(
                            "metadata-location".to_string(),
                            self.concurrent_metadata_location.clone(),
                        )],
                    },
                )
                .await
                .map_err(|error| object_store::Error::Generic {
                    store: "catalog-advancing-store",
                    source: Box::new(std::io::Error::other(error.to_string())),
                })?;
            Ok(())
        }
    }

    impl std::fmt::Debug for CatalogAdvancingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CatalogAdvancingStore")
                .field("table", &self.table)
                .field(
                    "concurrent_metadata_location",
                    &self.concurrent_metadata_location,
                )
                .finish_non_exhaustive()
        }
    }

    impl std::fmt::Display for CatalogAdvancingStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CatalogAdvancingStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for CatalogAdvancingStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            let result = self.memory_store.put_opts(location, payload, opts).await?;
            if matches!(self.race_point, CatalogRacePoint::CandidateMetadataWrite)
                && location.as_ref().contains("/metadata/00002-")
                && location.as_ref().ends_with(".metadata.json")
            {
                self.advance_catalog_once().await?;
            }
            Ok(result)
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.memory_store.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            let result = self.memory_store.get_opts(location, options).await?;
            if matches!(self.race_point, CatalogRacePoint::MetadataRead)
                && self.trigger_read_path.as_ref() == Some(location)
            {
                self.advance_catalog_once().await?;
            }
            Ok(result)
        }

        async fn get_ranges(
            &self,
            location: &Path,
            ranges: &[Range<u64>],
        ) -> object_store::Result<Vec<Bytes>> {
            self.memory_store.get_ranges(location, ranges).await
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.memory_store.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.memory_store.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.memory_store.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.memory_store.copy_opts(from, to, options).await
        }
    }

    async fn object_payloads(
        store: &object_store::memory::InMemory,
        prefix: &Path,
    ) -> BTreeMap<String, Bytes> {
        let objects = store
            .list(Some(prefix))
            .try_collect::<Vec<_>>()
            .await
            .expect("list object payloads");
        let mut payloads = BTreeMap::new();
        for object in objects {
            let bytes = store
                .get(&object.location)
                .await
                .expect("get object payload")
                .bytes()
                .await
                .expect("read object payload");
            payloads.insert(object.location.to_string(), bytes);
        }
        payloads
    }

    fn readable_parquet_rows() -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![23_i64, 24_i64]))],
        )
        .expect("readable concurrent rows");
        let mut bytes = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut bytes, schema, None).expect("Parquet writer");
            writer.write(&batch).expect("write Parquet rows");
            writer.close().expect("close Parquet writer");
        }
        Bytes::from(bytes)
    }

    async fn prepare_readable_concurrent_metadata(
        table_url: &Url,
        store_ctx: &StoreContext,
        current_metadata: &TableMetadata,
        iceberg_schema: &IcebergSchema,
        partition_spec: &PartitionSpec,
        relative_data_path: &str,
    ) -> (TableMetadata, String, Bytes) {
        let parquet_bytes = readable_parquet_rows();
        store_ctx
            .prefixed
            .put(
                &Path::from(relative_data_path),
                PutPayload::from(parquet_bytes.clone()),
            )
            .await
            .expect("stage readable concurrent rows");
        let data_file_url = table_url
            .join(relative_data_path)
            .expect("concurrent data file URL")
            .to_string();
        let data_file = DataFile {
            content: DataContentType::Data,
            file_path: data_file_url.clone(),
            file_format: DataFileFormat::Parquet,
            partition: vec![],
            record_count: 2,
            file_size_in_bytes: parquet_bytes.len() as u64,
            column_sizes: HashMap::new(),
            value_counts: HashMap::new(),
            null_value_counts: HashMap::new(),
            nan_value_counts: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            block_size_in_bytes: None,
            key_metadata: None,
            split_offsets: vec![],
            equality_ids: vec![],
            sort_order_id: None,
            first_row_id: None,
            partition_spec_id: partition_spec.spec_id(),
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        };
        let parent = current_metadata
            .current_snapshot()
            .cloned()
            .expect("current snapshot");
        let transaction = Transaction::new(
            table_url.to_string(),
            parent,
            current_metadata.last_sequence_number,
        );
        let manifest_metadata = transaction.default_manifest_metadata(
            iceberg_schema,
            partition_spec,
            current_metadata.format_version,
        );
        let action_commit = SnapshotProducer::new(
            &transaction,
            vec![data_file],
            Some(store_ctx.clone()),
            Some(manifest_metadata),
        )
        .with_partition_specs(current_metadata.partition_specs.clone())
        .prepare(SnapshotUpdateKind::FullOverwrite)
        .await
        .expect("prepare readable concurrent snapshot")
        .into_action_commit();

        let mut concurrent_metadata = current_metadata.clone();
        for update in action_commit.updates() {
            match update {
                TableUpdate::AddSnapshot { snapshot } => {
                    concurrent_metadata.last_sequence_number = snapshot.sequence_number();
                    concurrent_metadata.current_snapshot_id = Some(snapshot.snapshot_id());
                    concurrent_metadata.snapshots.push(snapshot.clone());
                    concurrent_metadata.snapshot_log.push(SnapshotLog {
                        timestamp_ms: snapshot.timestamp_ms,
                        snapshot_id: snapshot.snapshot_id(),
                    });
                }
                TableUpdate::SetSnapshotRef {
                    ref_name,
                    reference,
                } => {
                    concurrent_metadata
                        .refs
                        .insert(ref_name.clone(), reference.clone());
                }
                _ => {}
            }
        }
        (concurrent_metadata, data_file_url, parquet_bytes)
    }

    async fn assert_readable_concurrent_rows(
        store_ctx: &StoreContext,
        metadata: &TableMetadata,
        expected_file_url: &str,
        expected_bytes: &Bytes,
    ) {
        let live_files = IcebergCommitExec::current_live_data_files(store_ctx, metadata)
            .await
            .expect("read concurrent snapshot manifests");
        assert_eq!(live_files.len(), 1);
        assert_eq!(live_files[0].file_path, expected_file_url);
        assert_eq!(live_files[0].record_count, 2);
        let (store, path) = store_ctx
            .resolve(expected_file_url)
            .expect("resolve concurrent data file");
        let bytes = store
            .get(&path)
            .await
            .expect("get concurrent data file")
            .bytes()
            .await
            .expect("read concurrent data file");
        assert_eq!(&bytes, expected_bytes);

        let reader = ParquetRecordBatchReaderBuilder::try_new(bytes)
            .expect("read Parquet metadata")
            .build()
            .expect("build Parquet reader");
        let mut rows = Vec::new();
        for batch in reader {
            let batch = batch.expect("read Parquet batch");
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id column");
            rows.extend((0..ids.len()).map(|index| ids.value(index)));
        }
        assert_eq!(rows, vec![23, 24]);
    }

    fn table_metadata_at_snapshot(snapshot_id: Option<i64>) -> TableMetadata {
        TableMetadata {
            format_version: FormatVersion::V2,
            table_uuid: None,
            location: "file:///tmp/table".to_string(),
            last_sequence_number: 2,
            last_updated_ms: 0,
            last_column_id: 0,
            schemas: vec![],
            current_schema_id: 0,
            partition_specs: vec![],
            default_spec_id: 0,
            last_partition_id: 0,
            properties: HashMap::new(),
            current_snapshot_id: snapshot_id,
            next_row_id: None,
            encryption_keys: vec![],
            snapshots: vec![],
            snapshot_log: vec![],
            metadata_log: vec![],
            sort_orders: vec![],
            default_sort_order_id: None,
            refs: HashMap::new(),
            statistics: vec![],
            partition_statistics: vec![],
        }
    }

    fn validation_only_delete_input(table_url: &Url) -> Result<Arc<dyn ExecutionPlan>> {
        let batch = encode_commit_meta(CommitMeta {
            table_uri: table_url.to_string(),
            row_count: 0,
            ..Default::default()
        })?;
        let schema = iceberg_action_schema()?;
        let input: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_from_batches(schema, vec![batch])?;
        Ok(input)
    }

    #[test]
    fn validation_only_delete_rejects_stale_snapshot_and_writes_no_metadata() -> Result<()> {
        futures::executor::block_on(async {
            let table_url = Url::parse("memory://strict-delete/table")
                .map_err(|error| DataFusionError::Plan(error.to_string()))?;
            let object_store = Arc::new(object_store::memory::InMemory::new());
            let store_ctx = StoreContext::new(object_store.clone(), &table_url)?;
            let metadata = table_metadata_at_snapshot(Some(2))
                .to_json()
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            let metadata_bytes = Bytes::from(metadata);
            let metadata_path = object_store::path::Path::from("metadata/v1.metadata.json");
            store_ctx
                .prefixed
                .put(
                    &metadata_path,
                    object_store::PutPayload::from(metadata_bytes.clone()),
                )
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            let objects_before = store_ctx
                .prefixed
                .list(None)
                .try_collect::<Vec<_>>()
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("memory://strict-delete")
                    .map_err(|error| DataFusionError::Plan(error.to_string()))?,
                object_store,
            );

            let stale: Arc<dyn ExecutionPlan> = Arc::new(
                IcebergCommitExec::new(
                    validation_only_delete_input(&table_url)?,
                    table_url.clone(),
                    None,
                    SnapshotUpdateKind::CopyOnWriteDelete,
                )
                .with_caller_expected_snapshot_id(Some(1)),
            );
            let error = datafusion::physical_plan::collect(stale, context.task_ctx())
                .await
                .expect_err("advanced snapshot must reject a validation-only delete");
            assert!(error.to_string().contains("expected snapshot Some(1)"));
            assert!(error.to_string().contains("found Some(2)"));

            let current: Arc<dyn ExecutionPlan> = Arc::new(
                IcebergCommitExec::new(
                    validation_only_delete_input(&table_url)?,
                    table_url,
                    None,
                    SnapshotUpdateKind::CopyOnWriteDelete,
                )
                .with_caller_expected_snapshot_id(Some(2)),
            );
            let batches = datafusion::physical_plan::collect(current, context.task_ctx()).await?;
            assert_eq!(batches.len(), 1);
            let counts = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| DataFusionError::Internal("count must be Int64".to_string()))?;
            assert_eq!(counts.value(0), 0);

            let objects_after = store_ctx
                .prefixed
                .list(None)
                .try_collect::<Vec<_>>()
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            assert_eq!(objects_after, objects_before);
            let metadata_after = store_ctx
                .prefixed
                .get(&metadata_path)
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?
                .bytes()
                .await
                .map_err(|error| DataFusionError::External(Box::new(error)))?;
            assert_eq!(metadata_after, metadata_bytes);
            Ok(())
        })
    }

    #[test]
    fn validation_only_delete_rejects_head_advanced_during_metadata_read() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/validation-race/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let base_store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(base_store, &table_url).expect("store context");
            let iceberg_schema = IcebergSchema::builder()
                .with_schema_id(1)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema");
            let partition_spec = PartitionSpec::builder().with_spec_id(1).build();
            let properties = vec![("format-version".to_string(), "2".to_string())];
            let bootstrap = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                iceberg_schema.clone(),
                partition_spec.clone(),
                &properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");
            let current_snapshot = SnapshotBuilder::new()
                .with_snapshot_id(17)
                .with_sequence_number(1)
                .with_timestamp_ms(123)
                .with_manifest_list("")
                .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
                .with_schema_id(iceberg_schema.schema_id())
                .build()
                .expect("current snapshot");
            let mut current_metadata = bootstrap.table_metadata;
            current_metadata.last_sequence_number = current_snapshot.sequence_number();
            current_metadata.current_snapshot_id = Some(current_snapshot.snapshot_id());
            current_metadata.snapshots = vec![current_snapshot.clone()];
            current_metadata.snapshot_log = vec![SnapshotLog {
                timestamp_ms: current_snapshot.timestamp_ms,
                snapshot_id: current_snapshot.snapshot_id(),
            }];
            current_metadata.refs.insert(
                MAIN_BRANCH.to_string(),
                SnapshotReference {
                    snapshot_id: current_snapshot.snapshot_id(),
                    retention: SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            );
            let current_json = current_metadata.to_json().expect("current metadata JSON");
            let current_bytes = Bytes::from(
                encode_metadata_file(&bootstrap.metadata_file, &current_json)
                    .expect("current metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(bootstrap.metadata_file.as_str()),
                    PutPayload::from(current_bytes.clone()),
                )
                .await
                .expect("write current metadata");

            let (concurrent_metadata, concurrent_data_file_url, concurrent_parquet_bytes) =
                prepare_readable_concurrent_metadata(
                    &table_url,
                    &store_ctx,
                    &current_metadata,
                    &iceberg_schema,
                    &partition_spec,
                    "data/concurrent.parquet",
                )
                .await;
            let concurrent_snapshot_id = concurrent_metadata
                .current_snapshot_id
                .expect("concurrent snapshot ID");
            let concurrent_metadata_file = "metadata/v2.metadata.json";
            let concurrent_json = concurrent_metadata
                .to_json()
                .expect("concurrent metadata JSON");
            let concurrent_bytes = Bytes::from(
                encode_metadata_file(concurrent_metadata_file, &concurrent_json)
                    .expect("concurrent metadata bytes"),
            );
            let base_root = Path::from("tmp/validation-race");
            let base_v1 = Path::from("tmp/validation-race/metadata/v1.metadata.json");
            let base_v2 = Path::from("tmp/validation-race/metadata/v2.metadata.json");
            let base_hint = Path::from("tmp/validation-race/metadata/version-hint.text");
            let mut expected_objects = object_payloads(memory.as_ref(), &base_root).await;
            expected_objects.insert(base_v2.to_string(), concurrent_bytes.clone());
            expected_objects.insert(base_hint.to_string(), Bytes::from_static(b"2"));

            let racing_store = Arc::new(FilesystemRaceStore {
                memory_store: Arc::clone(&memory),
                race_point: FilesystemRacePoint::MetadataRead,
                trigger_path: base_v1.clone(),
                concurrent_metadata_path: base_v2.clone(),
                concurrent_metadata: concurrent_bytes.clone(),
                version_hint_path: base_hint.clone(),
                concurrent_version_hint: Bytes::from_static(b"2"),
                failed_delete_path: None,
                conflict_injected: AtomicBool::new(false),
            });
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("file:///").expect("file store URL"),
                racing_store.clone(),
            );
            let commit: Arc<dyn ExecutionPlan> = Arc::new(
                IcebergCommitExec::new(
                    validation_only_delete_input(&table_url).expect("validation input"),
                    table_url,
                    None,
                    SnapshotUpdateKind::CopyOnWriteDelete,
                )
                .with_caller_expected_snapshot_id(Some(current_snapshot.snapshot_id())),
            );

            let error = datafusion::physical_plan::collect(commit, context.task_ctx())
                .await
                .expect_err("head advanced during validation-only execution must fail");

            assert!(error.to_string().contains("expected snapshot Some(17)"));
            assert!(
                error
                    .to_string()
                    .contains(&format!("found Some({concurrent_snapshot_id})"))
            );
            assert!(racing_store.conflict_injected.load(Ordering::SeqCst));
            assert_eq!(
                object_payloads(memory.as_ref(), &base_root).await,
                expected_objects
            );
            assert_eq!(
                memory
                    .get(&base_v1)
                    .await
                    .expect("current metadata")
                    .bytes()
                    .await
                    .expect("current metadata bytes"),
                current_bytes
            );
            assert_eq!(
                memory
                    .get(&base_v2)
                    .await
                    .expect("concurrent metadata")
                    .bytes()
                    .await
                    .expect("concurrent metadata bytes"),
                concurrent_bytes
            );
            assert_readable_concurrent_rows(
                &store_ctx,
                &concurrent_metadata,
                &concurrent_data_file_url,
                &concurrent_parquet_bytes,
            )
            .await;
        });
    }

    #[test]
    fn catalog_validation_only_delete_rejects_head_advanced_during_metadata_read() {
        futures::executor::block_on(async {
            let table_url =
                Url::parse("memory://catalog-validation-race/table/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let base_store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(base_store, &table_url).expect("store context");
            let iceberg_schema = IcebergSchema::builder()
                .with_schema_id(1)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema");
            let partition_spec = PartitionSpec::builder().with_spec_id(1).build();
            let table_properties = vec![
                ("format-version".to_string(), "2".to_string()),
                ("metadata.test".to_string(), "true".to_string()),
            ];
            let bootstrap = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                iceberg_schema.clone(),
                partition_spec.clone(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");
            let current_snapshot = SnapshotBuilder::new()
                .with_snapshot_id(17)
                .with_sequence_number(1)
                .with_timestamp_ms(123)
                .with_manifest_list("")
                .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
                .with_schema_id(iceberg_schema.schema_id())
                .build()
                .expect("current snapshot");
            let mut current_metadata = bootstrap.table_metadata;
            current_metadata.last_sequence_number = current_snapshot.sequence_number();
            current_metadata.current_snapshot_id = Some(current_snapshot.snapshot_id());
            current_metadata.snapshots = vec![current_snapshot.clone()];
            current_metadata.snapshot_log = vec![SnapshotLog {
                timestamp_ms: current_snapshot.timestamp_ms,
                snapshot_id: current_snapshot.snapshot_id(),
            }];
            current_metadata.refs.insert(
                MAIN_BRANCH.to_string(),
                SnapshotReference {
                    snapshot_id: current_snapshot.snapshot_id(),
                    retention: SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            );
            let current_json = current_metadata.to_json().expect("current metadata JSON");
            let current_bytes = Bytes::from(
                encode_metadata_file(&bootstrap.metadata_file, &current_json)
                    .expect("current metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(bootstrap.metadata_file.as_str()),
                    PutPayload::from(current_bytes),
                )
                .await
                .expect("write current metadata");
            let (concurrent_metadata, concurrent_data_file_url, concurrent_parquet_bytes) =
                prepare_readable_concurrent_metadata(
                    &table_url,
                    &store_ctx,
                    &current_metadata,
                    &iceberg_schema,
                    &partition_spec,
                    "data/concurrent.parquet",
                )
                .await;
            let concurrent_snapshot_id = concurrent_metadata
                .current_snapshot_id
                .expect("concurrent snapshot ID");
            let concurrent_metadata_file = "metadata/00002-concurrent.metadata.json".to_string();
            let concurrent_json = concurrent_metadata
                .to_json()
                .expect("concurrent metadata JSON");
            let concurrent_bytes = Bytes::from(
                encode_metadata_file(&concurrent_metadata_file, &concurrent_json)
                    .expect("concurrent metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(concurrent_metadata_file.as_str()),
                    PutPayload::from(concurrent_bytes),
                )
                .await
                .expect("stage concurrent metadata");
            let current_metadata_location =
                IcebergCommitExec::table_metadata_location(&table_url, &bootstrap.metadata_file)
                    .expect("current metadata location");
            let concurrent_metadata_location =
                IcebergCommitExec::table_metadata_location(&table_url, &concurrent_metadata_file)
                    .expect("concurrent metadata location");

            let database: Namespace = vec![Arc::<str>::from("default")]
                .try_into()
                .expect("database namespace");
            let catalog = Arc::new(MemoryCatalogProvider::new(
                "sail".to_string(),
                database.clone(),
                None,
            ));
            catalog
                .create_table(
                    &database,
                    "items",
                    CreateTableOptions {
                        columns: vec![],
                        comment: None,
                        constraints: vec![],
                        location: Some(table_url.to_string()),
                        format: "iceberg".to_string(),
                        partition_by: vec![],
                        sort_by: vec![],
                        bucket_by: None,
                        mode: CreateTableMode::Create,
                        properties: vec![
                            ("metadata-location".to_string(), current_metadata_location),
                            ("metadata.test".to_string(), "true".to_string()),
                        ],
                        is_external: true,
                        is_write_precondition: false,
                    },
                )
                .await
                .expect("catalog table");
            let catalog_manager = CatalogManager::try_new(CatalogManagerOptions {
                catalogs: HashMap::from([(
                    "sail".to_string(),
                    catalog.clone() as Arc<dyn CatalogProvider>,
                )]),
                default_catalog: "sail".to_string(),
                default_database: vec!["default".to_string()],
                global_temporary_database: vec!["global_temp".to_string()],
            })
            .expect("catalog manager");
            let lakehouse_table = LakehouseExecutionContext::catalog_table_context(
                CatalogProviderId("sail".to_string()),
                vec![
                    "sail".to_string(),
                    "default".to_string(),
                    "items".to_string(),
                ],
                CatalogTableIdentity {
                    table_id: Some("items".to_string()),
                    table_uri: Some(table_url.to_string()),
                },
                LakehouseOperation::Write,
                LakehouseFormat::Iceberg,
                LakehouseAuthority::CatalogAuthoritative {
                    lifecycle: TableLifecycle::External,
                    pointer: MetadataPointerAuthority::CatalogPropertyCas,
                    commit: CommitAuthority::IcebergMetadataLocationCas,
                },
                ScanAuthority::ClientTableFormat,
            );
            let objects_before = object_payloads(memory.as_ref(), &Path::from("table")).await;
            let racing_store = Arc::new(CatalogAdvancingStore {
                memory_store: Arc::clone(&memory),
                catalog: Arc::clone(&catalog),
                database: database.clone(),
                table: "items".to_string(),
                concurrent_metadata_location: concurrent_metadata_location.clone(),
                race_point: CatalogRacePoint::MetadataRead,
                trigger_read_path: Some(Path::from("table/metadata/v1.metadata.json")),
                conflict_injected: AtomicBool::new(false),
            });
            let mut state = SessionStateBuilder::new().build();
            state.config_mut().set_extension(Arc::new(catalog_manager));
            let context = SessionContext::new_with_state(state);
            context.runtime_env().register_object_store(
                &Url::parse("memory://catalog-validation-race").expect("catalog store URL"),
                racing_store.clone(),
            );
            let commit: Arc<dyn ExecutionPlan> = Arc::new(
                IcebergCommitExec::new(
                    validation_only_delete_input(&table_url).expect("validation input"),
                    table_url,
                    Some(lakehouse_table),
                    SnapshotUpdateKind::CopyOnWriteDelete,
                )
                .with_caller_expected_snapshot_id(Some(current_snapshot.snapshot_id())),
            );

            let error = datafusion::physical_plan::collect(commit, context.task_ctx())
                .await
                .expect_err("catalog head advanced during validation-only execution must fail");

            assert!(error.to_string().contains("expected snapshot Some(17)"));
            assert!(
                error
                    .to_string()
                    .contains(&format!("found Some({concurrent_snapshot_id})"))
            );
            assert!(racing_store.conflict_injected.load(Ordering::SeqCst));
            assert_eq!(
                object_payloads(memory.as_ref(), &Path::from("table")).await,
                objects_before
            );
            let status = catalog
                .get_table(&database, "items")
                .await
                .expect("catalog status");
            assert_eq!(
                crate::catalog_support::commit::catalog_table_info_from_status(&status)
                    .metadata_location
                    .as_deref(),
                Some(concurrent_metadata_location.as_str())
            );
            assert_readable_concurrent_rows(
                &store_ctx,
                &concurrent_metadata,
                &concurrent_data_file_url,
                &concurrent_parquet_bytes,
            )
            .await;
        });
    }

    #[test]
    fn planned_delete_snapshot_requirement_rejects_concurrent_branch_advance() {
        let metadata = Arc::new(Mutex::new(table_metadata_at_snapshot(Some(1))));
        let barrier = Arc::new(Barrier::new(2));
        let delete_metadata = Arc::clone(&metadata);
        let delete_barrier = Arc::clone(&barrier);
        let delete = std::thread::spawn(move || {
            let requirement = {
                let metadata = delete_metadata.lock().expect("metadata lock");
                expected_snapshot_requirement(Some(metadata.current_snapshot_id))
                    .expect("DELETE must capture its read snapshot")
            };
            delete_barrier.wait();
            delete_barrier.wait();
            let metadata = delete_metadata.lock().expect("metadata lock");
            IcebergCommitExec::validate_requirements(Some(&metadata), &[requirement])
        });

        barrier.wait();
        metadata.lock().expect("metadata lock").current_snapshot_id = Some(2);
        barrier.wait();

        let error = delete
            .join()
            .expect("DELETE validation thread")
            .expect_err("planned snapshot 1 must conflict with current snapshot 2");

        assert!(error.to_string().contains("expected snapshot Some(1)"));
        assert!(error.to_string().contains("found Some(2)"));
    }

    #[test]
    fn empty_read_snapshot_requirement_preserves_none() {
        let requirement = expected_snapshot_requirement(Some(None))
            .expect("planned empty snapshot must produce a requirement");
        assert!(
            IcebergCommitExec::validate_requirements(
                Some(&table_metadata_at_snapshot(None)),
                std::slice::from_ref(&requirement),
            )
            .is_ok()
        );
        assert!(
            IcebergCommitExec::validate_requirements(
                Some(&table_metadata_at_snapshot(Some(2))),
                &[requirement],
            )
            .is_err()
        );
    }

    #[test]
    fn cleanup_removes_absolute_and_relative_task_files() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/table/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let store: Arc<dyn object_store::ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(store, &table_url).expect("store context");
            let absolute_path = object_store::path::Path::from("tmp/table/data/absolute.parquet");
            let relative_path = object_store::path::Path::from("data/relative.parquet");
            memory
                .put(&absolute_path, Bytes::from_static(b"absolute").into())
                .await
                .expect("write absolute task file");
            store_ctx
                .prefixed
                .put(&relative_path, Bytes::from_static(b"relative").into())
                .await
                .expect("write relative task file");

            cleanup_uncommitted_task_files(
                &store_ctx,
                &[
                    "file:///tmp/table/data/absolute.parquet".to_string(),
                    "data/relative.parquet".to_string(),
                ],
            )
            .await
            .expect("clean up task files");

            assert!(matches!(
                memory.head(&absolute_path).await,
                Err(object_store::Error::NotFound { .. })
            ));
            assert!(matches!(
                store_ctx.prefixed.head(&relative_path).await,
                Err(object_store::Error::NotFound { .. })
            ));
        });
    }

    #[test]
    fn caller_expected_snapshot_race_preserves_concurrent_filesystem_commit() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/commit-conflict/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let base_store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(base_store, &table_url).expect("store context");
            let iceberg_schema = IcebergSchema::builder()
                .with_schema_id(1)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema");
            let partition_spec = PartitionSpec::builder().with_spec_id(1).build();
            let table_properties = vec![("format-version".to_string(), "2".to_string())];
            let bootstrap = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                iceberg_schema.clone(),
                partition_spec.clone(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");

            let current_snapshot = SnapshotBuilder::new()
                .with_snapshot_id(17)
                .with_sequence_number(1)
                .with_timestamp_ms(123)
                .with_manifest_list("")
                .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
                .with_schema_id(iceberg_schema.schema_id())
                .build()
                .expect("current snapshot");
            let mut current_metadata = bootstrap.table_metadata;
            current_metadata.last_sequence_number = current_snapshot.sequence_number();
            current_metadata.current_snapshot_id = Some(current_snapshot.snapshot_id());
            current_metadata.snapshots = vec![current_snapshot.clone()];
            current_metadata.snapshot_log = vec![SnapshotLog {
                timestamp_ms: current_snapshot.timestamp_ms,
                snapshot_id: current_snapshot.snapshot_id(),
            }];
            current_metadata.refs.insert(
                MAIN_BRANCH.to_string(),
                SnapshotReference {
                    snapshot_id: current_snapshot.snapshot_id(),
                    retention: SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            );
            let metadata_json = current_metadata.to_json().expect("metadata JSON");
            let metadata_bytes = Bytes::from(
                encode_metadata_file(&bootstrap.metadata_file, &metadata_json)
                    .expect("metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(bootstrap.metadata_file.as_str()),
                    PutPayload::from(metadata_bytes.clone()),
                )
                .await
                .expect("overwrite current metadata");

            let (concurrent_metadata, concurrent_data_file_url, concurrent_parquet_bytes) =
                prepare_readable_concurrent_metadata(
                    &table_url,
                    &store_ctx,
                    &current_metadata,
                    &iceberg_schema,
                    &partition_spec,
                    "data/concurrent.parquet",
                )
                .await;
            let concurrent_snapshot_id = concurrent_metadata
                .current_snapshot_id
                .expect("concurrent snapshot ID");
            let concurrent_metadata_file = "metadata/v2.metadata.json";
            let concurrent_metadata_json = concurrent_metadata
                .to_json()
                .expect("concurrent metadata JSON");
            let concurrent_metadata_bytes = Bytes::from(
                encode_metadata_file(concurrent_metadata_file, &concurrent_metadata_json)
                    .expect("concurrent metadata bytes"),
            );
            let base_v1_path = Path::from("tmp/commit-conflict/metadata/v1.metadata.json");
            let base_v2_path = Path::from("tmp/commit-conflict/metadata/v2.metadata.json");
            let base_version_hint_path =
                Path::from("tmp/commit-conflict/metadata/version-hint.text");
            let mut expected_objects =
                object_payloads(memory.as_ref(), &Path::from("tmp/commit-conflict")).await;
            expected_objects.insert(base_v2_path.to_string(), concurrent_metadata_bytes.clone());
            expected_objects.insert(base_version_hint_path.to_string(), Bytes::from_static(b"2"));

            let task_file_path = Path::from("data/task.parquet");
            store_ctx
                .prefixed
                .put(
                    &task_file_path,
                    PutPayload::from(Bytes::from_static(b"task-data")),
                )
                .await
                .expect("task file");
            let data_file = DataFile {
                content: DataContentType::Data,
                file_path: "file:///tmp/commit-conflict/data/task.parquet".to_string(),
                file_format: DataFileFormat::Parquet,
                partition: vec![],
                record_count: 1,
                file_size_in_bytes: 9,
                column_sizes: HashMap::new(),
                value_counts: HashMap::new(),
                null_value_counts: HashMap::new(),
                nan_value_counts: HashMap::new(),
                lower_bounds: HashMap::new(),
                upper_bounds: HashMap::new(),
                block_size_in_bytes: None,
                key_metadata: None,
                split_offsets: vec![],
                equality_ids: vec![],
                sort_order_id: None,
                first_row_id: None,
                partition_spec_id: partition_spec.spec_id(),
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
            };
            let action_schema = iceberg_action_schema().expect("action schema");
            let action_batch = datafusion::arrow::compute::concat_batches(
                &action_schema,
                &[
                    encode_add_data_files(vec![data_file]).expect("add action"),
                    encode_commit_meta(CommitMeta {
                        table_uri: table_url.to_string(),
                        row_count: 1,
                        requirements: vec![],
                        table_properties,
                        lakehouse_table: None,
                        schema: None,
                        partition_spec: None,
                    })
                    .expect("commit metadata action"),
                ],
            )
            .expect("action batch");
            let input = MemorySourceConfig::try_new_exec(
                &[vec![action_batch]],
                Arc::clone(&action_schema),
                None,
            )
            .expect("memory input");
            let commit =
                IcebergCommitExec::new(input, table_url, None, SnapshotUpdateKind::FastAppend)
                    .with_caller_expected_snapshot_id(Some(current_snapshot.snapshot_id()));
            let conflict_store = Arc::new(FilesystemRaceStore {
                memory_store: Arc::clone(&memory),
                race_point: FilesystemRacePoint::BeforeCandidateCreate,
                trigger_path: base_v2_path.clone(),
                concurrent_metadata_path: base_v2_path.clone(),
                concurrent_metadata: concurrent_metadata_bytes.clone(),
                version_hint_path: base_version_hint_path.clone(),
                concurrent_version_hint: Bytes::from_static(b"2"),
                failed_delete_path: None,
                conflict_injected: AtomicBool::new(false),
            });
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("file:///").expect("file store URL"),
                conflict_store.clone(),
            );

            let mut output = commit
                .execute(0, context.task_ctx())
                .expect("commit stream");
            let error = output
                .next()
                .await
                .expect("commit result")
                .expect_err("concurrent head must reject the caller expectation");

            assert!(error.to_string().contains("expected snapshot Some(17)"));
            assert!(
                error
                    .to_string()
                    .contains(&format!("found Some({concurrent_snapshot_id})"))
            );
            assert!(conflict_store.conflict_injected.load(Ordering::SeqCst));
            let objects_after =
                object_payloads(memory.as_ref(), &Path::from("tmp/commit-conflict")).await;
            assert_eq!(objects_after, expected_objects);
            let v1_bytes = memory
                .get(&base_v1_path)
                .await
                .expect("current metadata")
                .bytes()
                .await
                .expect("current metadata bytes");
            let v2_bytes = memory
                .get(&base_v2_path)
                .await
                .expect("concurrent metadata")
                .bytes()
                .await
                .expect("concurrent metadata bytes");
            let version_hint = memory
                .get(&base_version_hint_path)
                .await
                .expect("version hint")
                .bytes()
                .await
                .expect("version hint bytes");
            assert_eq!(v1_bytes, metadata_bytes);
            assert_eq!(v2_bytes, concurrent_metadata_bytes);
            assert_eq!(version_hint, Bytes::from_static(b"2"));
            let loaded_concurrent_metadata =
                TableMetadata::from_json(&v2_bytes).expect("load concurrent metadata");
            assert_eq!(
                loaded_concurrent_metadata.current_snapshot_id,
                Some(concurrent_snapshot_id)
            );
            assert_readable_concurrent_rows(
                &store_ctx,
                &loaded_concurrent_metadata,
                &concurrent_data_file_url,
                &concurrent_parquet_bytes,
            )
            .await;
            assert!(matches!(
                store_ctx.prefixed.head(&task_file_path).await,
                Err(object_store::Error::NotFound { .. })
            ));
        });
    }

    #[test]
    fn metadata_cleanup_failure_preserves_unknown_publication_state() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/cleanup-unknown/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let base_store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(base_store, &table_url).expect("store context");
            let iceberg_schema = IcebergSchema::builder()
                .with_schema_id(1)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema");
            let partition_spec = PartitionSpec::builder().with_spec_id(1).build();
            let table_properties = vec![("format-version".to_string(), "2".to_string())];
            let bootstrap = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                iceberg_schema.clone(),
                partition_spec.clone(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");
            let current_snapshot = SnapshotBuilder::new()
                .with_snapshot_id(17)
                .with_sequence_number(1)
                .with_timestamp_ms(123)
                .with_manifest_list("")
                .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
                .with_schema_id(iceberg_schema.schema_id())
                .build()
                .expect("current snapshot");
            let mut current_metadata = bootstrap.table_metadata;
            current_metadata.last_sequence_number = current_snapshot.sequence_number();
            current_metadata.current_snapshot_id = Some(current_snapshot.snapshot_id());
            current_metadata.snapshots = vec![current_snapshot.clone()];
            current_metadata.snapshot_log = vec![SnapshotLog {
                timestamp_ms: current_snapshot.timestamp_ms,
                snapshot_id: current_snapshot.snapshot_id(),
            }];
            current_metadata.refs.insert(
                MAIN_BRANCH.to_string(),
                SnapshotReference {
                    snapshot_id: current_snapshot.snapshot_id(),
                    retention: SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            );
            let current_json = current_metadata.to_json().expect("current metadata JSON");
            let current_bytes = Bytes::from(
                encode_metadata_file(&bootstrap.metadata_file, &current_json)
                    .expect("current metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(bootstrap.metadata_file.as_str()),
                    PutPayload::from(current_bytes.clone()),
                )
                .await
                .expect("write current metadata");

            let (concurrent_metadata, concurrent_data_file_url, concurrent_parquet_bytes) =
                prepare_readable_concurrent_metadata(
                    &table_url,
                    &store_ctx,
                    &current_metadata,
                    &iceberg_schema,
                    &partition_spec,
                    "data/concurrent.parquet",
                )
                .await;
            let concurrent_metadata_file = "metadata/00002-concurrent.metadata.json".to_string();
            let concurrent_json = concurrent_metadata
                .to_json()
                .expect("concurrent metadata JSON");
            let concurrent_bytes = Bytes::from(
                encode_metadata_file(&concurrent_metadata_file, &concurrent_json)
                    .expect("concurrent metadata bytes"),
            );
            let base_root = Path::from("tmp/cleanup-unknown");
            let candidate_metadata_path =
                Path::from("tmp/cleanup-unknown/metadata/v2.metadata.json");
            let concurrent_metadata_path =
                Path::from("tmp/cleanup-unknown/metadata/00002-concurrent.metadata.json");
            let version_hint_path = Path::from("tmp/cleanup-unknown/metadata/version-hint.text");
            let objects_before_attempt = object_payloads(memory.as_ref(), &base_root).await;

            let task_file_path = Path::from("data/task.parquet");
            let task_file_url = table_url
                .join("data/task.parquet")
                .expect("task file URL")
                .to_string();
            store_ctx
                .prefixed
                .put(
                    &task_file_path,
                    PutPayload::from(Bytes::from_static(b"task-data")),
                )
                .await
                .expect("task file");
            let data_file = DataFile {
                content: DataContentType::Data,
                file_path: task_file_url.clone(),
                file_format: DataFileFormat::Parquet,
                partition: vec![],
                record_count: 1,
                file_size_in_bytes: 9,
                column_sizes: HashMap::new(),
                value_counts: HashMap::new(),
                null_value_counts: HashMap::new(),
                nan_value_counts: HashMap::new(),
                lower_bounds: HashMap::new(),
                upper_bounds: HashMap::new(),
                block_size_in_bytes: None,
                key_metadata: None,
                split_offsets: vec![],
                equality_ids: vec![],
                sort_order_id: None,
                first_row_id: None,
                partition_spec_id: partition_spec.spec_id(),
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
            };
            let action_schema = iceberg_action_schema().expect("action schema");
            let action_batch = datafusion::arrow::compute::concat_batches(
                &action_schema,
                &[
                    encode_add_data_files(vec![data_file]).expect("add action"),
                    encode_commit_meta(CommitMeta {
                        table_uri: table_url.to_string(),
                        row_count: 1,
                        requirements: vec![],
                        table_properties,
                        lakehouse_table: None,
                        schema: None,
                        partition_spec: None,
                    })
                    .expect("commit metadata action"),
                ],
            )
            .expect("action batch");
            let input = MemorySourceConfig::try_new_exec(
                &[vec![action_batch]],
                Arc::clone(&action_schema),
                None,
            )
            .expect("memory input");
            let commit =
                IcebergCommitExec::new(input, table_url, None, SnapshotUpdateKind::FastAppend)
                    .with_caller_expected_snapshot_id(Some(current_snapshot.snapshot_id()));
            let failing_store = Arc::new(FilesystemRaceStore {
                memory_store: Arc::clone(&memory),
                race_point: FilesystemRacePoint::AfterCandidateCreate,
                trigger_path: candidate_metadata_path.clone(),
                concurrent_metadata_path: concurrent_metadata_path.clone(),
                concurrent_metadata: concurrent_bytes.clone(),
                version_hint_path: version_hint_path.clone(),
                concurrent_version_hint: Bytes::from_static(b"00002-concurrent.metadata.json"),
                failed_delete_path: Some(candidate_metadata_path.clone()),
                conflict_injected: AtomicBool::new(false),
            });
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("file:///").expect("file store URL"),
                failing_store.clone(),
            );

            let mut output = commit
                .execute(0, context.task_ctx())
                .expect("commit stream");
            let error = output
                .next()
                .await
                .expect("commit result")
                .expect_err("metadata cleanup failure makes publication state unknown");

            assert!(publication_state_is_unknown(&error));
            assert!(
                error
                    .to_string()
                    .contains("injected metadata deletion failure")
            );
            assert!(failing_store.conflict_injected.load(Ordering::SeqCst));
            let objects_after = object_payloads(memory.as_ref(), &base_root).await;
            for (path, bytes) in &objects_before_attempt {
                if path.ends_with("metadata/version-hint.text") {
                    continue;
                }
                assert_eq!(
                    objects_after.get(path),
                    Some(bytes),
                    "changed object {path}"
                );
            }
            assert_eq!(
                objects_after.get(concurrent_metadata_path.as_ref()),
                Some(&concurrent_bytes)
            );
            assert_eq!(
                objects_after.get(version_hint_path.as_ref()),
                Some(&Bytes::from_static(b"00002-concurrent.metadata.json"))
            );
            assert_eq!(
                objects_after.get("tmp/cleanup-unknown/data/task.parquet"),
                Some(&Bytes::from_static(b"task-data"))
            );

            let candidate_bytes = objects_after
                .get(candidate_metadata_path.as_ref())
                .expect("uncertain candidate metadata is retained");
            let candidate_metadata = TableMetadata::from_json(candidate_bytes)
                .expect("retained candidate metadata is readable");
            let candidate_live_files =
                IcebergCommitExec::current_live_data_files(&store_ctx, &candidate_metadata)
                    .await
                    .expect("retained candidate manifests are readable");
            assert_eq!(candidate_live_files.len(), 1);
            assert_eq!(candidate_live_files[0].file_path, task_file_url);
            assert_eq!(candidate_live_files[0].record_count, 1);

            let candidate_snapshot = candidate_metadata
                .current_snapshot()
                .expect("candidate snapshot");
            let (_, candidate_manifest_list_path) = store_ctx
                .resolve(candidate_snapshot.manifest_list())
                .expect("candidate manifest-list path");
            let candidate_manifest_list =
                load_manifest_list(&store_ctx, candidate_snapshot.manifest_list())
                    .await
                    .expect("candidate manifest list");
            let mut expected_paths = objects_before_attempt
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>();
            expected_paths.insert(candidate_metadata_path.to_string());
            expected_paths.insert(concurrent_metadata_path.to_string());
            expected_paths.insert(version_hint_path.to_string());
            expected_paths.insert("tmp/cleanup-unknown/data/task.parquet".to_string());
            expected_paths.insert(candidate_manifest_list_path.to_string());
            for manifest in candidate_manifest_list.entries() {
                let (_, path) = store_ctx
                    .resolve(&manifest.manifest_path)
                    .expect("candidate manifest path");
                expected_paths.insert(path.to_string());
            }
            assert_eq!(
                objects_after.keys().cloned().collect::<BTreeSet<_>>(),
                expected_paths
            );
            assert_readable_concurrent_rows(
                &store_ctx,
                &concurrent_metadata,
                &concurrent_data_file_url,
                &concurrent_parquet_bytes,
            )
            .await;
        });
    }

    #[test]
    fn task_cleanup_failure_is_returned_and_retains_the_failed_file() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/task-cleanup-failure/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let failed_path = Path::from("tmp/task-cleanup-failure/data/task.parquet");
            let store: Arc<dyn ObjectStore> = Arc::new(FilesystemRaceStore {
                memory_store: Arc::clone(&memory),
                race_point: FilesystemRacePoint::MetadataRead,
                trigger_path: Path::from("never"),
                concurrent_metadata_path: Path::from("never"),
                concurrent_metadata: Bytes::new(),
                version_hint_path: Path::from("never"),
                concurrent_version_hint: Bytes::new(),
                failed_delete_path: Some(failed_path.clone()),
                conflict_injected: AtomicBool::new(false),
            });
            let store_ctx = StoreContext::new(store, &table_url).expect("store context");
            let task_url = table_url
                .join("data/task.parquet")
                .expect("task URL")
                .to_string();
            store_ctx
                .prefixed
                .put(
                    &Path::from("data/task.parquet"),
                    PutPayload::from(Bytes::from_static(b"task")),
                )
                .await
                .expect("task file");

            let error = cleanup_uncommitted_task_files(&store_ctx, &[task_url])
                .await
                .expect_err("task cleanup failure must be surfaced");

            assert!(
                error
                    .to_string()
                    .contains("injected metadata deletion failure")
            );
            memory
                .head(&failed_path)
                .await
                .expect("failed task deletion is retained");
        });
    }

    #[test]
    fn caller_expected_snapshot_race_cleans_metadata_location_cas_attempt() {
        futures::executor::block_on(async {
            let table_url = Url::parse("memory://catalog-race/table/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let base_store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(base_store, &table_url).expect("store context");
            let iceberg_schema = IcebergSchema::builder()
                .with_schema_id(1)
                .with_fields([Arc::new(NestedField::required(
                    1,
                    "id",
                    Type::Primitive(PrimitiveType::Int),
                ))])
                .build()
                .expect("schema");
            let partition_spec = PartitionSpec::builder().with_spec_id(1).build();
            let table_properties = vec![
                ("format-version".to_string(), "2".to_string()),
                ("metadata.test".to_string(), "true".to_string()),
            ];
            let bootstrap = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                iceberg_schema.clone(),
                partition_spec.clone(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");

            let current_snapshot = SnapshotBuilder::new()
                .with_snapshot_id(17)
                .with_sequence_number(1)
                .with_timestamp_ms(123)
                .with_manifest_list("")
                .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
                .with_schema_id(iceberg_schema.schema_id())
                .build()
                .expect("current snapshot");
            let mut current_metadata = bootstrap.table_metadata;
            current_metadata.last_sequence_number = current_snapshot.sequence_number();
            current_metadata.current_snapshot_id = Some(current_snapshot.snapshot_id());
            current_metadata.snapshots = vec![current_snapshot.clone()];
            current_metadata.snapshot_log = vec![SnapshotLog {
                timestamp_ms: current_snapshot.timestamp_ms,
                snapshot_id: current_snapshot.snapshot_id(),
            }];
            current_metadata.refs.insert(
                MAIN_BRANCH.to_string(),
                SnapshotReference {
                    snapshot_id: current_snapshot.snapshot_id(),
                    retention: SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            );
            let current_metadata_json = current_metadata.to_json().expect("current metadata JSON");
            let current_metadata_bytes = Bytes::from(
                encode_metadata_file(&bootstrap.metadata_file, &current_metadata_json)
                    .expect("current metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(bootstrap.metadata_file.as_str()),
                    PutPayload::from(current_metadata_bytes.clone()),
                )
                .await
                .expect("overwrite current metadata");

            let (concurrent_metadata, concurrent_data_file_url, concurrent_parquet_bytes) =
                prepare_readable_concurrent_metadata(
                    &table_url,
                    &store_ctx,
                    &current_metadata,
                    &iceberg_schema,
                    &partition_spec,
                    "data/concurrent.parquet",
                )
                .await;
            let concurrent_snapshot_id = concurrent_metadata
                .current_snapshot_id
                .expect("concurrent snapshot ID");
            let concurrent_metadata_file = "metadata/00003-concurrent.metadata.json".to_string();
            let concurrent_metadata_json = concurrent_metadata
                .to_json()
                .expect("concurrent metadata JSON");
            let concurrent_metadata_bytes = Bytes::from(
                encode_metadata_file(&concurrent_metadata_file, &concurrent_metadata_json)
                    .expect("concurrent metadata bytes"),
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(concurrent_metadata_file.as_str()),
                    PutPayload::from(concurrent_metadata_bytes.clone()),
                )
                .await
                .expect("stage concurrent metadata");
            let current_metadata_location =
                IcebergCommitExec::table_metadata_location(&table_url, &bootstrap.metadata_file)
                    .expect("current metadata location");
            let concurrent_metadata_location =
                IcebergCommitExec::table_metadata_location(&table_url, &concurrent_metadata_file)
                    .expect("concurrent metadata location");

            let database: Namespace = vec![Arc::<str>::from("default")]
                .try_into()
                .expect("database namespace");
            let catalog = Arc::new(MemoryCatalogProvider::new(
                "sail".to_string(),
                database.clone(),
                None,
            ));
            catalog
                .create_table(
                    &database,
                    "items",
                    CreateTableOptions {
                        columns: vec![],
                        comment: None,
                        constraints: vec![],
                        location: Some(table_url.to_string()),
                        format: "iceberg".to_string(),
                        partition_by: vec![],
                        sort_by: vec![],
                        bucket_by: None,
                        mode: CreateTableMode::Create,
                        properties: vec![
                            (
                                "metadata-location".to_string(),
                                current_metadata_location.clone(),
                            ),
                            ("metadata.test".to_string(), "true".to_string()),
                        ],
                        is_external: true,
                        is_write_precondition: false,
                    },
                )
                .await
                .expect("catalog table");
            let catalog_manager = CatalogManager::try_new(CatalogManagerOptions {
                catalogs: HashMap::from([(
                    "sail".to_string(),
                    catalog.clone() as Arc<dyn CatalogProvider>,
                )]),
                default_catalog: "sail".to_string(),
                default_database: vec!["default".to_string()],
                global_temporary_database: vec!["global_temp".to_string()],
            })
            .expect("catalog manager");
            let lakehouse_table = LakehouseExecutionContext::catalog_table_context(
                CatalogProviderId("sail".to_string()),
                vec![
                    "sail".to_string(),
                    "default".to_string(),
                    "items".to_string(),
                ],
                CatalogTableIdentity {
                    table_id: Some("items".to_string()),
                    table_uri: Some(table_url.to_string()),
                },
                LakehouseOperation::Write,
                LakehouseFormat::Iceberg,
                LakehouseAuthority::CatalogAuthoritative {
                    lifecycle: TableLifecycle::External,
                    pointer: MetadataPointerAuthority::CatalogPropertyCas,
                    commit: CommitAuthority::IcebergMetadataLocationCas,
                },
                ScanAuthority::ClientTableFormat,
            );

            let objects_before = object_payloads(memory.as_ref(), &Path::from("table")).await;
            let task_file_path = Path::from("data/task.parquet");
            store_ctx
                .prefixed
                .put(
                    &task_file_path,
                    PutPayload::from(Bytes::from_static(b"task-data")),
                )
                .await
                .expect("task file");
            let data_file = DataFile {
                content: DataContentType::Data,
                file_path: "memory://catalog-race/table/data/task.parquet".to_string(),
                file_format: DataFileFormat::Parquet,
                partition: vec![],
                record_count: 1,
                file_size_in_bytes: 9,
                column_sizes: HashMap::new(),
                value_counts: HashMap::new(),
                null_value_counts: HashMap::new(),
                nan_value_counts: HashMap::new(),
                lower_bounds: HashMap::new(),
                upper_bounds: HashMap::new(),
                block_size_in_bytes: None,
                key_metadata: None,
                split_offsets: vec![],
                equality_ids: vec![],
                sort_order_id: None,
                first_row_id: None,
                partition_spec_id: partition_spec.spec_id(),
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
            };
            let action_schema = iceberg_action_schema().expect("action schema");
            let action_batch = datafusion::arrow::compute::concat_batches(
                &action_schema,
                &[
                    encode_add_data_files(vec![data_file]).expect("add action"),
                    encode_commit_meta(CommitMeta {
                        table_uri: table_url.to_string(),
                        row_count: 1,
                        requirements: vec![],
                        table_properties: vec![
                            ("metadata-location".to_string(), current_metadata_location),
                            ("metadata.test".to_string(), "true".to_string()),
                        ],
                        lakehouse_table: Some(lakehouse_table.clone()),
                        schema: None,
                        partition_spec: None,
                    })
                    .expect("commit metadata action"),
                ],
            )
            .expect("action batch");
            let input = MemorySourceConfig::try_new_exec(
                &[vec![action_batch]],
                Arc::clone(&action_schema),
                None,
            )
            .expect("memory input");
            let commit = IcebergCommitExec::new(
                input,
                table_url.clone(),
                Some(lakehouse_table),
                SnapshotUpdateKind::FastAppend,
            )
            .with_caller_expected_snapshot_id(Some(current_snapshot.snapshot_id()));

            let racing_store = Arc::new(CatalogAdvancingStore {
                memory_store: Arc::clone(&memory),
                catalog: Arc::clone(&catalog),
                database: database.clone(),
                table: "items".to_string(),
                concurrent_metadata_location: concurrent_metadata_location.clone(),
                race_point: CatalogRacePoint::CandidateMetadataWrite,
                trigger_read_path: None,
                conflict_injected: AtomicBool::new(false),
            });
            let mut state = SessionStateBuilder::new().build();
            state.config_mut().set_extension(Arc::new(catalog_manager));
            let context = SessionContext::new_with_state(state);
            context.runtime_env().register_object_store(
                &Url::parse("memory://catalog-race").expect("catalog store URL"),
                racing_store.clone(),
            );

            let mut output = commit
                .execute(0, context.task_ctx())
                .expect("commit stream");
            let error = output
                .next()
                .await
                .expect("commit result")
                .expect_err("metadata-location race must reject the caller expectation");

            assert!(error.to_string().contains("expected snapshot Some(17)"));
            assert!(
                error
                    .to_string()
                    .contains(&format!("found Some({concurrent_snapshot_id})"))
            );
            assert!(racing_store.conflict_injected.load(Ordering::SeqCst));
            let objects_after = object_payloads(memory.as_ref(), &Path::from("table")).await;
            assert_eq!(objects_after, objects_before);
            let concurrent_bytes_after = store_ctx
                .prefixed
                .get(&Path::from(concurrent_metadata_file.as_str()))
                .await
                .expect("concurrent metadata")
                .bytes()
                .await
                .expect("concurrent metadata bytes");
            assert_eq!(concurrent_bytes_after, concurrent_metadata_bytes);
            assert_readable_concurrent_rows(
                &store_ctx,
                &concurrent_metadata,
                &concurrent_data_file_url,
                &concurrent_parquet_bytes,
            )
            .await;
            assert!(matches!(
                store_ctx.prefixed.head(&task_file_path).await,
                Err(object_store::Error::NotFound { .. })
            ));
            let status = catalog
                .get_table(&database, "items")
                .await
                .expect("catalog status");
            assert_eq!(
                crate::catalog_support::commit::catalog_table_info_from_status(&status)
                    .metadata_location
                    .as_deref(),
                Some(concurrent_metadata_location.as_str())
            );
        });
    }
}
