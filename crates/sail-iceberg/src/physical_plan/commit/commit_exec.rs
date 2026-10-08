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

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::{Distribution, EquivalenceProperties, PhysicalExpr};
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
use crate::lake_source::{
    catalog_managed_iceberg_from_properties, metadata_location_from_properties,
    resolve_iceberg_metadata_location,
};
use crate::operations::bootstrap::{
    BootstrapResult, NewTableMetadataStyle, PersistStrategy, bootstrap_first_snapshot,
    bootstrap_new_table_with_style, is_bootstrap_metadata_conflict, prepare_bootstrap_snapshot,
};
use crate::operations::helpers::format_version_for_schema;
use crate::operations::{SnapshotProducer, SnapshotUpdateKind, Transaction};
use crate::physical_plan::action_schema::{CommitMeta, decode_actions_and_meta_from_batch};
use crate::physical_plan::commit::IcebergCommitInfo;
use crate::snapshot_properties::validate_snapshot_properties;
use crate::spec::catalog::TableUpdate;
use crate::spec::manifest::ManifestStatus;
use crate::spec::metadata::table_metadata::SnapshotLog;
use crate::spec::partition::{UnboundPartitionField, UnboundPartitionSpec};
use crate::spec::snapshots::MAIN_BRANCH;
use crate::spec::{
    DataContentType, DataFile, Literal, PartitionKey, PartitionSpec, Schema as IcebergSchema,
    StructType, TableMetadata, TableRequirement, Type,
};
use crate::table::metadata_loader::{
    encode_metadata_file, load_metadata_file_bytes, metadata_file_extension_from_properties,
    metadata_file_version_from_path, metadata_location_to_object_path_string, write_version_hint,
};
use crate::utils::get_object_store_from_context;
use crate::utils::metadata::{
    ExistingMetadataFile, metadata_files_for_version, reconcile_existing_metadata_file,
};
const MAX_COMMIT_RETRIES: usize = 5;

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

fn is_retryable_catalog_conflict(error: &DataFusionError) -> bool {
    matches!(
        error,
        DataFusionError::External(source)
            if source.downcast_ref::<CatalogError>().is_some_and(|error| {
                matches!(
                    error,
                    CatalogError::Conflict(_) | CatalogError::StaleMetadata(_)
                )
            })
    )
}

/// Remove a bootstrap whose catalog pointer update was never sent.
///
/// Nothing can reference these artifacts yet. Once a pointer update has been sent, its
/// outcome can be ambiguous, so the artifacts are kept and the outcome is reconciled with
/// [`IcebergCommitExec::publish_catalog_pointer`] instead.
async fn discard_unsent_bootstrap(
    bootstrap_result: BootstrapResult,
    store_ctx: &StoreContext,
    error: DataFusionError,
) -> DataFusionError {
    match bootstrap_result.cleanup(store_ctx).await {
        Ok(()) => error,
        Err(cleanup_error) => DataFusionError::Execution(format!(
            "{error}; unpublished Iceberg bootstrap cleanup also failed: {cleanup_error}"
        )),
    }
}

/// The catalog entry of a filesystem-mode table, in which a commit records the metadata
/// location of its new head.
struct FilesystemCatalogRegistration<'a> {
    catalog_table: &'a [String],
    table_properties: &'a [(String, String)],
    /// The metadata location recorded for the table when the write was planned, else the one
    /// the catalog held when the commit started.
    recorded_metadata_location: Option<&'a str>,
    /// Whether the commit built on the metadata the catalog pointer names instead of the
    /// metadata directory listing. That happens only when the planned table properties mark
    /// the table as catalog managed while its catalog entry at commit time does not.
    builds_on_catalog_pointer: bool,
}

/// What a catalog metadata pointer update did.
#[derive(Debug)]
enum PointerUpdateOutcome {
    /// The catalog head is the new metadata or was built on it, so the write committed.
    Committed,
    /// The catalog refused the update and its head neither is nor was built on the new
    /// metadata, so the update can never apply. Carries the update error.
    Rejected(DataFusionError),
}

/// Whether a catalog metadata pointer update failed because the catalog refused it.
///
/// A refused update can never apply later. A conflict only counts as a refusal for a
/// compare-and-swap update, because once the pointer has moved away from the expected previous
/// location a copy of the request that is still in flight can no longer apply. Any other
/// failure, such as a timeout or an unavailable catalog, can leave the request in flight, so
/// it may still apply after the error was returned.
fn is_pointer_update_rejection(error: &DataFusionError, compare_and_swap: bool) -> bool {
    let DataFusionError::External(source) = error else {
        return false;
    };
    source
        .downcast_ref::<CatalogError>()
        .is_some_and(|error| match error {
            CatalogError::Conflict(_) | CatalogError::StaleMetadata(_) => compare_and_swap,
            CatalogError::NotFound(_, _)
            | CatalogError::Unauthorized(_)
            | CatalogError::Forbidden(_)
            | CatalogError::InvalidArgument(_)
            | CatalogError::CommitRejected(_)
            | CatalogError::ReadOnly(_)
            | CatalogError::NotSupported(_)
            | CatalogError::UnsupportedCapability(_) => true,
            _ => false,
        })
}

fn pointer_update_state_unknown(
    new_metadata_location: &str,
    error: &DataFusionError,
    detail: String,
) -> DataFusionError {
    DataFusionError::External(Box::new(CatalogError::CommitStateUnknown(format!(
        "Iceberg catalog pointer update to {new_metadata_location} failed ({error}) and {detail}; \
         the metadata file and its snapshot files were kept"
    ))))
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

    /// Require the main branch to be at this snapshot when the write commits.
    pub fn with_caller_expected_snapshot_id(mut self, snapshot_id: Option<i64>) -> Self {
        self.caller_expected_snapshot_id = snapshot_id;
        self
    }

    /// Add caller properties to the summary of the snapshot this write commits.
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

    /// Reject publication controls that could never commit as requested.
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

    fn apply_partition_spec_update(
        table_meta: &mut TableMetadata,
        new_spec: PartitionSpec,
    ) -> Result<()> {
        let spec_id = new_spec.spec_id();
        if let Some(previous) = table_meta
            .partition_specs
            .iter()
            .find(|spec| spec.spec_id() == spec_id)
        {
            if previous != &new_spec {
                return Err(DataFusionError::Plan(format!(
                    "Cannot replace Iceberg partition spec {spec_id} with a different definition"
                )));
            }
        } else {
            table_meta.partition_specs.push(new_spec.clone());
        }
        table_meta.default_spec_id = spec_id;
        table_meta.last_partition_id = table_meta
            .last_partition_id
            .max(new_spec.last_assigned_field_id());
        Ok(())
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

    /// Record in the catalog the metadata a filesystem-mode commit has written.
    ///
    /// Readers of a filesystem-mode table find its head by listing the metadata directory, so
    /// a commit that built on that listing landed once its metadata file was written, and the
    /// catalog only records that head. A registration that fails or that the catalog refuses,
    /// such as a compare-and-swap conflict against a stale recorded location, therefore does
    /// not fail the commit: a caller that retried it would append the same rows twice. It is
    /// logged as a warning.
    ///
    /// A commit that built on the metadata the catalog pointer names did not read the listing,
    /// so its metadata file is not known to be the head that readers find. It publishes the
    /// pointer with [`Self::publish_catalog_pointer`] instead and reports a refusal.
    async fn register_filesystem_metadata_location(
        context: &Arc<TaskContext>,
        object_store: &Arc<dyn object_store::ObjectStore>,
        registration: &FilesystemCatalogRegistration<'_>,
        table_url: &Url,
        metadata_file: &str,
        new_snapshot_id: Option<i64>,
    ) -> Result<()> {
        if registration.builds_on_catalog_pointer {
            let new_metadata_location = Self::table_metadata_location(table_url, metadata_file)?;
            return match Self::publish_catalog_pointer(
                context,
                object_store,
                registration.catalog_table,
                registration.table_properties,
                registration.recorded_metadata_location,
                &new_metadata_location,
                new_snapshot_id,
            )
            .await?
            {
                PointerUpdateOutcome::Committed => Ok(()),
                PointerUpdateOutcome::Rejected(error) => Err(error),
            };
        }
        let table = registration.catalog_table.join(".");
        let new_metadata_location = match Self::table_metadata_location(table_url, metadata_file) {
            Ok(location) => location,
            Err(error) => {
                log::warn!(
                    "Iceberg table {table} committed metadata file {metadata_file} but its \
                     location could not be resolved to record in the catalog; the metadata \
                     directory stays the source of truth: {error}"
                );
                return Ok(());
            }
        };
        if let Err(error) = Self::update_catalog_metadata_location(
            context,
            registration.catalog_table,
            registration.table_properties,
            registration.recorded_metadata_location,
            &new_metadata_location,
        )
        .await
        {
            log::warn!(
                "Iceberg table {table} committed metadata {new_metadata_location} but recording \
                 it in the catalog failed; the metadata directory stays the source of truth: \
                 {error}"
            );
        }
        Ok(())
    }

    /// Point the catalog at metadata that is already written, when the commit built on the
    /// metadata the catalog pointer names.
    ///
    /// An update that reports an error is reconciled with [`Self::reconcile_pointer_update`],
    /// because the request was sent and may have applied. The caller keeps every file the
    /// commit wrote whatever this returns.
    async fn publish_catalog_pointer(
        context: &Arc<TaskContext>,
        object_store: &Arc<dyn object_store::ObjectStore>,
        catalog_table: &[String],
        existing_properties: &[(String, String)],
        previous_metadata_location: Option<&str>,
        new_metadata_location: &str,
        new_snapshot_id: Option<i64>,
    ) -> Result<PointerUpdateOutcome> {
        match Self::update_catalog_metadata_location(
            context,
            catalog_table,
            existing_properties,
            previous_metadata_location,
            new_metadata_location,
        )
        .await
        {
            Ok(()) => Ok(PointerUpdateOutcome::Committed),
            Err(error) => {
                Self::reconcile_pointer_update(
                    context,
                    object_store,
                    catalog_table,
                    previous_metadata_location,
                    new_metadata_location,
                    new_snapshot_id,
                    error,
                )
                .await
            }
        }
    }

    /// Find out whether a catalog pointer update that reported an error took effect.
    ///
    /// The update can apply even though its response is lost, and a retried request can then
    /// report a conflict against its own write. The catalog pointer is reloaded: the update
    /// applied when the pointer names the new metadata location, or when the metadata it names
    /// shows that another writer already committed on top of it. That metadata then contains
    /// the new snapshot, or its metadata log lists the new metadata location once a snapshot
    /// expiry removed that snapshot.
    ///
    /// Otherwise the update is reported as rejected only when the catalog refused it, as
    /// [`is_pointer_update_rejection`] decides. Every other outcome is an unknown commit state
    /// that names the new metadata location: a failed request that may still be in flight and
    /// apply later, a pointer that cannot be reloaded, a pointer that disappeared although the
    /// update expected one or current metadata that cannot be read.
    ///
    /// The caller keeps every file the commit wrote whatever this returns. A pointer that does
    /// not reference them now does not prove that no catalog state ever will.
    async fn reconcile_pointer_update(
        context: &Arc<TaskContext>,
        object_store: &Arc<dyn object_store::ObjectStore>,
        catalog_table: &[String],
        previous_metadata_location: Option<&str>,
        new_metadata_location: &str,
        new_snapshot_id: Option<i64>,
        error: DataFusionError,
    ) -> Result<PointerUpdateOutcome> {
        let current_location =
            match Self::load_catalog_metadata_location(context, catalog_table).await {
                Ok(location) => location,
                Err(reload_error) => {
                    return Err(pointer_update_state_unknown(
                        new_metadata_location,
                        &error,
                        format!("reloading the catalog pointer failed: {reload_error}"),
                    ));
                }
            };
        let committed = match current_location.as_deref() {
            Some(current_location) if current_location == new_metadata_location => true,
            Some(current_location) => match new_snapshot_id {
                Some(snapshot_id) => {
                    let current_metadata = load_metadata_file_bytes(object_store, current_location)
                        .await
                        .and_then(|bytes| {
                            TableMetadata::from_json(&bytes)
                                .map_err(|error| DataFusionError::External(Box::new(error)))
                        });
                    match current_metadata {
                        Ok(metadata) => {
                            metadata
                                .snapshots
                                .iter()
                                .any(|snapshot| snapshot.snapshot_id() == snapshot_id)
                                || metadata
                                    .metadata_log
                                    .iter()
                                    .any(|entry| entry.metadata_file == new_metadata_location)
                        }
                        Err(load_error) => {
                            return Err(pointer_update_state_unknown(
                                new_metadata_location,
                                &error,
                                format!(
                                    "the current catalog metadata {current_location} could not be read: {load_error}"
                                ),
                            ));
                        }
                    }
                }
                None => false,
            },
            // The catalog lookup reports a missing table as having no pointer, so a pointer that
            // disappeared after the update expected one tells nothing about the update.
            None if previous_metadata_location.is_some() => {
                return Err(pointer_update_state_unknown(
                    new_metadata_location,
                    &error,
                    "the catalog no longer reports a metadata pointer".to_string(),
                ));
            }
            None => false,
        };
        if committed {
            log::warn!(
                "Iceberg catalog pointer update to {new_metadata_location} reported an error but \
                 the catalog head is the new metadata or was built on it: {error}"
            );
            Ok(PointerUpdateOutcome::Committed)
        } else if is_pointer_update_rejection(&error, previous_metadata_location.is_some()) {
            Ok(PointerUpdateOutcome::Rejected(error))
        } else {
            Err(pointer_update_state_unknown(
                new_metadata_location,
                &error,
                "the catalog pointer does not reference the new metadata or its snapshot, but the \
                 request may still apply"
                    .to_string(),
            ))
        }
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
        schema: &IcebergSchema,
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
        let partition_type = default_spec
            .partition_type(schema)
            .map_err(DataFusionError::Plan)?;
        let touched_partitions = added_data_files
            .iter()
            .map(|file| Self::canonical_partition_key(file, &partition_type))
            .collect::<Result<HashSet<_>>>()?;
        let mut paths = live_data_files
            .iter()
            .filter_map(|file| {
                let key = Self::canonical_partition_key(file, &partition_type);
                match key {
                    Ok(key) if touched_partitions.contains(&key) => {
                        Some(Ok(file.file_path.clone()))
                    }
                    Ok(_) => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        paths.sort();
        paths.dedup();
        Ok(paths)
    }

    fn canonical_partition_key(
        file: &DataFile,
        partition_type: &StructType,
    ) -> Result<PartitionKey> {
        let values = file
            .partition
            .iter()
            .zip(partition_type.fields())
            .map(|(value, field)| {
                let Some(Literal::Primitive(value)) = value else {
                    return match value {
                        None => Ok(None),
                        Some(_) => Err(DataFusionError::Plan(
                            "Iceberg partition values must be primitive literals".to_string(),
                        )),
                    };
                };
                let Type::Primitive(expected_type) = field.field_type.as_ref() else {
                    return Err(DataFusionError::Plan(
                        "Iceberg partition fields must have primitive result types".to_string(),
                    ));
                };
                let value = expected_type
                    .promote_literal(value)
                    .ok_or_else(|| {
                        DataFusionError::Plan(format!(
                            "Iceberg partition value {value:?} is incompatible with {expected_type}"
                        ))
                    })?
                    .into_owned();
                Ok(Some(Literal::Primitive(value)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PartitionKey::new(file.partition_spec_id, &values))
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
        let incoming_removals = std::mem::take(&mut incoming.removed_data_file_paths);
        incoming
            .removed_data_file_paths
            .clone_from(&existing.removed_data_file_paths);
        incoming.row_count = existing.row_count;
        if existing != &incoming {
            return Err(DataFusionError::Internal(
                "inconsistent commit_meta actions from Iceberg writer partitions".to_string(),
            ));
        }
        existing.removed_data_file_paths.extend(incoming_removals);
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

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    #[expect(deprecated)]
    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: datafusion::physical_plan::ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.with_new_children(children)
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
            // Writer output may be replayed after publication. Commit attempts do not own
            // these task files, including when input consumption or publication fails.
            while let Some(batch_result) = data.next().await {
                let batch = batch_result?;
                if batch.num_rows() == 0 {
                    continue;
                }
                let (adds, deletes, meta) = decode_actions_and_meta_from_batch(&batch)?;
                added_data_files.extend(adds);
                added_delete_files.extend(deletes);
                for meta in meta {
                    Self::merge_writer_commit_meta(&mut commit_meta, meta)?;
                }
            }
            let has_publication_controls =
                caller_expected_snapshot_id.is_some() || !snapshot_properties.is_empty();
            // No-op path (e.g. IgnoreIfExists on existing table): no rows, no meta.
            if commit_meta.is_none() && added_data_files.is_empty() && added_delete_files.is_empty()
            {
                // A controlled write must validate its expectations, so it cannot be skipped.
                if has_publication_controls {
                    return Err(DataFusionError::Internal(
                        "controlled Iceberg write produced no commit metadata".to_string(),
                    ));
                }
                return commit_count_batch(schema, 0);
            }

            let commit_meta = commit_meta.ok_or_else(|| {
                DataFusionError::Internal(
                    "missing commit_meta action from writer output".to_string(),
                )
            })?;

            let skip_empty_commit = commit_meta.skip_empty_commit;
            let mut removed_data_file_paths = planned_removed_data_file_paths;
            removed_data_file_paths.extend(commit_meta.removed_data_file_paths);
            removed_data_file_paths.sort();
            removed_data_file_paths.dedup();

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
            if let Some(requirement) =
                caller_expected_snapshot_requirement(commit_info.caller_expected_snapshot_id)
                && !commit_info.requirements.contains(&requirement)
            {
                commit_info.requirements.push(requirement);
            }
            if !snapshot_update_kind.is_targeted_rewrite()
                && (dynamic_partition_overwrite || !removed_data_file_paths.is_empty())
            {
                return Err(DataFusionError::Internal(
                    "scoped overwrite requires a copy-on-write snapshot update".to_string(),
                ));
            }
            if dynamic_partition_overwrite && !removed_data_file_paths.is_empty() {
                return Err(DataFusionError::Internal(
                    "dynamic partition overwrite cannot carry planned removal paths".to_string(),
                ));
            }
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
            )?;
            let catalog_recorded_metadata_location =
                metadata_location_from_properties(&commit_info.table_properties)
                    .or_else(|| catalog_table_info.metadata_location.clone());
            // A catalog entry can precede the first metadata commit for a write planned without
            // a base table. Only that plan may initialize the catalog pointer with a CAS update.
            let initializes_catalog_metadata_pointer = expected_snapshot_id.is_none()
                && catalog_recorded_metadata_location.is_none()
                && catalog_commit_mode.uses_metadata_location_update();
            let catalog_metadata_location = if initializes_catalog_metadata_pointer {
                None
            } else {
                resolve_iceberg_metadata_location(
                    commit_info.lakehouse_table.as_ref(),
                    catalog_recorded_metadata_location.clone(),
                    catalog_table_info.is_catalog_managed_iceberg_table
                        || catalog_managed_iceberg_from_properties(&commit_info.table_properties),
                )?
            };

            // Managed external catalogs use the authoritative metadata-location.
            // Path tables may record metadata-location in the session catalog for display, but
            // their current state is discovered from the authoritative metadata directory listing.
            let latest_meta_res = if initializes_catalog_metadata_pointer {
                None
            } else {
                Some(match catalog_metadata_location.as_deref() {
                    Some(location) => Ok(metadata_location_to_object_path_string(location)?),
                    None => {
                        crate::table::find_latest_metadata_file(&object_store, &table_url).await
                    }
                })
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
            let filesystem_registration = catalog_table
                .as_deref()
                .filter(|_| matches!(catalog_commit_mode, IcebergCatalogCommitMode::Filesystem))
                .map(|catalog_table| FilesystemCatalogRegistration {
                    catalog_table,
                    table_properties: &commit_info.table_properties,
                    recorded_metadata_location: catalog_recorded_metadata_location.as_deref(),
                    builds_on_catalog_pointer: catalog_metadata_location.is_some(),
                });
            log::debug!(
                "Iceberg catalog commit context: table={:?}, metadata_location={:?}, mode={:?}",
                catalog_table,
                catalog_metadata_location,
                catalog_commit_mode
            );

            let initial_latest_meta = if let Some(Ok(path)) = latest_meta_res {
                path
            } else {
                Self::validate_requirements(None, &commit_info.requirements)?;
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
                            return Err(discard_unsent_bootstrap(
                                bootstrap_result,
                                &store_ctx,
                                error,
                            )
                            .await);
                        }
                    };
                    // The update is sent, so the bootstrap is kept whatever its outcome.
                    match Self::publish_catalog_pointer(
                        &context,
                        &object_store,
                        catalog_table,
                        &commit_info.table_properties,
                        None,
                        &new_metadata_location,
                        bootstrap_result.table_metadata.current_snapshot_id,
                    )
                    .await?
                    {
                        PointerUpdateOutcome::Committed => {}
                        PointerUpdateOutcome::Rejected(error) => return Err(error),
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
                    if let Some(registration) = &filesystem_registration {
                        Self::register_filesystem_metadata_location(
                            &context,
                            &object_store,
                            registration,
                            &table_url,
                            &bootstrap_result.metadata_file,
                            bootstrap_result.table_metadata.current_snapshot_id,
                        )
                        .await?;
                    }
                }

                return commit_count_batch(schema, commit_info.row_count);
            };

            let mut attempt = 0;
            loop {
                attempt += 1;
                let catalog_metadata_location = if attempt == 1 {
                    catalog_metadata_location.clone()
                } else if let Some(catalog_table) = catalog_metadata_table {
                    match Self::load_catalog_metadata_location(&context, catalog_table).await? {
                        Some(location) => Some(location),
                        // The catalog pointer is the head of this table. The metadata directory
                        // can hold the uncommitted metadata of an earlier attempt, so a retry
                        // never builds on a directory listing in its place.
                        None if catalog_metadata_location.is_some() => {
                            return Err(DataFusionError::Execution(format!(
                                "Iceberg catalog no longer reports a metadata location for \
                                 catalog-authoritative table {} on commit attempt {attempt}",
                                catalog_table.join(".")
                            )));
                        }
                        None => None,
                    }
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
                if dynamic_partition_overwrite {
                    let default_spec = table_meta.default_partition_spec().ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata has no default partition spec".to_string(),
                        )
                    })?;
                    let live_data_files =
                        Self::current_live_data_files(&store_ctx, &table_meta).await?;
                    let current_schema = table_meta.current_schema().ok_or_else(|| {
                        DataFusionError::Plan(
                            "Iceberg table metadata has no current schema".to_string(),
                        )
                    })?;
                    removed_data_file_paths = Self::dynamic_partition_overwrite_paths(
                        &commit_info.data_files,
                        &live_data_files,
                        default_spec,
                        current_schema,
                    )?;
                }
                if (skip_empty_commit
                    || (snapshot_update_kind.is_targeted_rewrite() && dynamic_partition_overwrite))
                    && commit_info.data_files.is_empty()
                    && commit_info.delete_files.is_empty()
                    && removed_data_file_paths.is_empty()
                {
                    // Skipping publishes no snapshot, so the snapshot properties would be
                    // dropped and the caller could not tell this write from one that never ran.
                    // A controlled write must publish what its controls describe.
                    if has_publication_controls {
                        return plan_err!(
                            "Iceberg write with publication controls changes no data and would \
                             publish no snapshot, so its snapshot properties and expected \
                             snapshot ID cannot be honored"
                        );
                    }
                    return commit_count_batch(schema, commit_info.row_count);
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
                    let spec = new_spec;
                    let spec_id = spec.spec_id();
                    let should_add_spec = !table_meta
                        .partition_specs
                        .iter()
                        .any(|partition_spec| partition_spec.spec_id() == spec_id);
                    let should_set_default_spec = table_meta.default_spec_id != spec_id;
                    Self::apply_partition_spec_update(&mut table_meta, spec.clone())?;
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
                let row_lineage_start_row_id = table_meta.row_lineage_start_row_id();

                // If metadata exists but there is no current snapshot (e.g. from a CREATE TABLE),
                // bootstrap the first snapshot as a normal metadata version.
                if maybe_snapshot.is_none() {
                    let mut catalog_fallback_table = catalog_metadata_update_table;
                    if let Some(catalog_table) = catalog_commit_table {
                        let mut prepared_snapshot = prepare_bootstrap_snapshot(
                            &table_url,
                            &store_ctx,
                            &commit_info,
                            &table_meta,
                        )
                        .await?;
                        let action_requirements =
                            prepared_snapshot.action_commit().requirements().to_vec();
                        if let Err(error) =
                            Self::validate_requirements(Some(&table_meta), &action_requirements)
                        {
                            prepared_snapshot.cleanup().await;
                            return Err(error);
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
                                prepared_snapshot.cleanup().await;
                                return Err(DataFusionError::Internal(
                                    "missing lakehouse context for Iceberg catalog commit"
                                        .to_string(),
                                ));
                            }
                        };
                        prepared_snapshot.publication_started();
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
                            Err(error) => return Err(error),
                        };
                        match catalog_outcome {
                            CatalogCommitOutcome::Committed(committed) => {
                                if let Some(metadata_location) = committed.metadata_location() {
                                    log::debug!(
                                        "Iceberg catalog commit returned metadata-location={metadata_location}"
                                    );
                                }
                                if committed.payload().is_some() {
                                    log::trace!("Iceberg catalog commit returned a payload");
                                }
                                prepared_snapshot.commit_succeeded();
                                return commit_count_batch(schema, commit_info.row_count);
                            }
                            CatalogCommitOutcome::NotSupported => {
                                prepared_snapshot.cleanup().await;
                                if matches!(
                                    catalog_commit_mode,
                                    IcebergCatalogCommitMode::CompatibilityCatalogCommit
                                ) {
                                    catalog_fallback_table = Some(catalog_table);
                                } else {
                                    return Err(DataFusionError::Plan(
                                        "Iceberg catalog commit is not supported by the resolved catalog authority"
                                            .to_string(),
                                    ));
                                }
                            }
                            CatalogCommitOutcome::Conflict => {
                                prepared_snapshot.cleanup().await;
                                if attempt >= MAX_COMMIT_RETRIES {
                                    return Err(commit_conflict_error());
                                }
                                continue;
                            }
                        }
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
                        // A concurrent writer created this metadata version first. Retry from
                        // the new head, which revalidates every snapshot requirement.
                        Err(error) if is_bootstrap_metadata_conflict(&error) => {
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
                                return Err(discard_unsent_bootstrap(
                                    bootstrap_result,
                                    &store_ctx,
                                    error,
                                )
                                .await);
                            }
                        };
                        // The update is sent, so the bootstrap is kept whatever its outcome.
                        match Self::publish_catalog_pointer(
                            &context,
                            &object_store,
                            catalog_table,
                            &commit_info.table_properties,
                            Some(previous_metadata_location),
                            &new_metadata_location,
                            bootstrap_result.table_metadata.current_snapshot_id,
                        )
                        .await?
                        {
                            PointerUpdateOutcome::Committed => {}
                            // The retry writes its metadata file, manifest list and manifests
                            // under new names, so this attempt's files stay behind as orphans
                            // and are never overwritten.
                            PointerUpdateOutcome::Rejected(error)
                                if is_retryable_catalog_conflict(&error) =>
                            {
                                if attempt >= MAX_COMMIT_RETRIES {
                                    return Err(commit_conflict_error());
                                }
                                continue;
                            }
                            PointerUpdateOutcome::Rejected(error) => return Err(error),
                        }
                    } else if let Some(registration) = &filesystem_registration {
                        Self::register_filesystem_metadata_location(
                            &context,
                            &object_store,
                            registration,
                            &table_url,
                            &bootstrap_result.metadata_file,
                            bootstrap_result.table_metadata.current_snapshot_id,
                        )
                        .await?;
                    }

                    return commit_count_batch(schema, commit_info.row_count);
                }

                let snapshot = maybe_snapshot.ok_or_else(|| {
                    DataFusionError::Plan("No current snapshot in table metadata".to_string())
                })?;

                let current_version = metadata_file_version_from_path(&latest_meta).unwrap_or(0);
                let next_version = current_version + 1;

                // Catalog commits are ordered by their expected metadata pointer, not
                // by unreferenced metadata objects left in the table directory.
                let existing_for_next = if catalog_commit_mode.uses_catalog_metadata() {
                    vec![]
                } else {
                    metadata_files_for_version(&store_ctx, next_version).await?
                };
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
                let mut prepared_snapshot = SnapshotProducer::new(
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
                .mark_dynamic_partition_overwrite(dynamic_partition_overwrite)
                .prepare(commit_info.snapshot_update_kind)
                .await
                .map_err(DataFusionError::Execution)?;

                // Apply updates (only handle the ones we emit: AddSnapshot, SetSnapshotRef)
                let action_requirements = prepared_snapshot.action_commit().requirements().to_vec();
                if let Err(error) =
                    Self::validate_requirements(Some(&table_meta), &action_requirements)
                {
                    prepared_snapshot.cleanup().await;
                    return Err(error);
                }
                let mut action_updates = prepared_snapshot.action_commit().updates().to_vec();
                for update in &mut action_updates {
                    if let TableUpdate::SetSnapshotRef {
                        ref_name,
                        reference,
                    } = update
                        && let Some(previous) = table_meta.refs.get(ref_name)
                    {
                        reference.retention = previous.retention.clone();
                    }
                }
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
                            prepared_snapshot.cleanup().await;
                            return Err(DataFusionError::Internal(
                                "missing lakehouse context for Iceberg catalog commit".to_string(),
                            ));
                        }
                    };
                    prepared_snapshot.publication_started();
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
                        Err(error) => return Err(error),
                    };
                    match catalog_outcome {
                        CatalogCommitOutcome::Committed(committed) => {
                            if let Some(metadata_location) = committed.metadata_location() {
                                log::debug!(
                                    "Iceberg catalog commit returned metadata-location={metadata_location}"
                                );
                            }
                            if committed.payload().is_some() {
                                log::trace!("Iceberg catalog commit returned a payload");
                            }
                            prepared_snapshot.commit_succeeded();
                            return commit_count_batch(schema, commit_info.row_count);
                        }
                        CatalogCommitOutcome::NotSupported
                            if matches!(
                                catalog_commit_mode,
                                IcebergCatalogCommitMode::CompatibilityCatalogCommit
                            ) =>
                        {
                            prepared_snapshot.publication_did_not_happen();
                        }
                        CatalogCommitOutcome::NotSupported => {
                            prepared_snapshot.cleanup().await;
                            return Err(DataFusionError::Plan(
                                "Iceberg catalog commit is not supported by the resolved catalog authority"
                                    .to_string(),
                            ));
                        }
                        CatalogCommitOutcome::Conflict => {
                            prepared_snapshot.cleanup().await;
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
                        prepared_snapshot.cleanup().await;
                        return Err(error);
                    }
                };

                log::trace!(
                    "Writing metadata: {} snapshot_id={:?} table_url={}",
                    metadata_file,
                    table_meta.current_snapshot_id,
                    table_url
                );

                let metadata_path = object_store::path::Path::from(metadata_file.as_str());
                let put_opts = object_store::PutOptions {
                    mode: object_store::PutMode::Create,
                    ..Default::default()
                };
                let metadata_bytes = Bytes::from(metadata_bytes);
                let payload = object_store::PutPayload::from(metadata_bytes.clone());
                prepared_snapshot.publication_started();
                match store_ctx
                    .prefixed
                    .put_opts(&metadata_path, payload, put_opts)
                    .await
                {
                    Ok(_) => {}
                    // A retried put whose first attempt landed reports its own file as
                    // existing. The file is read back before anything is removed, and an
                    // unreadable file fails the commit as an unknown state with every
                    // artifact kept.
                    Err(error @ object_store::Error::AlreadyExists { .. }) => {
                        match reconcile_existing_metadata_file(
                            &store_ctx,
                            &metadata_path,
                            &metadata_bytes,
                        )
                        .await?
                        {
                            ExistingMetadataFile::Written => {
                                log::warn!(
                                    "Metadata file {metadata_file} for version {next_version} was \
                                     reported as existing but holds the bytes this commit sent, \
                                     so the write landed: {error}"
                                );
                            }
                            ExistingMetadataFile::Concurrent => {
                                log::warn!(
                                    "Metadata file {} already exists for version {}. Retrying attempt {}",
                                    metadata_file,
                                    next_version,
                                    attempt
                                );
                                prepared_snapshot.publication_did_not_happen();
                                prepared_snapshot.cleanup().await;
                                if attempt >= MAX_COMMIT_RETRIES {
                                    return Err(commit_conflict_error());
                                }
                                continue;
                            }
                        }
                    }
                    Err(error) => {
                        return Err(DataFusionError::External(Box::new(error)));
                    }
                }
                let version_files = if catalog_commit_mode.uses_catalog_metadata() {
                    vec![]
                } else {
                    metadata_files_for_version(&store_ctx, next_version).await?
                };
                let conflict_after_write = version_files.iter().any(|path| path != &metadata_file);
                if conflict_after_write {
                    log::warn!(
                        "Concurrent metadata writes detected for version {}: {:?}. Retrying attempt {}",
                        next_version,
                        version_files,
                        attempt
                    );
                    match store_ctx.prefixed.delete(&metadata_path).await {
                        Ok(()) | Err(object_store::Error::NotFound { .. }) => {
                            prepared_snapshot.cleanup().await;
                        }
                        Err(error) => {
                            return Err(DataFusionError::Execution(format!(
                                "failed to remove conflicted Iceberg metadata file {metadata_file}; commit state is uncertain: {error}"
                            )));
                        }
                    }
                    if attempt >= MAX_COMMIT_RETRIES {
                        return Err(commit_conflict_error());
                    }
                    continue;
                }
                log::trace!("Metadata written successfully");
                prepared_snapshot.commit_succeeded();

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

                // The metadata file is written and nothing of this commit is deleted from here
                // on. A pointer update that reports an error is reconciled, not cleaned up, and
                // a filesystem-mode registration handles its own failure.
                if let Some(catalog_table) = catalog_metadata_update_table {
                    match Self::publish_catalog_pointer(
                        &context,
                        &object_store,
                        catalog_table,
                        &commit_info.table_properties,
                        catalog_metadata_location.as_deref(),
                        &metadata_location,
                        table_meta.current_snapshot_id,
                    )
                    .await?
                    {
                        PointerUpdateOutcome::Committed => {}
                        PointerUpdateOutcome::Rejected(error) => return Err(error),
                    }
                } else if let Some(registration) = &filesystem_registration {
                    Self::register_filesystem_metadata_location(
                        &context,
                        &object_store,
                        registration,
                        &table_url,
                        &metadata_file,
                        table_meta.current_snapshot_id,
                    )
                    .await?;
                }

                return commit_count_batch(schema, commit_info.row_count);
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
    use std::collections::HashMap;
    use std::ops::Range;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier, Mutex};

    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::prelude::SessionContext;
    use futures::stream::BoxStream;
    use futures::{StreamExt, TryStreamExt};
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
        ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use sail_catalog::error::{CatalogObject, CatalogResult};
    use sail_catalog::manager::{CatalogManager, CatalogManagerOptions};
    use sail_catalog::provider::{
        AlterTableOptions, CatalogProvider, CreateDatabaseOptions, CreateTableMode,
        CreateTableOptions, CreateViewOptions, DropDatabaseOptions, DropTableOptions,
        DropViewOptions, Namespace,
    };
    use sail_common_datafusion::catalog::managed::{
        metadata_location_update, previous_metadata_location_update,
    };
    use sail_common_datafusion::catalog::{
        CatalogProviderId, CatalogTableIdentity, CommitAuthority, DatabaseStatus,
        LakehouseAuthority, LakehouseFormat, LakehouseOperation, MetadataPointerAuthority,
        ScanAuthority, TableLifecycle, TableStatus,
    };

    use super::*;
    use crate::catalog_support::commit::catalog_table_info_from_status;
    use crate::physical_plan::action_schema::{
        encode_add_data_files, encode_commit_meta, iceberg_action_schema,
    };
    use crate::spec::metadata::table_metadata::MetadataLog;
    use crate::spec::transform::Transform;
    use crate::spec::types::values::{Literal, PrimitiveLiteral};
    use crate::spec::types::{NestedField, PrimitiveType, Type};
    use crate::spec::{
        DataContentType, DataFileFormat, FormatVersion, Operation, SnapshotBuilder,
        SnapshotReference, SnapshotRetention,
    };

    #[test]
    fn scoped_overwrite_initializes_lineage_after_schema_evolution() {
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
                removed_data_file_paths: vec![],
                skip_empty_commit: false,
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
            let commit = IcebergCommitExec::new(
                input,
                table_url.clone(),
                None,
                SnapshotUpdateKind::CopyOnWrite,
            );
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("file:///").expect("file store URL"),
                memory.clone(),
            );

            let mut output = commit
                .execute(0, context.task_ctx())
                .expect("commit stream");
            output
                .next()
                .await
                .expect("commit result")
                .expect("v3 scoped overwrite");
            let store: Arc<dyn ObjectStore> = memory;
            let location = crate::table::find_latest_metadata_file(&store, &table_url)
                .await
                .expect("metadata location");
            let bytes = load_metadata_file_bytes(&store, &location)
                .await
                .expect("metadata bytes");
            let metadata = TableMetadata::from_json(&bytes).expect("metadata");
            assert_eq!(metadata.format_version, FormatVersion::V3);
            assert!(metadata.next_row_id.is_some());
        });
    }

    fn partitioned_data_file(path: &str, spec_id: i32, value: i32) -> DataFile {
        partitioned_data_file_with_literal(path, spec_id, PrimitiveLiteral::Int(value))
    }

    fn partitioned_data_file_with_literal(
        path: &str,
        spec_id: i32,
        value: PrimitiveLiteral,
    ) -> DataFile {
        DataFile {
            content: DataContentType::Data,
            file_path: path.to_string(),
            file_format: DataFileFormat::Parquet,
            partition: vec![Some(Literal::Primitive(value))],
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

    fn identity_partition_schema(partition_type: PrimitiveType) -> IcebergSchema {
        IcebergSchema::builder()
            .with_fields([Arc::new(NestedField::optional(
                2,
                "part",
                Type::Primitive(partition_type),
            ))])
            .build()
            .expect("partition schema")
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
        let schema = identity_partition_schema(PrimitiveType::Int);
        let paths =
            IcebergCommitExec::dynamic_partition_overwrite_paths(&added, &live, &spec, &schema)
                .expect("dynamic overwrite paths");
        assert_eq!(paths, vec!["old-2.parquet"]);
    }

    #[test]
    fn dynamic_partition_overwrite_matches_distinct_nan_payloads() {
        let spec = identity_partition_spec();
        let added = vec![partitioned_data_file_with_literal(
            "new-nan.parquet",
            3,
            PrimitiveLiteral::Float(ordered_float::OrderedFloat(f32::from_bits(0x7fc0_0001))),
        )];
        let live = vec![partitioned_data_file_with_literal(
            "old-nan.parquet",
            3,
            PrimitiveLiteral::Float(ordered_float::OrderedFloat(f32::from_bits(0xffc0_0042))),
        )];
        let schema = identity_partition_schema(PrimitiveType::Float);

        let paths =
            IcebergCommitExec::dynamic_partition_overwrite_paths(&added, &live, &spec, &schema)
                .expect("dynamic overwrite paths");

        assert_eq!(paths, vec!["old-nan.parquet"]);
    }

    #[test]
    fn dynamic_partition_overwrite_rejects_mismatched_partition_spec() {
        let spec = identity_partition_spec();
        let added = vec![partitioned_data_file("new.parquet", 4, 2)];
        let schema = identity_partition_schema(PrimitiveType::Int);
        let error =
            IcebergCommitExec::dynamic_partition_overwrite_paths(&added, &[], &spec, &schema)
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
            &identity_partition_schema(PrimitiveType::Int),
        )
        .expect("empty dynamic overwrite");
        assert!(paths.is_empty());
    }

    #[derive(Debug)]
    enum MetadataWriteFault {
        Conflict(Bytes),
        LostAcknowledgement,
        /// The write lands, its response is lost and the store's retry of the same request
        /// reports that the file already exists.
        LandedThenExists,
        /// As [`Self::LandedThenExists`], and every later read of the file fails.
        LandedThenExistsUnreadable,
        /// The store reports that the file already exists, but it is gone when read.
        ExistsThenMissing,
    }

    #[derive(Debug)]
    struct FaultInjectingMetadataStore {
        memory_store: Arc<object_store::memory::InMemory>,
        fault: MetadataWriteFault,
        /// The suffix of the metadata file whose first write hits the fault.
        fault_path: &'static str,
        fault_injected: AtomicBool,
        deleted: Arc<Mutex<Vec<String>>>,
    }

    impl FaultInjectingMetadataStore {
        fn new(
            memory_store: Arc<object_store::memory::InMemory>,
            fault: MetadataWriteFault,
            fault_path: &'static str,
        ) -> Self {
            Self {
                memory_store,
                fault,
                fault_path,
                fault_injected: AtomicBool::new(false),
                deleted: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn deleted(&self) -> Vec<String> {
            self.deleted.lock().expect("deleted paths").clone()
        }
    }

    impl std::fmt::Display for FaultInjectingMetadataStore {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FaultInjectingMetadataStore")
        }
    }

    #[async_trait::async_trait]
    impl ObjectStore for FaultInjectingMetadataStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            if location.as_ref().ends_with(self.fault_path)
                && !self.fault_injected.swap(true, Ordering::SeqCst)
            {
                let already_exists = || object_store::Error::AlreadyExists {
                    path: location.to_string(),
                    source: std::io::Error::other("412 Precondition Failed on a retried PUT")
                        .into(),
                };
                match &self.fault {
                    MetadataWriteFault::Conflict(metadata) => {
                        self.memory_store
                            .put(location, PutPayload::from(metadata.clone()))
                            .await?;
                    }
                    MetadataWriteFault::LostAcknowledgement => {
                        self.memory_store.put_opts(location, payload, opts).await?;
                        return Err(object_store::Error::Generic {
                            store: "fault injection",
                            source: std::io::Error::other("lost metadata write acknowledgement")
                                .into(),
                        });
                    }
                    MetadataWriteFault::LandedThenExists
                    | MetadataWriteFault::LandedThenExistsUnreadable => {
                        self.memory_store.put_opts(location, payload, opts).await?;
                        return Err(already_exists());
                    }
                    MetadataWriteFault::ExistsThenMissing => return Err(already_exists()),
                }
            }
            self.memory_store.put_opts(location, payload, opts).await
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
            if matches!(self.fault, MetadataWriteFault::LandedThenExistsUnreadable)
                && self.fault_injected.load(Ordering::SeqCst)
                && location.as_ref().ends_with(self.fault_path)
            {
                return Err(object_store::Error::Generic {
                    store: "fault injection",
                    source: std::io::Error::other("injected metadata read failure").into(),
                });
            }
            self.memory_store.get_opts(location, options).await
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
            let deleted = Arc::clone(&self.deleted);
            self.memory_store.delete_stream(
                locations
                    .inspect(move |location| {
                        if let Ok(location) = location {
                            deleted
                                .lock()
                                .expect("deleted paths")
                                .push(location.to_string());
                        }
                    })
                    .boxed(),
            )
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
    fn copy_on_write_commit_accumulates_partition_removals() {
        let mut accumulated = None;
        let first = CommitMeta {
            table_uri: "file:///tmp/cow/".to_string(),
            row_count: 1,
            removed_data_file_paths: vec!["first.parquet".to_string()],
            skip_empty_commit: true,
            ..Default::default()
        };
        let mut second = first.clone();
        second.row_count = 2;
        second.removed_data_file_paths = vec!["second.parquet".to_string()];
        IcebergCommitExec::merge_writer_commit_meta(&mut accumulated, first.clone())
            .expect("first partition");
        IcebergCommitExec::merge_writer_commit_meta(&mut accumulated, second)
            .expect("second partition");
        let expected = CommitMeta {
            row_count: 3,
            removed_data_file_paths: vec![
                "first.parquet".to_string(),
                "second.parquet".to_string(),
            ],
            ..first.clone()
        };
        assert_eq!(accumulated, Some(expected));
        let incompatible = CommitMeta {
            skip_empty_commit: false,
            ..first
        };
        assert!(
            IcebergCommitExec::merge_writer_commit_meta(&mut accumulated, incompatible).is_err()
        );
    }

    #[test]
    fn empty_predicate_overwrite_validates_expected_snapshot() {
        for skip_empty_commit in [false, true] {
            futures::executor::block_on(async {
                let table_url =
                    Url::parse("file:///tmp/empty-predicate-overwrite/").expect("table URL");
                let memory = Arc::new(object_store::memory::InMemory::new());
                let store: Arc<dyn ObjectStore> = memory.clone();
                let store_ctx = StoreContext::new(store, &table_url).expect("store context");
                let iceberg_schema = IcebergSchema::builder()
                    .with_schema_id(1)
                    .with_fields([Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    ))])
                    .build()
                    .expect("schema");
                let table_properties = vec![("format-version".to_string(), "2".to_string())];
                crate::operations::bootstrap::bootstrap_empty_table_metadata(
                    &table_url,
                    &store_ctx,
                    iceberg_schema,
                    PartitionSpec::unpartitioned_spec(),
                    &table_properties,
                    NewTableMetadataStyle::Hadoop,
                )
                .await
                .expect("bootstrap metadata");

                let action_schema = iceberg_action_schema().expect("action schema");
                let action_batch = encode_commit_meta(CommitMeta {
                    table_uri: table_url.to_string(),
                    row_count: 0,
                    removed_data_file_paths: vec![],
                    skip_empty_commit,
                    requirements: vec![],
                    table_properties,
                    lakehouse_table: None,
                    schema: None,
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
                        .with_expected_snapshot_id(Some(Some(99)));
                let context = SessionContext::new();
                context.runtime_env().register_object_store(
                    &Url::parse("file:///").expect("file store URL"),
                    memory,
                );

                let mut output = commit
                    .execute(0, context.task_ctx())
                    .expect("commit stream");
                let error = output
                    .next()
                    .await
                    .expect("commit result")
                    .expect_err("stale empty overwrite must conflict");
                assert!(error.to_string().contains("expected snapshot Some(99)"));
            });
        }
    }

    #[test]
    fn replayed_commit_keeps_files_published_by_the_previous_attempt() {
        for lose_acknowledgement in [false, true] {
            futures::executor::block_on(async {
                let table_url = Url::parse("file:///tmp/replayed-commit/").expect("table URL");
                let memory = Arc::new(object_store::memory::InMemory::new());
                let store: Arc<dyn ObjectStore> = memory.clone();
                let store_ctx = StoreContext::new(store, &table_url).expect("store context");
                let iceberg_schema = IcebergSchema::builder()
                    .with_fields([Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    ))])
                    .build()
                    .expect("schema");
                let table_properties = vec![("format-version".to_string(), "2".to_string())];
                crate::operations::bootstrap::bootstrap_empty_table_metadata(
                    &table_url,
                    &store_ctx,
                    iceberg_schema,
                    PartitionSpec::unpartitioned_spec(),
                    &table_properties,
                    NewTableMetadataStyle::Hadoop,
                )
                .await
                .expect("bootstrap metadata");
                let task_path = Path::from("data/task.parquet");
                store_ctx
                    .prefixed
                    .put(&task_path, Bytes::from_static(b"task data").into())
                    .await
                    .expect("write task file");
                let mut data_file = partitioned_data_file("data/task.parquet", 0, 0);
                data_file.partition.clear();
                let action_schema = iceberg_action_schema().expect("action schema");
                let actions = datafusion::arrow::compute::concat_batches(
                    &action_schema,
                    &[
                        encode_add_data_files(vec![data_file]).expect("add action"),
                        encode_commit_meta(CommitMeta {
                            table_uri: table_url.to_string(),
                            row_count: 1,
                            table_properties,
                            ..Default::default()
                        })
                        .expect("commit metadata"),
                    ],
                )
                .expect("writer actions");
                let input = MemorySourceConfig::try_new_exec(&[vec![actions]], action_schema, None)
                    .expect("replayable writer output");
                let commit = IcebergCommitExec::new(
                    input,
                    table_url.clone(),
                    None,
                    SnapshotUpdateKind::CopyOnWrite,
                )
                .with_expected_snapshot_id(Some(None));
                let context = SessionContext::new();
                let publication_store: Arc<dyn ObjectStore> = if lose_acknowledgement {
                    Arc::new(FaultInjectingMetadataStore::new(
                        memory.clone(),
                        MetadataWriteFault::LostAcknowledgement,
                        "metadata/v2.metadata.json",
                    ))
                } else {
                    memory.clone()
                };
                context.runtime_env().register_object_store(
                    &Url::parse("file:///").expect("file store URL"),
                    publication_store,
                );
                let result = commit
                    .execute(0, context.task_ctx())
                    .expect("first attempt")
                    .try_collect::<Vec<_>>()
                    .await;
                if lose_acknowledgement {
                    result.expect_err("publication succeeded but its acknowledgement was lost");
                } else {
                    let batches = result.expect("published first attempt");
                    assert_eq!(batches[0].num_rows(), 1);
                }
                store_ctx
                    .prefixed
                    .head(&task_path)
                    .await
                    .expect("publication errors must retain possibly committed task files");

                let error = commit
                    .execute(0, context.task_ctx())
                    .expect("replayed attempt")
                    .try_collect::<Vec<_>>()
                    .await
                    .expect_err("replayed overwrite has a stale snapshot");
                assert!(error.to_string().contains("expected snapshot None"));
                store_ctx
                    .prefixed
                    .head(&task_path)
                    .await
                    .expect("a rejected retry must retain the published task file");
                let store: Arc<dyn ObjectStore> = memory;
                let location = crate::table::find_latest_metadata_file(&store, &table_url)
                    .await
                    .expect("committed metadata location");
                let bytes = load_metadata_file_bytes(&store, &location)
                    .await
                    .expect("committed metadata bytes");
                let metadata = TableMetadata::from_json(&bytes).expect("committed metadata");
                assert_eq!(metadata.snapshots.len(), 1);
                let snapshot = &metadata.snapshots[0];
                let manifests = load_manifest_list(&store_ctx, snapshot.manifest_list())
                    .await
                    .expect("committed manifest list");
                let manifest = load_manifest(&store_ctx, &manifests.entries()[0].manifest_path)
                    .await
                    .expect("committed manifest");
                assert_eq!(
                    manifest.entries()[0].data_file.file_path,
                    "data/task.parquet"
                );
            });
        }
    }

    #[test]
    fn metadata_conflict_refreshes_listing_and_commits_the_next_version() {
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
                        removed_data_file_paths: vec![],
                        skip_empty_commit: false,
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
                IcebergCommitExec::new(input, table_url, None, SnapshotUpdateKind::FastAppend);
            let conflict_store = Arc::new(FaultInjectingMetadataStore::new(
                Arc::clone(&memory),
                MetadataWriteFault::Conflict(metadata_bytes),
                "metadata/v2.metadata.json",
            ));
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("file:///").expect("file store URL"),
                conflict_store.clone(),
            );

            let mut output = commit
                .execute(0, context.task_ctx())
                .expect("commit stream");
            let batch = output
                .next()
                .await
                .expect("commit result")
                .expect("commit must retry from the authoritative metadata listing");

            assert_eq!(batch.num_rows(), 1);
            assert!(conflict_store.fault_injected.load(Ordering::SeqCst));
            let metadata_prefix = Path::from("tmp/commit-conflict/metadata");
            let metadata_objects = memory
                .list(Some(&metadata_prefix))
                .try_collect::<Vec<_>>()
                .await
                .expect("metadata listing");
            let metadata_paths = metadata_objects
                .iter()
                .map(|object| object.location.as_ref())
                .collect::<Vec<_>>();
            assert!(
                metadata_paths
                    .iter()
                    .any(|path| path.ends_with("metadata/v2.metadata.json"))
            );
            assert!(
                metadata_paths
                    .iter()
                    .any(|path| path.ends_with("metadata/v3.metadata.json"))
            );
            store_ctx
                .prefixed
                .head(&task_file_path)
                .await
                .expect("successfully committed task file");
        });
    }

    fn metadata_at_snapshot_bytes(
        metadata: &TableMetadata,
        metadata_file: &str,
        schema_id: i32,
        snapshot_id: i64,
        sequence_number: i64,
    ) -> Bytes {
        let snapshot = SnapshotBuilder::new()
            .with_snapshot_id(snapshot_id)
            .with_sequence_number(sequence_number)
            .with_timestamp_ms(123 + sequence_number)
            .with_manifest_list("")
            .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
            .with_schema_id(schema_id)
            .build()
            .expect("snapshot");
        let mut metadata = metadata.clone();
        metadata.last_sequence_number = sequence_number;
        metadata.current_snapshot_id = Some(snapshot_id);
        metadata.snapshots.push(snapshot.clone());
        metadata.snapshot_log.push(SnapshotLog {
            timestamp_ms: snapshot.timestamp_ms,
            snapshot_id,
        });
        metadata.refs.insert(
            MAIN_BRANCH.to_string(),
            SnapshotReference {
                snapshot_id,
                retention: SnapshotRetention::Branch {
                    min_snapshots_to_keep: None,
                    max_snapshot_age_ms: None,
                    max_ref_age_ms: None,
                },
            },
        );
        let json = metadata.to_json().expect("metadata JSON");
        Bytes::from(encode_metadata_file(metadata_file, &json).expect("metadata bytes"))
    }

    #[test]
    fn caller_expected_snapshot_rejects_a_head_that_advanced_before_publication() {
        futures::executor::block_on(async {
            let table_url = Url::parse("file:///tmp/caller-expected-snapshot/").expect("table URL");
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
            let table_properties = vec![("format-version".to_string(), "2".to_string())];
            let bootstrap = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                iceberg_schema.clone(),
                PartitionSpec::builder().with_spec_id(1).build(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");
            let current_bytes = metadata_at_snapshot_bytes(
                &bootstrap.table_metadata,
                &bootstrap.metadata_file,
                iceberg_schema.schema_id(),
                17,
                1,
            );
            store_ctx
                .prefixed
                .put(
                    &Path::from(bootstrap.metadata_file.as_str()),
                    PutPayload::from(current_bytes.clone()),
                )
                .await
                .expect("current metadata");
            let current_metadata = TableMetadata::from_json(&current_bytes).expect("metadata");
            // A concurrent writer publishes snapshot 18 as the next metadata version just
            // before this commit writes its own candidate.
            let concurrent_bytes = metadata_at_snapshot_bytes(
                &current_metadata,
                "metadata/v2.metadata.json",
                iceberg_schema.schema_id(),
                18,
                2,
            );
            let action_schema = iceberg_action_schema().expect("action schema");
            let action_batch = encode_commit_meta(CommitMeta {
                table_uri: table_url.to_string(),
                table_properties,
                ..Default::default()
            })
            .expect("commit metadata action");
            let input = MemorySourceConfig::try_new_exec(
                &[vec![action_batch]],
                Arc::clone(&action_schema),
                None,
            )
            .expect("memory input");
            let commit: Arc<dyn ExecutionPlan> = Arc::new(
                IcebergCommitExec::new(input, table_url, None, SnapshotUpdateKind::FastAppend)
                    .with_caller_expected_snapshot_id(Some(17))
                    .with_snapshot_properties(vec![(
                        "hoist.publication-id".to_string(),
                        "publication-1".to_string(),
                    )]),
            );
            let conflict_store = Arc::new(FaultInjectingMetadataStore::new(
                Arc::clone(&memory),
                MetadataWriteFault::Conflict(concurrent_bytes.clone()),
                "metadata/v2.metadata.json",
            ));
            let context = SessionContext::new();
            context.runtime_env().register_object_store(
                &Url::parse("file:///").expect("file store URL"),
                conflict_store.clone(),
            );

            let error = datafusion::physical_plan::collect(commit, context.task_ctx())
                .await
                .expect_err("the advanced head must reject the caller expectation");

            assert!(conflict_store.fault_injected.load(Ordering::SeqCst));
            assert!(
                error.to_string().contains("expected snapshot Some(17)")
                    && error.to_string().contains("found Some(18)"),
                "{error}"
            );
            let metadata_objects = memory
                .list(Some(&Path::from("tmp/caller-expected-snapshot/metadata")))
                .map_ok(|object| object.location.to_string())
                .try_collect::<Vec<_>>()
                .await
                .expect("metadata listing");
            assert!(
                metadata_objects
                    .iter()
                    .all(|path| path.ends_with("v1.metadata.json")
                        || path.ends_with("v2.metadata.json")
                        || path.ends_with("version-hint.text")),
                "the rejected commit left artifacts behind: {metadata_objects:?}"
            );
            let published = memory
                .get(&Path::from(
                    "tmp/caller-expected-snapshot/metadata/v2.metadata.json",
                ))
                .await
                .expect("concurrent metadata")
                .bytes()
                .await
                .expect("concurrent metadata bytes");
            assert_eq!(published, concurrent_bytes);
        });
    }

    /// A fault that a catalog metadata pointer update hits.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PointerUpdateFault {
        /// The update applies, then its response is lost.
        AppliedResponseLost,
        /// The update applies, another writer commits on top of it and keeps its snapshot, then
        /// the response is lost. The new metadata log does not name the update, as when the
        /// writer records locations in another form or trims the log.
        AppliedThenAdvanced,
        /// The update applies, another writer commits on top of it, names it in the metadata
        /// log and expires its snapshot, then the response is lost.
        AppliedThenAdvancedAndExpired,
        /// The update applies, its response is lost and every later catalog read fails.
        AppliedThenReloadFails,
        /// The update applies, then the request times out.
        AppliedThenTimedOut,
        /// The request times out before the catalog applies it, and may still be in flight.
        NotApplied,
        /// The catalog refuses the update.
        Refused,
        /// The first update loses a compare-and-swap race without applying.
        ConflictOnce,
        /// The first update conflicts without applying, and every later catalog read reports
        /// the table as missing.
        ConflictThenTableMissing,
        /// The first update conflicts without applying. The read that reconciles it still finds
        /// the table, and every later catalog read reports the table as missing.
        ConflictThenTableMissingOnRetry,
    }

    /// The catalog reads [`PointerFaultCatalog`] answers before it reports the table missing.
    const TABLE_PRESENT: usize = usize::MAX;

    const POINTER_TABLE_URL: &str = "memory://bootstrap-pointer/table/";
    const POINTER_DATA_FILE: &str = "memory://bootstrap-pointer/table/data/task.parquet";
    const APPENDED_DATA_FILE: &str = "memory://bootstrap-pointer/table/data/appended.parquet";
    const ADVANCED_METADATA_LOCATION: &str =
        "memory://bootstrap-pointer/table/metadata/00009-advanced.metadata.json";

    /// A memory catalog whose metadata pointer updates hit a [`PointerUpdateFault`].
    struct PointerFaultCatalog {
        inner: sail_catalog_memory::MemoryCatalogProvider,
        memory_store: Arc<object_store::memory::InMemory>,
        fault: PointerUpdateFault,
        /// Whether pointer updates hit the fault. Updates pass through while it is unset.
        armed: AtomicBool,
        updates: AtomicUsize,
        reads_fail: AtomicBool,
        /// Table reads left before every read reports the table missing, or [`TABLE_PRESENT`].
        reads_before_missing: AtomicUsize,
    }

    impl PointerFaultCatalog {
        /// A retried compare-and-swap update finds the pointer already moved and conflicts. An
        /// unconditional update reports the timeout that lost its response.
        fn lost_response_error(properties: &[(String, String)]) -> CatalogError {
            if previous_metadata_location_update(properties).is_some() {
                CatalogError::Conflict("base metadata location changed".to_string())
            } else {
                CatalogError::External("request timed out".to_string())
            }
        }

        /// Commit another snapshot on top of the metadata at `location`, as a concurrent
        /// writer would, and move the pointer to it. With `expire`, the commit names `location`
        /// in its metadata log and expires the snapshot that was current there, so only the log
        /// still shows that it was built on `location`.
        async fn advance(
            &self,
            database: &Namespace,
            table: &str,
            location: &str,
            expire: bool,
        ) -> CatalogResult<()> {
            let store: Arc<dyn ObjectStore> = self.memory_store.clone();
            let bytes = load_metadata_file_bytes(&store, location)
                .await
                .expect("published metadata");
            let mut metadata = TableMetadata::from_json(&bytes).expect("published metadata");
            let parent = metadata
                .current_snapshot()
                .cloned()
                .expect("published snapshot");
            let snapshot = SnapshotBuilder::new()
                .with_snapshot_id(parent.snapshot_id() ^ 1)
                .with_parent_snapshot_id(parent.snapshot_id())
                .with_sequence_number(parent.sequence_number() + 1)
                .with_timestamp_ms(parent.timestamp_ms() + 1)
                .with_manifest_list(parent.manifest_list())
                .with_summary(crate::spec::snapshots::Summary::new(Operation::Append))
                .with_schema_id(metadata.current_schema_id)
                .build()
                .expect("advanced snapshot");
            metadata.last_sequence_number = snapshot.sequence_number();
            metadata.current_snapshot_id = Some(snapshot.snapshot_id());
            metadata.refs.insert(
                MAIN_BRANCH.to_string(),
                SnapshotReference {
                    snapshot_id: snapshot.snapshot_id(),
                    retention: SnapshotRetention::Branch {
                        min_snapshots_to_keep: None,
                        max_snapshot_age_ms: None,
                        max_ref_age_ms: None,
                    },
                },
            );
            metadata.snapshots.push(snapshot);
            if expire {
                metadata.metadata_log.push(MetadataLog {
                    timestamp_ms: metadata.last_updated_ms,
                    metadata_file: location.to_string(),
                });
                metadata
                    .snapshots
                    .retain(|snapshot| snapshot.snapshot_id() != parent.snapshot_id());
                metadata
                    .snapshot_log
                    .retain(|entry| entry.snapshot_id != parent.snapshot_id());
            }
            let json = metadata.to_json().expect("advanced metadata JSON");
            let path = metadata_location_to_object_path_string(ADVANCED_METADATA_LOCATION)
                .expect("advanced metadata path");
            self.memory_store
                .put(
                    &Path::from(path.as_str()),
                    PutPayload::from(Bytes::from(
                        encode_metadata_file(&path, &json).expect("advanced metadata bytes"),
                    )),
                )
                .await
                .expect("advanced metadata");
            self.inner
                .alter_table(
                    database,
                    table,
                    AlterTableOptions::SetTableProperties {
                        properties: vec![
                            (
                                "metadata_location".to_string(),
                                ADVANCED_METADATA_LOCATION.to_string(),
                            ),
                            (
                                "previous_metadata_location".to_string(),
                                location.to_string(),
                            ),
                        ],
                    },
                )
                .await
        }
    }

    #[async_trait::async_trait]
    impl CatalogProvider for PointerFaultCatalog {
        fn get_name(&self) -> &str {
            self.inner.get_name()
        }

        async fn create_database(
            &self,
            database: &Namespace,
            options: CreateDatabaseOptions,
        ) -> CatalogResult<DatabaseStatus> {
            self.inner.create_database(database, options).await
        }

        async fn get_database(&self, database: &Namespace) -> CatalogResult<DatabaseStatus> {
            self.inner.get_database(database).await
        }

        async fn list_databases(
            &self,
            prefix: Option<&Namespace>,
        ) -> CatalogResult<Vec<DatabaseStatus>> {
            self.inner.list_databases(prefix).await
        }

        async fn drop_database(
            &self,
            database: &Namespace,
            options: DropDatabaseOptions,
        ) -> CatalogResult<()> {
            self.inner.drop_database(database, options).await
        }

        async fn create_table(
            &self,
            database: &Namespace,
            table: &str,
            options: CreateTableOptions,
        ) -> CatalogResult<TableStatus> {
            self.inner.create_table(database, table, options).await
        }

        async fn get_table(&self, database: &Namespace, table: &str) -> CatalogResult<TableStatus> {
            if self.reads_fail.load(Ordering::SeqCst) {
                return Err(CatalogError::External("catalog unavailable".to_string()));
            }
            match self.reads_before_missing.load(Ordering::SeqCst) {
                TABLE_PRESENT => {}
                0 => {
                    return Err(CatalogError::NotFound(
                        CatalogObject::Table,
                        table.to_string(),
                    ));
                }
                reads => self.reads_before_missing.store(reads - 1, Ordering::SeqCst),
            }
            self.inner.get_table(database, table).await
        }

        async fn list_tables(&self, database: &Namespace) -> CatalogResult<Vec<TableStatus>> {
            self.inner.list_tables(database).await
        }

        async fn drop_table(
            &self,
            database: &Namespace,
            table: &str,
            options: DropTableOptions,
        ) -> CatalogResult<()> {
            self.inner.drop_table(database, table, options).await
        }

        async fn alter_table(
            &self,
            database: &Namespace,
            table: &str,
            options: AlterTableOptions,
        ) -> CatalogResult<()> {
            let AlterTableOptions::SetTableProperties { properties } = &options else {
                return self.inner.alter_table(database, table, options).await;
            };
            if !self.armed.load(Ordering::SeqCst) {
                return self.inner.alter_table(database, table, options).await;
            }
            let properties = properties.clone();
            let first_update = self.updates.fetch_add(1, Ordering::SeqCst) == 0;
            match self.fault {
                PointerUpdateFault::NotApplied => {
                    Err(CatalogError::External("request timed out".to_string()))
                }
                PointerUpdateFault::Refused => Err(CatalogError::Forbidden(
                    "not allowed to alter the table".to_string(),
                )),
                PointerUpdateFault::ConflictOnce if first_update => Err(CatalogError::Conflict(
                    "base metadata location changed".to_string(),
                )),
                PointerUpdateFault::ConflictThenTableMissing
                | PointerUpdateFault::ConflictThenTableMissingOnRetry
                    if first_update =>
                {
                    let reads = usize::from(
                        self.fault == PointerUpdateFault::ConflictThenTableMissingOnRetry,
                    );
                    self.reads_before_missing.store(reads, Ordering::SeqCst);
                    Err(CatalogError::Conflict(
                        "base metadata location changed".to_string(),
                    ))
                }
                PointerUpdateFault::ConflictOnce
                | PointerUpdateFault::ConflictThenTableMissing
                | PointerUpdateFault::ConflictThenTableMissingOnRetry => {
                    self.inner.alter_table(database, table, options).await
                }
                PointerUpdateFault::AppliedResponseLost
                | PointerUpdateFault::AppliedThenAdvanced
                | PointerUpdateFault::AppliedThenAdvancedAndExpired
                | PointerUpdateFault::AppliedThenReloadFails
                | PointerUpdateFault::AppliedThenTimedOut => {
                    self.inner.alter_table(database, table, options).await?;
                    if matches!(
                        self.fault,
                        PointerUpdateFault::AppliedThenAdvanced
                            | PointerUpdateFault::AppliedThenAdvancedAndExpired
                    ) {
                        let location = metadata_location_update(&properties)
                            .expect("updated metadata location");
                        let expire =
                            self.fault == PointerUpdateFault::AppliedThenAdvancedAndExpired;
                        self.advance(database, table, location, expire).await?;
                    }
                    if self.fault == PointerUpdateFault::AppliedThenReloadFails {
                        self.reads_fail.store(true, Ordering::SeqCst);
                    }
                    if self.fault == PointerUpdateFault::AppliedThenTimedOut {
                        return Err(CatalogError::External("request timed out".to_string()));
                    }
                    Err(Self::lost_response_error(&properties))
                }
            }
        }

        async fn create_view(
            &self,
            database: &Namespace,
            view: &str,
            options: CreateViewOptions,
        ) -> CatalogResult<TableStatus> {
            self.inner.create_view(database, view, options).await
        }

        async fn get_view(&self, database: &Namespace, view: &str) -> CatalogResult<TableStatus> {
            self.inner.get_view(database, view).await
        }

        async fn list_views(&self, database: &Namespace) -> CatalogResult<Vec<TableStatus>> {
            self.inner.list_views(database).await
        }

        async fn drop_view(
            &self,
            database: &Namespace,
            view: &str,
            options: DropViewOptions,
        ) -> CatalogResult<()> {
            self.inner.drop_view(database, view, options).await
        }
    }

    /// Where readers of a [`pointer_fixture`] table find its head.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum PointerMode {
        /// The catalog pointer is the head, and a commit moves it with a compare-and-swap update.
        CatalogPointer,
        /// The metadata directory listing is the head, and a commit records it in the catalog.
        FilesystemRegistered,
        /// A filesystem-mode table whose write was planned as catalog managed while its catalog
        /// entry is not. Readers list the metadata directory, but a commit builds on the
        /// metadata the catalog pointer names and moves the pointer with a compare-and-swap
        /// update.
        FilesystemOnCatalogPointer,
    }

    impl PointerMode {
        fn authority(self) -> LakehouseAuthority {
            match self {
                Self::CatalogPointer => LakehouseAuthority::CatalogAuthoritative {
                    lifecycle: TableLifecycle::External,
                    pointer: MetadataPointerAuthority::CatalogPropertyCas,
                    commit: CommitAuthority::IcebergMetadataLocationCas,
                },
                Self::FilesystemRegistered | Self::FilesystemOnCatalogPointer => {
                    LakehouseAuthority::CatalogRegistered {
                        lifecycle: TableLifecycle::External,
                        pointer: MetadataPointerAuthority::StorageDiscovery,
                        commit: CommitAuthority::Filesystem,
                    }
                }
            }
        }

        /// The table properties a write is planned with.
        fn table_properties(self) -> Vec<(String, String)> {
            let mut properties = vec![("format-version".to_string(), "2".to_string())];
            if self == Self::FilesystemOnCatalogPointer {
                properties.push(("table_type".to_string(), "ICEBERG".to_string()));
            }
            properties
        }
    }

    struct PointerFixture {
        mode: PointerMode,
        memory: Arc<object_store::memory::InMemory>,
        catalog: Arc<PointerFaultCatalog>,
        database: Namespace,
        store_ctx: StoreContext,
        initial_metadata_location: Option<String>,
        /// The objects that existed before the commit under test.
        preexisting_objects: Vec<String>,
        context: SessionContext,
        table_url: Url,
        lakehouse_table: LakehouseExecutionContext,
        schema: IcebergSchema,
        table_properties: Vec<(String, String)>,
    }

    impl PointerFixture {
        /// Append `data_file` with three rows to the table.
        async fn commit(&self, data_file: &str) -> Result<Vec<RecordBatch>> {
            let mut data_file = partitioned_data_file(data_file, 0, 0);
            data_file.partition.clear();
            data_file.record_count = 3;
            let action_schema = iceberg_action_schema().expect("action schema");
            let actions = datafusion::arrow::compute::concat_batches(
                &action_schema,
                &[
                    encode_add_data_files(vec![data_file]).expect("add action"),
                    encode_commit_meta(CommitMeta {
                        table_uri: self.table_url.to_string(),
                        row_count: 3,
                        table_properties: self.table_properties.clone(),
                        schema: Some(self.schema.clone()),
                        ..Default::default()
                    })
                    .expect("commit metadata action"),
                ],
            )
            .expect("writer actions");
            let input = MemorySourceConfig::try_new_exec(&[vec![actions]], action_schema, None)
                .expect("writer output");
            let commit: Arc<dyn ExecutionPlan> = Arc::new(IcebergCommitExec::new(
                input,
                self.table_url.clone(),
                Some(self.lakehouse_table.clone()),
                SnapshotUpdateKind::FastAppend,
            ));
            datafusion::physical_plan::collect(commit, self.context.task_ctx()).await
        }

        async fn objects(&self) -> Vec<String> {
            self.memory
                .list(None)
                .map_ok(|object| object.location.to_string())
                .try_collect::<Vec<_>>()
                .await
                .expect("list objects")
        }

        /// The metadata location the catalog names, read past the injected faults.
        async fn catalog_pointer(&self) -> Option<String> {
            let status = self
                .catalog
                .inner
                .get_table(&self.database, "items")
                .await
                .expect("catalog table");
            catalog_table_info_from_status(&status).metadata_location
        }

        /// The sorted data files of the table head, loading every file on the way. Like a
        /// reader, this finds the head through the catalog pointer, or through the metadata
        /// directory listing in filesystem mode.
        async fn committed_data_files(&self) -> Vec<String> {
            let store: Arc<dyn ObjectStore> = self.memory.clone();
            let location = match self.mode {
                PointerMode::CatalogPointer => {
                    self.catalog_pointer().await.expect("published pointer")
                }
                PointerMode::FilesystemRegistered | PointerMode::FilesystemOnCatalogPointer => {
                    crate::table::find_latest_metadata_file(&store, &self.table_url)
                        .await
                        .expect("listed metadata")
                }
            };
            let bytes = load_metadata_file_bytes(&store, &location)
                .await
                .expect("published metadata");
            let metadata = TableMetadata::from_json(&bytes).expect("published metadata");
            let snapshot = metadata.current_snapshot().expect("published snapshot");
            let manifests = load_manifest_list(&self.store_ctx, snapshot.manifest_list())
                .await
                .expect("published manifest list");
            let mut paths = Vec::new();
            for manifest_file in manifests.entries() {
                let manifest = load_manifest(&self.store_ctx, &manifest_file.manifest_path)
                    .await
                    .expect("published manifest");
                paths.extend(
                    manifest
                        .entries()
                        .iter()
                        .map(|entry| entry.data_file.file_path.clone()),
                );
            }
            paths.sort();
            paths
        }

        /// The metadata files and manifest lists the commit under test wrote.
        async fn written_artifacts(&self) -> (Vec<String>, Vec<String>) {
            let objects = self
                .objects()
                .await
                .into_iter()
                .filter(|path| !self.preexisting_objects.contains(path))
                .filter(|path| !ADVANCED_METADATA_LOCATION.ends_with(path.as_str()))
                .collect::<Vec<_>>();
            let metadata_files = objects
                .iter()
                .filter(|path| path.ends_with(".metadata.json"))
                .cloned()
                .collect();
            let manifest_lists = objects
                .iter()
                .filter(|path| path.contains("metadata/snap-"))
                .cloned()
                .collect();
            (metadata_files, manifest_lists)
        }
    }

    /// A catalog table in `mode` whose metadata pointer updates hit `fault`.
    ///
    /// Without existing metadata a commit bootstraps a new table. With it, the catalog already
    /// names empty table metadata and a commit bootstraps the first snapshot inside its
    /// retry loop.
    async fn pointer_fixture(
        mode: PointerMode,
        fault: PointerUpdateFault,
        existing_metadata: bool,
    ) -> PointerFixture {
        let table_url = Url::parse(POINTER_TABLE_URL).expect("table URL");
        let memory = Arc::new(object_store::memory::InMemory::new());
        let store: Arc<dyn ObjectStore> = memory.clone();
        let store_ctx = StoreContext::new(store, &table_url).expect("store context");
        let schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields([Arc::new(NestedField::required(
                1,
                "id",
                Type::Primitive(PrimitiveType::Int),
            ))])
            .build()
            .expect("schema");
        let table_properties = mode.table_properties();
        let initial_metadata_location = if existing_metadata {
            let empty = crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                schema.clone(),
                PartitionSpec::unpartitioned_spec(),
                &table_properties,
                NewTableMetadataStyle::Uuid,
            )
            .await
            .expect("empty table metadata");
            Some(
                table_metadata_location(&table_url, &empty.metadata_file)
                    .expect("empty table metadata location"),
            )
        } else {
            None
        };

        let database: Namespace = vec![Arc::<str>::from("default")]
            .try_into()
            .expect("database namespace");
        let inner = sail_catalog_memory::MemoryCatalogProvider::new(
            "sail".to_string(),
            database.clone(),
            None,
        );
        inner
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
                    properties: initial_metadata_location
                        .iter()
                        .map(|location| ("metadata_location".to_string(), location.clone()))
                        .collect(),
                    is_external: true,
                    is_write_precondition: false,
                },
            )
            .await
            .expect("catalog table");
        let catalog = Arc::new(PointerFaultCatalog {
            inner,
            memory_store: Arc::clone(&memory),
            fault,
            armed: AtomicBool::new(true),
            updates: AtomicUsize::new(0),
            reads_fail: AtomicBool::new(false),
            reads_before_missing: AtomicUsize::new(TABLE_PRESENT),
        });
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
            mode.authority(),
            ScanAuthority::ClientLakeSource,
        );
        let mut state = datafusion::execution::SessionStateBuilder::new().build();
        state.config_mut().set_extension(Arc::new(catalog_manager));
        let context = SessionContext::new_with_state(state);
        context.runtime_env().register_object_store(
            &Url::parse("memory://bootstrap-pointer").expect("store URL"),
            memory.clone(),
        );

        let mut fixture = PointerFixture {
            mode,
            memory,
            catalog,
            database,
            store_ctx,
            initial_metadata_location,
            preexisting_objects: vec![],
            context,
            table_url,
            lakehouse_table,
            schema,
            table_properties,
        };
        fixture.preexisting_objects = fixture.objects().await;
        fixture
    }

    /// Bootstrap a catalog table whose metadata pointer update hits `fault`, as described on
    /// [`pointer_fixture`].
    async fn run_bootstrap_with_pointer_fault(
        mode: PointerMode,
        fault: PointerUpdateFault,
        existing_metadata: bool,
    ) -> (Result<Vec<RecordBatch>>, PointerFixture) {
        let fixture = pointer_fixture(mode, fault, existing_metadata).await;
        let result = fixture.commit(POINTER_DATA_FILE).await;
        (result, fixture)
    }

    /// Append to a catalog table that already has a snapshot, so that the commit takes the
    /// normal path and its metadata pointer update hits `fault`.
    async fn run_append_with_pointer_fault(
        mode: PointerMode,
        fault: PointerUpdateFault,
    ) -> (Result<Vec<RecordBatch>>, PointerFixture) {
        let mut fixture = pointer_fixture(mode, fault, true).await;
        fixture.catalog.armed.store(false, Ordering::SeqCst);
        fixture
            .commit(POINTER_DATA_FILE)
            .await
            .expect("first snapshot");
        fixture.catalog.armed.store(true, Ordering::SeqCst);
        fixture.initial_metadata_location = fixture.catalog_pointer().await;
        fixture.preexisting_objects = fixture.objects().await;
        let result = fixture.commit(APPENDED_DATA_FILE).await;
        (result, fixture)
    }

    fn committed_count(batches: &[RecordBatch]) -> Option<i64> {
        batches
            .first()?
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|counts| counts.value(0))
    }

    fn is_commit_state_unknown(error: &DataFusionError) -> bool {
        matches!(
            error.find_root(),
            DataFusionError::External(source)
                if source
                    .downcast_ref::<CatalogError>()
                    .is_some_and(|error| matches!(error, CatalogError::CommitStateUnknown(_)))
        )
    }

    #[test]
    fn applied_bootstrap_pointer_update_that_reports_an_error_commits() {
        futures::executor::block_on(async {
            for existing_metadata in [false, true] {
                for fault in [
                    PointerUpdateFault::AppliedResponseLost,
                    PointerUpdateFault::AppliedThenAdvanced,
                ] {
                    let (result, fixture) = run_bootstrap_with_pointer_fault(
                        PointerMode::CatalogPointer,
                        fault,
                        existing_metadata,
                    )
                    .await;
                    let case = format!("{fault:?}, existing metadata {existing_metadata}");
                    assert!(
                        result.is_ok(),
                        "{case}: an applied update must commit: {:?}",
                        result.as_ref().err()
                    );
                    let batches = result.expect("committed bootstrap");
                    assert_eq!(committed_count(&batches), Some(3), "{case}");
                    assert_eq!(
                        fixture.committed_data_files().await,
                        vec![POINTER_DATA_FILE.to_string()],
                        "{case}"
                    );
                    let pointer = fixture.catalog_pointer().await;
                    if fault == PointerUpdateFault::AppliedThenAdvanced {
                        assert_eq!(
                            pointer.as_deref(),
                            Some(ADVANCED_METADATA_LOCATION),
                            "{case}"
                        );
                    } else {
                        let (metadata_files, _) = fixture.written_artifacts().await;
                        assert_eq!(metadata_files.len(), 1, "{case}: {metadata_files:?}");
                        assert!(
                            pointer
                                .as_deref()
                                .is_some_and(|pointer| pointer.ends_with(&metadata_files[0])),
                            "{case}: {pointer:?}"
                        );
                    }
                }
            }
        });
    }

    /// Run every commit path whose pointer update hits `fault`: a bootstrap without and with
    /// existing metadata, then an append on the normal path.
    async fn run_pointer_update_cases(
        mode: PointerMode,
        fault: PointerUpdateFault,
    ) -> Vec<(String, Result<Vec<RecordBatch>>, PointerFixture)> {
        let mut cases = Vec::new();
        for existing_metadata in [false, true] {
            let (result, fixture) =
                run_bootstrap_with_pointer_fault(mode, fault, existing_metadata).await;
            cases.push((
                format!("bootstrap, existing metadata {existing_metadata}"),
                result,
                fixture,
            ));
        }
        let (result, fixture) = run_append_with_pointer_fault(mode, fault).await;
        cases.push(("append".to_string(), result, fixture));
        cases
    }

    #[test]
    fn timed_out_pointer_update_that_the_catalog_does_not_reflect_reports_unknown_commit_state() {
        futures::executor::block_on(async {
            for (case, result, fixture) in run_pointer_update_cases(
                PointerMode::CatalogPointer,
                PointerUpdateFault::NotApplied,
            )
            .await
            {
                let error = result.expect_err("an unconfirmed pointer update must fail the write");
                // The request may still be in flight, so a caller retry could commit twice.
                assert!(is_commit_state_unknown(&error), "{case}: {error}");
                assert_eq!(
                    fixture.catalog_pointer().await,
                    fixture.initial_metadata_location,
                    "{case}"
                );
                let (metadata_files, manifest_lists) = fixture.written_artifacts().await;
                assert_eq!(metadata_files.len(), 1, "{case}: {metadata_files:?}");
                assert_eq!(manifest_lists.len(), 1, "{case}: {manifest_lists:?}");
                assert!(
                    error.to_string().contains(&metadata_files[0]),
                    "{case}: {error}"
                );
            }
        });
    }

    #[test]
    fn refused_pointer_update_reports_the_error_and_keeps_the_files() {
        futures::executor::block_on(async {
            for (case, result, fixture) in
                run_pointer_update_cases(PointerMode::CatalogPointer, PointerUpdateFault::Refused)
                    .await
            {
                let error = result.expect_err("a refused pointer update must fail the write");
                assert!(!is_commit_state_unknown(&error), "{case}: {error}");
                assert!(
                    error.to_string().contains("not allowed to alter the table"),
                    "{case}: {error}"
                );
                assert_eq!(
                    fixture.catalog_pointer().await,
                    fixture.initial_metadata_location,
                    "{case}"
                );
                let (metadata_files, manifest_lists) = fixture.written_artifacts().await;
                assert_eq!(metadata_files.len(), 1, "{case}: {metadata_files:?}");
                assert_eq!(manifest_lists.len(), 1, "{case}: {manifest_lists:?}");
            }
        });
    }

    #[test]
    fn applied_pointer_update_after_a_commit_that_reports_an_error_commits() {
        futures::executor::block_on(async {
            for fault in [
                PointerUpdateFault::AppliedThenTimedOut,
                // A retried request conflicts with the update it already applied.
                PointerUpdateFault::AppliedResponseLost,
                PointerUpdateFault::AppliedThenAdvanced,
                // Only the metadata log of the advanced metadata still names the update.
                PointerUpdateFault::AppliedThenAdvancedAndExpired,
            ] {
                let (result, fixture) =
                    run_append_with_pointer_fault(PointerMode::CatalogPointer, fault).await;
                assert!(
                    result.is_ok(),
                    "{fault:?}: an applied update must commit: {:?}",
                    result.as_ref().err()
                );
                let batches = result.expect("committed append");
                assert_eq!(committed_count(&batches), Some(3), "{fault:?}");
                assert_eq!(
                    fixture.committed_data_files().await,
                    vec![
                        APPENDED_DATA_FILE.to_string(),
                        POINTER_DATA_FILE.to_string()
                    ],
                    "{fault:?}"
                );
            }
        });
    }

    #[test]
    fn bootstrap_pointer_update_that_cannot_be_reconciled_reports_unknown_commit_state() {
        futures::executor::block_on(async {
            for existing_metadata in [false, true] {
                let (result, fixture) = run_bootstrap_with_pointer_fault(
                    PointerMode::CatalogPointer,
                    PointerUpdateFault::AppliedThenReloadFails,
                    existing_metadata,
                )
                .await;
                let error = result.expect_err("an unconfirmed pointer update must fail the write");
                assert!(is_commit_state_unknown(&error), "{error}");
                let (metadata_files, _) = fixture.written_artifacts().await;
                assert_eq!(metadata_files.len(), 1, "{metadata_files:?}");
                assert!(error.to_string().contains(&metadata_files[0]), "{error}");
                fixture.catalog.reads_fail.store(false, Ordering::SeqCst);
                assert_eq!(
                    fixture.committed_data_files().await,
                    vec![POINTER_DATA_FILE.to_string()]
                );
            }
        });
    }

    #[test]
    fn retried_bootstrap_pointer_conflict_keeps_the_files_of_the_lost_attempt() {
        futures::executor::block_on(async {
            let (result, fixture) = run_bootstrap_with_pointer_fault(
                PointerMode::CatalogPointer,
                PointerUpdateFault::ConflictOnce,
                true,
            )
            .await;
            let batches = result.expect("the retry must commit");
            assert_eq!(committed_count(&batches), Some(3));
            assert_eq!(fixture.catalog.updates.load(Ordering::SeqCst), 2);
            assert_eq!(
                fixture.committed_data_files().await,
                vec![POINTER_DATA_FILE.to_string()]
            );
            let (metadata_files, manifest_lists) = fixture.written_artifacts().await;
            assert_eq!(metadata_files.len(), 2, "{metadata_files:?}");
            assert_eq!(manifest_lists.len(), 2, "{manifest_lists:?}");
        });
    }

    #[test]
    fn new_table_pointer_conflict_without_a_previous_location_reports_unknown_commit_state() {
        futures::executor::block_on(async {
            // A first pointer update compares against nothing, so a conflict cannot prove
            // that this update did not apply.
            let (result, fixture) = run_bootstrap_with_pointer_fault(
                PointerMode::CatalogPointer,
                PointerUpdateFault::ConflictOnce,
                false,
            )
            .await;
            let error = result.expect_err("an unconfirmed pointer update must fail the write");
            assert!(is_commit_state_unknown(&error), "{error}");
            assert_eq!(fixture.catalog.updates.load(Ordering::SeqCst), 1);
            let (metadata_files, _) = fixture.written_artifacts().await;
            assert_eq!(metadata_files.len(), 1, "{metadata_files:?}");
        });
    }

    #[test]
    fn bootstrap_pointer_conflict_whose_table_goes_missing_reports_unknown_commit_state() {
        futures::executor::block_on(async {
            let (result, fixture) = run_bootstrap_with_pointer_fault(
                PointerMode::CatalogPointer,
                PointerUpdateFault::ConflictThenTableMissing,
                true,
            )
            .await;
            let error = result.expect_err("an unconfirmed pointer update must fail the write");
            assert!(is_commit_state_unknown(&error), "{error}");
            assert_eq!(fixture.catalog.updates.load(Ordering::SeqCst), 1);
            let (metadata_files, _) = fixture.written_artifacts().await;
            assert_eq!(metadata_files.len(), 1, "{metadata_files:?}");
            assert!(error.to_string().contains(&metadata_files[0]), "{error}");
        });
    }

    #[test]
    fn commit_retry_without_a_catalog_pointer_does_not_build_on_the_metadata_listing() {
        futures::executor::block_on(async {
            let (result, fixture) = run_bootstrap_with_pointer_fault(
                PointerMode::CatalogPointer,
                PointerUpdateFault::ConflictThenTableMissingOnRetry,
                true,
            )
            .await;
            // The listing names the uncommitted metadata of the conflicted attempt.
            let error = result.expect_err("a retry without a catalog pointer must fail the write");
            assert!(
                error
                    .to_string()
                    .contains("no longer reports a metadata location"),
                "{error}"
            );
            assert_eq!(fixture.catalog.updates.load(Ordering::SeqCst), 1);
            assert_eq!(
                fixture.catalog_pointer().await,
                fixture.initial_metadata_location
            );
        });
    }

    #[test]
    fn filesystem_registration_that_fails_after_a_durable_write_commits() {
        futures::executor::block_on(async {
            // A caller that retried a write reported as failed would append its rows twice.
            for fault in [
                PointerUpdateFault::Refused,
                PointerUpdateFault::NotApplied,
                PointerUpdateFault::ConflictOnce,
            ] {
                for (path, result, fixture) in
                    run_pointer_update_cases(PointerMode::FilesystemRegistered, fault).await
                {
                    let case = format!("{fault:?}, {path}");
                    // Readers list the metadata directory, so the write landed with its file.
                    assert!(
                        result.is_ok(),
                        "{case}: a durable write must commit: {:?}",
                        result.as_ref().err()
                    );
                    let batches = result.expect("committed write");
                    assert_eq!(committed_count(&batches), Some(3), "{case}");
                    let mut expected = vec![POINTER_DATA_FILE.to_string()];
                    if path == "append" {
                        expected.insert(0, APPENDED_DATA_FILE.to_string());
                    }
                    assert_eq!(fixture.committed_data_files().await, expected, "{case}");
                    assert_eq!(
                        fixture.catalog_pointer().await,
                        fixture.initial_metadata_location,
                        "{case}"
                    );
                }
            }
        });
    }

    #[test]
    fn filesystem_commit_built_on_the_catalog_pointer_keeps_the_reconciled_pointer_update() {
        futures::executor::block_on(async {
            // The commit did not build on the listing, so its metadata file is not known to be
            // the head, and only the pointer update decides whether it committed.
            let mode = PointerMode::FilesystemOnCatalogPointer;
            for fault in [
                PointerUpdateFault::Refused,
                PointerUpdateFault::AppliedResponseLost,
            ] {
                let cases = vec![
                    (
                        "bootstrap",
                        run_bootstrap_with_pointer_fault(mode, fault, true).await,
                    ),
                    ("append", run_append_with_pointer_fault(mode, fault).await),
                ];
                for (path, (result, fixture)) in cases {
                    let case = format!("{fault:?}, {path}");
                    let pointer = fixture.catalog_pointer().await;
                    if fault == PointerUpdateFault::Refused {
                        let error =
                            result.expect_err("a refused pointer update must fail the write");
                        assert!(!is_commit_state_unknown(&error), "{case}: {error}");
                        assert!(
                            error.to_string().contains("not allowed to alter the table"),
                            "{case}: {error}"
                        );
                        assert_eq!(pointer, fixture.initial_metadata_location, "{case}");
                    } else {
                        let batches = result.expect("an applied pointer update must commit");
                        assert_eq!(committed_count(&batches), Some(3), "{case}");
                        let (metadata_files, _) = fixture.written_artifacts().await;
                        assert_eq!(metadata_files.len(), 1, "{case}: {metadata_files:?}");
                        assert!(
                            pointer
                                .as_deref()
                                .is_some_and(|pointer| pointer.ends_with(&metadata_files[0])),
                            "{case}: {pointer:?}"
                        );
                    }
                }
            }
        });
    }

    fn commit_plan(
        table_url: &Url,
        actions: RecordBatch,
        snapshot_update_kind: SnapshotUpdateKind,
    ) -> IcebergCommitExec {
        let input = MemorySourceConfig::try_new_exec(
            &[vec![actions]],
            iceberg_action_schema().expect("action schema"),
            None,
        )
        .expect("writer output");
        IcebergCommitExec::new(input, table_url.clone(), None, snapshot_update_kind)
    }

    #[test]
    fn controlled_empty_dynamic_partition_overwrite_fails_closed() {
        futures::executor::block_on(async {
            let table_url =
                Url::parse("file:///tmp/controlled-empty-overwrite/").expect("table URL");
            let memory = Arc::new(object_store::memory::InMemory::new());
            let store: Arc<dyn ObjectStore> = memory.clone();
            let store_ctx = StoreContext::new(Arc::clone(&store), &table_url).expect("store");
            let table_properties = vec![("format-version".to_string(), "2".to_string())];
            crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                IcebergSchema::builder()
                    .with_fields([Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    ))])
                    .build()
                    .expect("schema"),
                PartitionSpec::unpartitioned_spec(),
                &table_properties,
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("bootstrap metadata");
            let context = SessionContext::new();
            context
                .runtime_env()
                .register_object_store(&Url::parse("file:///").expect("file store URL"), memory);
            let mut data_file = partitioned_data_file("data/live.parquet", 0, 0);
            data_file.partition.clear();
            let append = datafusion::arrow::compute::concat_batches(
                &iceberg_action_schema().expect("action schema"),
                &[
                    encode_add_data_files(vec![data_file]).expect("add action"),
                    encode_commit_meta(CommitMeta {
                        table_uri: table_url.to_string(),
                        row_count: 1,
                        table_properties: table_properties.clone(),
                        ..Default::default()
                    })
                    .expect("commit metadata"),
                ],
            )
            .expect("append actions");
            datafusion::physical_plan::collect(
                Arc::new(commit_plan(
                    &table_url,
                    append,
                    SnapshotUpdateKind::FastAppend,
                )),
                context.task_ctx(),
            )
            .await
            .expect("append");
            let head = crate::table::find_latest_metadata_file(&store, &table_url)
                .await
                .expect("metadata location");
            let metadata = TableMetadata::from_json(
                &load_metadata_file_bytes(&store, &head)
                    .await
                    .expect("metadata bytes"),
            )
            .expect("metadata");
            let snapshot_id = metadata.current_snapshot_id.expect("appended snapshot");
            let empty_overwrite = encode_commit_meta(CommitMeta {
                table_uri: table_url.to_string(),
                table_properties,
                ..Default::default()
            })
            .expect("commit metadata");

            for (snapshot_properties, caller_expected_snapshot_id) in [
                (
                    vec![(
                        "hoist.publication-id".to_string(),
                        "publication-1".to_string(),
                    )],
                    None,
                ),
                (vec![], Some(snapshot_id)),
                (vec![], None),
            ] {
                let controlled =
                    !snapshot_properties.is_empty() || caller_expected_snapshot_id.is_some();
                let plan: Arc<dyn ExecutionPlan> = Arc::new(
                    commit_plan(
                        &table_url,
                        empty_overwrite.clone(),
                        SnapshotUpdateKind::CopyOnWrite,
                    )
                    .with_expected_snapshot_id(Some(Some(snapshot_id)))
                    .with_dynamic_partition_overwrite(true)
                    .with_caller_expected_snapshot_id(caller_expected_snapshot_id)
                    .with_snapshot_properties(snapshot_properties),
                );
                let result = datafusion::physical_plan::collect(plan, context.task_ctx()).await;
                if controlled {
                    let error =
                        result.expect_err("a controlled write that changes nothing must fail");
                    assert!(
                        error
                            .to_string()
                            .contains("publication controls changes no data"),
                        "{error}"
                    );
                } else {
                    let batches = result.expect("an uncontrolled empty overwrite is a no-op");
                    assert_eq!(committed_count(&batches), Some(0));
                }
                assert_eq!(
                    crate::table::find_latest_metadata_file(&store, &table_url)
                        .await
                        .expect("metadata location"),
                    head,
                    "an empty overwrite must not publish a snapshot"
                );
            }
        });
    }

    /// Where a path-table commit writes the metadata file whose create-only write hits a
    /// fault.
    #[derive(Debug, Clone, Copy)]
    enum MetadataWriteSite {
        /// A commit without table metadata bootstraps a new table in `v1`.
        NewTableBootstrap,
        /// A commit on empty table metadata `v1` bootstraps the first snapshot in `v2`.
        FirstSnapshotBootstrap,
        /// A commit on a table at snapshot metadata `v2` publishes `v3` on the normal path.
        NextVersion,
    }

    impl MetadataWriteSite {
        const ALL: [Self; 3] = [
            Self::NewTableBootstrap,
            Self::FirstSnapshotBootstrap,
            Self::NextVersion,
        ];

        fn metadata_file(self) -> &'static str {
            match self {
                Self::NewTableBootstrap => "metadata/v1.metadata.json",
                Self::FirstSnapshotBootstrap => "metadata/v2.metadata.json",
                Self::NextVersion => "metadata/v3.metadata.json",
            }
        }
    }

    struct MetadataWriteFaultRun {
        result: Result<Vec<RecordBatch>>,
        store: Arc<FaultInjectingMetadataStore>,
        memory: Arc<object_store::memory::InMemory>,
        store_ctx: StoreContext,
        table_url: Url,
    }

    impl MetadataWriteFaultRun {
        /// The latest metadata file and the live data files with their row counts, after
        /// loading every manifest list the metadata names and every manifest and data file
        /// of its current snapshot.
        async fn committed_table(&self) -> (String, Vec<(String, u64)>) {
            let store: Arc<dyn ObjectStore> = self.memory.clone();
            let location = crate::table::find_latest_metadata_file(&store, &self.table_url)
                .await
                .expect("latest metadata location");
            let bytes = load_metadata_file_bytes(&store, &location)
                .await
                .expect("latest metadata bytes");
            let metadata = TableMetadata::from_json(&bytes).expect("latest metadata");
            for snapshot in &metadata.snapshots {
                load_manifest_list(&self.store_ctx, snapshot.manifest_list())
                    .await
                    .expect("every snapshot names an existing manifest list");
            }
            let snapshot = metadata.current_snapshot().expect("current snapshot");
            let manifests = load_manifest_list(&self.store_ctx, snapshot.manifest_list())
                .await
                .expect("current manifest list");
            let mut files = Vec::new();
            for manifest_file in manifests.entries() {
                let manifest = load_manifest(&self.store_ctx, &manifest_file.manifest_path)
                    .await
                    .expect("current manifest");
                for entry in manifest.entries().iter().filter(|entry| {
                    matches!(
                        entry.status,
                        ManifestStatus::Added | ManifestStatus::Existing
                    )
                }) {
                    self.store_ctx
                        .prefixed
                        .head(&Path::from(entry.data_file.file_path.as_str()))
                        .await
                        .expect("live data file");
                    files.push((
                        entry.data_file.file_path.clone(),
                        entry.data_file.record_count,
                    ));
                }
            }
            files.sort();
            (location, files)
        }
    }

    fn append_actions(table_url: &Url, path: &str, rows: u64) -> RecordBatch {
        let mut data_file = partitioned_data_file(path, 0, 0);
        data_file.partition.clear();
        data_file.record_count = rows;
        datafusion::arrow::compute::concat_batches(
            &iceberg_action_schema().expect("action schema"),
            &[
                encode_add_data_files(vec![data_file]).expect("add action"),
                encode_commit_meta(CommitMeta {
                    table_uri: table_url.to_string(),
                    row_count: rows,
                    table_properties: vec![("format-version".to_string(), "2".to_string())],
                    schema: Some(
                        IcebergSchema::builder()
                            .with_fields([Arc::new(NestedField::required(
                                1,
                                "id",
                                Type::Primitive(PrimitiveType::Int),
                            ))])
                            .build()
                            .expect("schema"),
                    ),
                    ..Default::default()
                })
                .expect("commit metadata"),
            ],
        )
        .expect("append actions")
    }

    /// Append `data/second.parquet` with three rows to a path table whose create-only write
    /// of the metadata file at `site` hits `fault`. Before it, the table holds what the site
    /// needs: nothing, empty metadata, or a snapshot with `data/first.parquet` and two rows.
    async fn run_commit_with_metadata_write_fault(
        site: MetadataWriteSite,
        fault: MetadataWriteFault,
    ) -> MetadataWriteFaultRun {
        let table_url = Url::parse("file:///tmp/metadata-write-fault/").expect("table URL");
        let memory = Arc::new(object_store::memory::InMemory::new());
        let base_store: Arc<dyn ObjectStore> = memory.clone();
        let store_ctx = StoreContext::new(base_store, &table_url).expect("store context");
        let store = Arc::new(FaultInjectingMetadataStore::new(
            Arc::clone(&memory),
            fault,
            site.metadata_file(),
        ));
        let context = SessionContext::new();
        context.runtime_env().register_object_store(
            &Url::parse("file:///").expect("file store URL"),
            store.clone(),
        );
        if !matches!(site, MetadataWriteSite::NewTableBootstrap) {
            crate::operations::bootstrap::bootstrap_empty_table_metadata(
                &table_url,
                &store_ctx,
                IcebergSchema::builder()
                    .with_fields([Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    ))])
                    .build()
                    .expect("schema"),
                PartitionSpec::unpartitioned_spec(),
                &[("format-version".to_string(), "2".to_string())],
                NewTableMetadataStyle::Hadoop,
            )
            .await
            .expect("empty table metadata");
        }
        if matches!(site, MetadataWriteSite::NextVersion) {
            store_ctx
                .prefixed
                .put(
                    &Path::from("data/first.parquet"),
                    Bytes::from_static(b"first").into(),
                )
                .await
                .expect("first data file");
            datafusion::physical_plan::collect(
                Arc::new(commit_plan(
                    &table_url,
                    append_actions(&table_url, "data/first.parquet", 2),
                    SnapshotUpdateKind::FastAppend,
                )),
                context.task_ctx(),
            )
            .await
            .expect("first append");
        }
        store_ctx
            .prefixed
            .put(
                &Path::from("data/second.parquet"),
                Bytes::from_static(b"second").into(),
            )
            .await
            .expect("second data file");
        let result = datafusion::physical_plan::collect(
            Arc::new(commit_plan(
                &table_url,
                append_actions(&table_url, "data/second.parquet", 3),
                SnapshotUpdateKind::FastAppend,
            )),
            context.task_ctx(),
        )
        .await;
        assert!(
            store.fault_injected.load(Ordering::SeqCst),
            "{site:?}: the commit never wrote {}",
            site.metadata_file()
        );
        MetadataWriteFaultRun {
            result,
            store,
            memory,
            store_ctx,
            table_url,
        }
    }

    /// The ways `run` differs from a commit that keeps every file of its landed write and
    /// publishes `data/second.parquet` at `site`.
    async fn landed_write_failures(
        site: MetadataWriteSite,
        run: &MetadataWriteFaultRun,
    ) -> Vec<String> {
        let batches = match &run.result {
            Ok(batches) => batches,
            Err(error) => {
                return vec![format!(
                    "{site:?}: a write that landed must commit: {error}"
                )];
            }
        };
        let mut failures = Vec::new();
        if committed_count(batches) != Some(3) {
            failures.push(format!("{site:?}: count {:?}", committed_count(batches)));
        }
        let deleted = run.store.deleted();
        if !deleted.is_empty() {
            failures.push(format!("{site:?}: deleted {deleted:?}"));
        }
        let (location, files) = run.committed_table().await;
        if !location.ends_with(site.metadata_file()) {
            failures.push(format!("{site:?}: latest metadata {location}"));
        }
        let mut expected = vec![("data/second.parquet".to_string(), 3)];
        if matches!(site, MetadataWriteSite::NextVersion) {
            expected.insert(0, ("data/first.parquet".to_string(), 2));
        }
        if files != expected {
            failures.push(format!("{site:?}: live files {files:?}"));
        }
        failures
    }

    #[test]
    fn metadata_write_that_landed_before_it_reported_a_conflict_commits() {
        futures::executor::block_on(async {
            let mut failures = Vec::new();
            for site in MetadataWriteSite::ALL {
                let run = run_commit_with_metadata_write_fault(
                    site,
                    MetadataWriteFault::LandedThenExists,
                )
                .await;
                failures.extend(landed_write_failures(site, &run).await);
            }
            assert!(failures.is_empty(), "{failures:#?}");
        });
    }

    #[test]
    fn metadata_write_conflict_that_cannot_be_read_back_reports_unknown_state_and_deletes_nothing()
    {
        futures::executor::block_on(async {
            let mut failures = Vec::new();
            for site in MetadataWriteSite::ALL {
                for fault in [
                    MetadataWriteFault::LandedThenExistsUnreadable,
                    MetadataWriteFault::ExistsThenMissing,
                ] {
                    let case = format!("{site:?}, {fault:?}");
                    let run = run_commit_with_metadata_write_fault(site, fault).await;
                    match &run.result {
                        Ok(_) => failures.push(format!("{case}: an unconfirmed write committed")),
                        Err(error) if !is_commit_state_unknown(error) => {
                            failures.push(format!("{case}: not an unknown state: {error}"));
                        }
                        Err(error) if !error.to_string().contains(site.metadata_file()) => {
                            failures.push(format!("{case}: the error names no path: {error}"));
                        }
                        Err(_) => {}
                    }
                    let deleted = run.store.deleted();
                    if !deleted.is_empty() {
                        failures.push(format!("{case}: deleted {deleted:?}"));
                    }
                }
            }
            assert!(failures.is_empty(), "{failures:#?}");
        });
    }
}
