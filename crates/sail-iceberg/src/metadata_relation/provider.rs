use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::Schema as ArrowSchema;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_plan::ExecutionPlan;
use url::Url;

use super::{IcebergMetadataRelationType, entries, files};
use crate::options::r#gen::IcebergReadOptions;
use crate::physical_plan::metadata_relation_exec::{
    IcebergMetadataRelationExec, MetadataRelationScan, MetadataScanSource,
};
use crate::table::Table;
use crate::table::files::{
    EntrySelection, ManifestScope, SnapshotScope, balance_by_size, scan_manifests,
};

pub(crate) async fn metadata_relation_provider(
    session: &dyn Session,
    table_url: Url,
    metadata_location: Option<String>,
    relation_type: IcebergMetadataRelationType,
    options: &IcebergReadOptions,
) -> Result<Arc<dyn TableProvider>> {
    if !relation_type.is_supported() {
        return Err(DataFusionError::NotImplemented(
            relation_type.unsupported_reason(),
        ));
    }
    let table = Arc::new(
        Table::load_with_metadata_location(
            session.runtime_env().as_ref(),
            table_url,
            metadata_location,
        )
        .await?,
    );
    let temporal = options.snapshot_id.is_some()
        || options.use_ref.is_some()
        || options.timestamp_as_of.is_some();
    if temporal && relation_type != IcebergMetadataRelationType::Entries {
        return Err(DataFusionError::NotImplemented(format!(
            "Time travel is not supported for Iceberg metadata table '{}'",
            relation_type.name()
        )));
    }
    let snapshot = if relation_type == IcebergMetadataRelationType::Entries
        && (temporal || table.metadata().current_snapshot().is_some())
    {
        SnapshotScope::Id(table.scan_state(options)?.1.snapshot_id())
    } else {
        SnapshotScope::Current
    };
    let schema = match relation_type {
        IcebergMetadataRelationType::Entries => entries::schema(table.metadata())?,
        IcebergMetadataRelationType::Files => files::schema(table.metadata())?,
        _ => relation_type.schema(),
    };
    Ok(Arc::new(IcebergMetadataRelationProvider {
        table,
        relation_type,
        schema,
        snapshot,
    }))
}

struct IcebergMetadataRelationProvider {
    table: Arc<Table>,
    relation_type: IcebergMetadataRelationType,
    schema: Arc<ArrowSchema>,
    snapshot: SnapshotScope,
}

impl std::fmt::Debug for IcebergMetadataRelationProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergMetadataRelationProvider")
            .field("table_url", self.table.table_url())
            .field("metadata_location", &self.table.metadata_location())
            .field("relation_type", &self.relation_type)
            .finish()
    }
}

#[async_trait]
impl TableProvider for IcebergMetadataRelationProvider {
    fn schema(&self) -> Arc<ArrowSchema> {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        session: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let source = if matches!(
            self.relation_type,
            IcebergMetadataRelationType::Files | IcebergMetadataRelationType::Entries
        ) {
            MetadataScanSource::ManifestEntries {
                snapshot: self.snapshot,
                groups: balance_by_size(
                    scan_manifests(
                        self.table.store_context(),
                        self.table.metadata(),
                        &ManifestScope {
                            snapshots: self.snapshot,
                            content: None,
                        },
                    )
                    .await?,
                    session.config().target_partitions(),
                    |manifest| manifest.manifest_length.max(0) as u64,
                ),
                selection: if self.relation_type == IcebergMetadataRelationType::Entries {
                    EntrySelection::all()
                } else {
                    EntrySelection::live()
                },
            }
        } else {
            MetadataScanSource::TableMetadata
        };
        // Filters remain residual expressions above this scan.
        let limit = if filters.is_empty() { limit } else { None };
        Ok(Arc::new(IcebergMetadataRelationExec::try_new(
            self.schema.clone(),
            MetadataRelationScan {
                table_url: self.table.table_url().to_string(),
                metadata_location: self.table.metadata_location().to_string(),
                relation: self.relation_type,
                source,
                projection: projection
                    .cloned()
                    .unwrap_or_else(|| (0..self.schema.fields().len()).collect()),
                limit,
            },
        )?))
    }
}
