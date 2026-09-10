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

use std::sync::Arc;

use datafusion::arrow::compute::SortOptions;
use datafusion::catalog::Session;
use datafusion::common::Result;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::{ExecutionPlan, Partitioning};
use sail_common_datafusion::catalog::CatalogPartitionField;
use sail_common_datafusion::datasource::PhysicalSinkMode;
use url::Url;

use crate::operations::SnapshotUpdateKind;
use crate::physical_plan::write_context::IcebergWriteContext;
use crate::physical_plan::writer_exec::IcebergWriterExec;
use crate::physical_plan::writer_options::IcebergWriterExecOptions;
use crate::utils::partition_transform::format_partition_expr;

pub struct IcebergTableConfig {
    pub table_url: Url,
    pub partition_columns: Vec<CatalogPartitionField>,
    pub table_exists: bool,
    pub options: IcebergWriterExecOptions,
    pub write_context: IcebergWriteContext,
}

pub struct IcebergPlanBuilder<'a> {
    input: Arc<dyn ExecutionPlan>,
    table_config: IcebergTableConfig,
    sink_mode: PhysicalSinkMode,
    sort_order: Option<Vec<PhysicalSortExpr>>,
    expected_snapshot_id: Option<Option<i64>>,
    removed_data_file_paths: Vec<String>,
    dynamic_partition_overwrite: bool,
    session: &'a dyn Session,
}

impl<'a> IcebergPlanBuilder<'a> {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        table_config: IcebergTableConfig,
        sink_mode: PhysicalSinkMode,
        sort_order: Option<Vec<PhysicalSortExpr>>,
        session: &'a dyn Session,
    ) -> Self {
        Self {
            input,
            table_config,
            sink_mode,
            sort_order,
            expected_snapshot_id: None,
            removed_data_file_paths: Vec::new(),
            dynamic_partition_overwrite: false,
            session,
        }
    }

    pub fn with_expected_snapshot_id(mut self, expected_snapshot_id: Option<Option<i64>>) -> Self {
        self.expected_snapshot_id = expected_snapshot_id;
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

    pub async fn build(self) -> Result<Arc<dyn ExecutionPlan>> {
        self.add_projection_node(self.input.clone())
            .and_then(|plan| self.add_repartition_node(plan))
            .and_then(|plan| self.add_sort_node(plan))
            .and_then(|plan| self.add_writer_node(plan))
            .and_then(|plan| self.add_commit_node(plan))
    }

    fn add_projection_node(&self, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        // Validate that partition transform expressions refer to real source columns.
        // Do not reorder columns here: BDD "query result ordered" checks expect the original
        // table column order from `SELECT *`.
        let schema = input.schema();
        for field in &self.table_config.partition_columns {
            if schema.index_of(&field.column).is_err() {
                return Err(datafusion::common::DataFusionError::Plan(format!(
                    "Partition column '{}' not found in schema",
                    format_partition_expr(field)
                )));
            }
        }
        Ok(input)
    }

    fn add_repartition_node(
        &self,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Writer parallelism follows the session target rather than a literal, and no
        // key-based distribution is requested. `IcebergWriterExec` reports
        // `UnspecifiedDistribution`, so a hash requirement here was not honored end to end:
        // every writer task still received rows for every table partition. Grouping is
        // established inside each task by `add_sort_node` instead, which is what bounds how
        // many partition writers a task holds open, without capping how many tasks may write
        // into one table partition.
        let target_partitions = self.session.config().target_partitions().max(1);
        if input.properties().output_partitioning().partition_count() >= target_partitions {
            return Ok(input);
        }
        Ok(Arc::new(RepartitionExec::try_new(
            input,
            Partitioning::RoundRobinBatch(target_partitions),
        )?))
    }

    /// Sort task-locally by the partition source columns, then by any user sort order.
    ///
    /// This keeps rows for one partition contiguous within a task so `IcebergTableWriter`
    /// can finish a partition writer when the key changes. It is a grouping aid and not a
    /// guarantee: an order-preserving transform (identity, `truncate`, the date parts)
    /// groups by the computed partition value, while `bucket` does not, so the writer keeps
    /// its own bound on concurrently open writers.
    fn add_sort_node(&self, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let mut seen = std::collections::HashSet::new();
        let mut sort_exprs: Vec<PhysicalSortExpr> = Vec::new();
        for field in &self.table_config.partition_columns {
            if !seen.insert(field.column.clone()) {
                continue;
            }
            let idx = schema.index_of(&field.column).map_err(|_| {
                datafusion::common::DataFusionError::Plan(format!(
                    "Partition column '{}' not found in schema",
                    field.column
                ))
            })?;
            sort_exprs.push(PhysicalSortExpr {
                expr: Arc::new(Column::new(&field.column, idx)) as Arc<dyn PhysicalExpr>,
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            });
        }
        sort_exprs.extend(self.sort_order.clone().unwrap_or_default());

        let Some(lex) = LexOrdering::new(sort_exprs) else {
            return Ok(input);
        };
        Ok(Arc::new(
            SortExec::new(lex, input).with_preserve_partitioning(true),
        ))
    }

    fn add_writer_node(&self, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(IcebergWriterExec::new(
            input,
            self.table_config.table_url.clone(),
            self.table_config.partition_columns.clone(),
            self.sink_mode.clone(),
            self.table_config.table_exists,
            self.table_config.options.clone(),
            self.table_config.write_context.clone(),
        )?))
    }

    fn add_commit_node(&self, input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let snapshot_update_kind = if self.table_config.table_exists {
            match &self.sink_mode {
                PhysicalSinkMode::Overwrite => SnapshotUpdateKind::FullOverwrite,
                PhysicalSinkMode::OverwriteIf { .. } | PhysicalSinkMode::OverwritePartitions => {
                    SnapshotUpdateKind::CopyOnWrite
                }
                _ => SnapshotUpdateKind::FastAppend,
            }
        } else {
            SnapshotUpdateKind::FastAppend
        };
        Ok(Arc::new(
            crate::physical_plan::commit::commit_exec::IcebergCommitExec::new(
                input,
                self.table_config.table_url.clone(),
                self.table_config.options.lakehouse_table.clone(),
                snapshot_update_kind,
            )
            .with_expected_snapshot_id(self.expected_snapshot_id)
            .with_removed_data_file_paths(self.removed_data_file_paths.clone())
            .with_dynamic_partition_overwrite(self.dynamic_partition_overwrite),
        ))
    }
}
