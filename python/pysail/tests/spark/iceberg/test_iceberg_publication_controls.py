from pathlib import Path

import pytest
from pyiceberg.table import StaticTable
from pyspark.sql import functions as F  # noqa: N812

from pysail.testing.spark.utils.sql import escape_sql_string_literal
from pysail.tests.spark.iceberg.utils import pyiceberg_file_io_properties


def _create_partitioned_table(spark, table_name: str, location: Path) -> None:
    escaped_location = escape_sql_string_literal(str(location))
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(
        f"""
        CREATE TABLE {table_name} (id BIGINT, category STRING, value BIGINT)
        USING iceberg
        PARTITIONED BY (category)
        LOCATION '{escaped_location}'
        """
    )


def test_iceberg_predicate_overwrite_persists_snapshot_properties(spark, tmp_path):
    table_name = "iceberg_overwrite_snapshot_properties"
    location = tmp_path / table_name
    _create_partitioned_table(spark, table_name, location)
    try:
        schema = "id BIGINT, category STRING, value BIGINT"
        spark.createDataFrame([(1, "A", 10), (2, "B", 20)], schema=schema).writeTo(
            table_name
        ).append()

        replacement = spark.createDataFrame([(3, "A", 30)], schema=schema)
        (
            replacement.writeTo(table_name)
            .option("snapshot-property.hoist.publication-id", "publication-123")
            .option("snapshot-property.hoist.plan-digest", "sha256:abc123")
            .overwrite(F.col("category") == "A")
        )

        table = StaticTable.from_metadata(
            str(location),
            properties=pyiceberg_file_io_properties(),
        )
        snapshot = table.current_snapshot()
        assert snapshot is not None
        assert snapshot.summary is not None
        properties = snapshot.summary.additional_properties
        expected_properties = {
            "hoist.plan-digest": "sha256:abc123",
            "hoist.publication-id": "publication-123",
        }
        assert expected_properties.items() <= properties.items()
        assert int(properties["added-data-files"]) > 0
        assert int(properties["total-records"]) > 0
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def test_iceberg_predicate_overwrite_rejects_stale_expected_snapshot_id(spark, tmp_path):
    table_name = "iceberg_overwrite_expected_snapshot"
    location = tmp_path / table_name
    _create_partitioned_table(spark, table_name, location)
    try:
        schema = "id BIGINT, category STRING, value BIGINT"
        spark.createDataFrame([(1, "A", 10), (2, "B", 20)], schema=schema).writeTo(
            table_name
        ).append()
        table = StaticTable.from_metadata(
            str(location),
            properties=pyiceberg_file_io_properties(),
        )
        stale_snapshot = table.current_snapshot()
        assert stale_snapshot is not None

        spark.createDataFrame([(3, "C", 30)], schema=schema).writeTo(table_name).append()
        table = StaticTable.from_metadata(
            str(location),
            properties=pyiceberg_file_io_properties(),
        )
        current_snapshot = table.current_snapshot()
        assert current_snapshot is not None
        assert current_snapshot.snapshot_id != stale_snapshot.snapshot_id

        rows_before = [
            tuple(row)
            for row in spark.table(table_name)
            .select("id", "category", "value")
            .orderBy("id")
            .collect()
        ]
        metadata_files_before = set((location / "metadata").glob("*.metadata.json"))
        live_files_before = {str(task.file.file_path) for task in table.scan().plan_files()}

        replacement = spark.createDataFrame([(4, "A", 40)], schema=schema)
        expected_error = (
            rf"expected snapshot {stale_snapshot.snapshot_id}.*"
            rf"current snapshot {current_snapshot.snapshot_id}"
        )
        with pytest.raises(Exception, match=expected_error):
            (
                replacement.writeTo(table_name)
                .option("expected-snapshot-id", str(stale_snapshot.snapshot_id))
                .overwrite(F.col("category") == "A")
            )

        table = StaticTable.from_metadata(
            str(location),
            properties=pyiceberg_file_io_properties(),
        )
        snapshot_after = table.current_snapshot()
        assert snapshot_after is not None
        assert snapshot_after.snapshot_id == current_snapshot.snapshot_id
        assert [
            tuple(row)
            for row in spark.table(table_name)
            .select("id", "category", "value")
            .orderBy("id")
            .collect()
        ] == rows_before
        assert set((location / "metadata").glob("*.metadata.json")) == metadata_files_before
        assert {str(task.file.file_path) for task in table.scan().plan_files()} == live_files_before
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
