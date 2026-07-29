from pathlib import Path

from pyiceberg.table import StaticTable
from pyspark.sql import functions as F

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
        """  # noqa: S608
    )


def _rows(spark, table_name: str) -> list[tuple[int, str, int]]:
    rows = spark.table(table_name).select("id", "category", "value").orderBy("id")
    return [tuple(row) for row in rows.collect()]


def _live_data_file_paths(location: Path) -> set[str]:
    table = StaticTable.from_metadata(
        str(location),
        properties=pyiceberg_file_io_properties(),
    )
    return {str(task.file.file_path) for task in table.scan().plan_files()}


def _metadata_file_count(location: Path) -> int:
    return len(list((location / "metadata").glob("*.metadata.json")))


def test_iceberg_predicate_overwrite_rewrites_only_candidate_partition(spark, tmp_path):
    table_name = "iceberg_predicate_overwrite"
    location = tmp_path / table_name
    _create_partitioned_table(spark, table_name, location)
    try:
        spark.createDataFrame(
            [(1, "A", 10), (2, "B", 20), (3, "A", 30), (4, "B", 40)],
            schema="id BIGINT, category STRING, value BIGINT",
        ).writeTo(table_name).append()
        initial_live_files = _live_data_file_paths(location)

        spark.createDataFrame(
            [(5, "A", 100), (6, "A", 200)],
            schema="id BIGINT, category STRING, value BIGINT",
        ).writeTo(table_name).overwrite(F.col("category") == "A")

        assert _rows(spark, table_name) == [
            (2, "B", 20),
            (4, "B", 40),
            (5, "A", 100),
            (6, "A", 200),
        ]
        assert initial_live_files & _live_data_file_paths(location)
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def test_iceberg_dynamic_partition_overwrite_preserves_untouched_partitions(spark, tmp_path):
    table_name = "iceberg_dynamic_partition_overwrite"
    location = tmp_path / table_name
    _create_partitioned_table(spark, table_name, location)
    try:
        schema = "id BIGINT, category STRING, value BIGINT"
        spark.createDataFrame(
            [(1, "A", 10), (2, "B", 20), (3, "A", 30), (4, "B", 40)],
            schema=schema,
        ).writeTo(table_name).append()

        spark.createDataFrame(
            [(5, "A", 100), (6, "A", 200)],
            schema=schema,
        ).writeTo(table_name).overwritePartitions()
        assert _rows(spark, table_name) == [
            (2, "B", 20),
            (4, "B", 40),
            (5, "A", 100),
            (6, "A", 200),
        ]

        spark.createDataFrame(
            [(7, "C", 300)],
            schema=schema,
        ).writeTo(table_name).overwritePartitions()
        assert _rows(spark, table_name) == [
            (2, "B", 20),
            (4, "B", 40),
            (5, "A", 100),
            (6, "A", 200),
            (7, "C", 300),
        ]

        metadata_files_before = _metadata_file_count(location)
        spark.createDataFrame([], schema=schema).writeTo(table_name).overwritePartitions()
        assert _metadata_file_count(location) == metadata_files_before
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
