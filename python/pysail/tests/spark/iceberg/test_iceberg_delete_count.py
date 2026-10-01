import pytest

TABLE_MODES = {
    "copy-on-write": "'format-version' = '2'",
    "merge-on-read-v2": "'format-version' = '2', 'write.delete.mode' = 'merge-on-read'",
    "merge-on-read-v3": "'format-version' = '3', 'write.delete.mode' = 'merge-on-read'",
}


def _delete_count(spark, statement):
    rows = spark.sql(statement).selectExpr("CAST(count AS BIGINT) AS count").collect()
    assert len(rows) == 1
    assert rows[0]["count"] is not None
    return rows[0]["count"]


def _metadata_files(location):
    return sorted(path.name for path in (location / "metadata").glob("*.metadata.json"))


def _ids(spark, table_name):
    return [row["id"] for row in spark.table(table_name).orderBy("id").collect()]


def test_iceberg_delete_returns_affected_row_count(spark, tmp_path):
    table_name = "iceberg_delete_affected_count"
    location = (tmp_path / table_name).as_uri()
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"CREATE TABLE {table_name} (id BIGINT, category STRING) USING iceberg LOCATION '{location}'")
    try:
        spark.sql("INSERT INTO iceberg_delete_affected_count VALUES (1, 'delete'), (2, 'delete'), (3, 'keep')")

        result = (
            spark.sql("DELETE FROM iceberg_delete_affected_count WHERE category = 'delete'")
            .selectExpr("CAST(count AS BIGINT) AS count")
            .collect()
        )

        assert [row["count"] for row in result] == [2]
        remaining = spark.table(table_name).collect()
        expected_remaining_id = 3
        assert len(remaining) == 1
        assert remaining[0]["id"] == expected_remaining_id
        assert remaining[0]["category"] == "keep"
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


@pytest.mark.parametrize("mode", list(TABLE_MODES))
def test_iceberg_delete_counts_deleted_rows_on_every_write_path(spark, tmp_path, mode):
    table_name = "iceberg_delete_count_paths"
    location = tmp_path / table_name
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(
        f"""
        CREATE TABLE {table_name} (id INT, value STRING)
        USING iceberg
        LOCATION '{location.as_uri()}'
        TBLPROPERTIES ({TABLE_MODES[mode]})
        """
    )
    try:
        spark.sql(
            f"""
            INSERT INTO {table_name} SELECT /*+ COALESCE(1) */ * FROM VALUES
            (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')
            """
        )
        spark.sql(f"INSERT INTO {table_name} SELECT /*+ COALESCE(1) */ * FROM VALUES (10, 'x'), (11, 'y')")

        # A partial delete inside a data file rewrites it, or writes delete files.
        assert _delete_count(spark, f"DELETE FROM {table_name} WHERE id IN (2, 4)") == 2  # noqa: PLR2004
        assert _ids(spark, table_name) == [1, 3, 5, 10, 11]

        # A validated no-op delete reports zero and publishes no metadata.
        metadata_before = _metadata_files(location)
        assert _delete_count(spark, f"DELETE FROM {table_name} WHERE id = 999") == 0
        assert _metadata_files(location) == metadata_before

        # A predicate covering a whole data file removes it from metadata alone.
        assert _delete_count(spark, f"DELETE FROM {table_name} WHERE id >= 10") == 2  # noqa: PLR2004
        assert _ids(spark, table_name) == [1, 3, 5]

        # A subquery predicate takes the row-level path.
        assert (
            _delete_count(spark, f"DELETE FROM {table_name} WHERE id IN (SELECT id FROM {table_name} WHERE id = 3)")
            == 1
        )
        assert _ids(spark, table_name) == [1, 5]

        # An unconditional delete counts only the rows still live after earlier deletes.
        assert _delete_count(spark, f"DELETE FROM {table_name}") == 2  # noqa: PLR2004
        assert _ids(spark, table_name) == []
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
