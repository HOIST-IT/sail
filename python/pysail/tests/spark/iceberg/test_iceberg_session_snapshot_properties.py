from contextlib import contextmanager

import pytest
from pyiceberg.table import StaticTable

from pysail.tests.spark.iceberg.utils import pyiceberg_file_io_properties

RUN_ID = "spark.sql.iceberg.snapshot-property.hoist.run-id"
JOB = "spark.sql.iceberg.snapshot-property.hoist.job"
EXPECTED_SNAPSHOT = "spark.sail.iceberg.expected-snapshot-id"

TABLE_MODES = {
    "copy-on-write": "'format-version' = '2'",
    "merge-on-read": (
        "'format-version' = '2', 'write.delete.mode' = 'merge-on-read', "
        "'write.update.mode' = 'merge-on-read', 'write.merge.mode' = 'merge-on-read'"
    ),
    "merge-on-read-v3": (
        "'format-version' = '3', 'write.delete.mode' = 'merge-on-read', "
        "'write.update.mode' = 'merge-on-read', 'write.merge.mode' = 'merge-on-read'"
    ),
}

ROW_LEVEL_COMMANDS = {
    "merge": """
        MERGE INTO {table} t USING (SELECT 2L AS id, 'B' AS category, 21L AS value) s
        ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET value = s.value
        WHEN NOT MATCHED THEN INSERT *
        """,
    "update": "UPDATE {table} SET value = 31 WHERE id = 3",
    "delete": "DELETE FROM {table} WHERE id = 2",
    # Removes whole data files only, so it commits without a write.
    "metadata-delete": "DELETE FROM {table} WHERE category = 'A'",
}


@contextmanager
def _session_conf(spark, settings):
    for key, value in settings.items():
        spark.conf.set(key, value)
    try:
        yield
    finally:
        for key in settings:
            spark.conf.unset(key)


def _create_table(spark, table_name, location, mode):
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(
        f"""
        CREATE TABLE {table_name} (id BIGINT, category STRING, value BIGINT)
        USING iceberg
        PARTITIONED BY (category)
        LOCATION '{location.as_uri()}'
        TBLPROPERTIES ({TABLE_MODES[mode]})
        """
    )
    spark.sql(
        f"""
        INSERT INTO {table_name} SELECT /*+ COALESCE(1) */ * FROM VALUES
        (1L, 'A', 10L), (2L, 'B', 20L), (3L, 'B', 30L)
        """
    )


def _current_snapshot(location):
    table = StaticTable.from_metadata(str(location), properties=pyiceberg_file_io_properties())
    snapshot = table.current_snapshot()
    assert snapshot is not None
    assert snapshot.summary is not None
    return snapshot


def _rows(spark, table_name):
    return [tuple(row) for row in spark.table(table_name).orderBy("id").collect()]


@pytest.mark.parametrize("command", list(ROW_LEVEL_COMMANDS))
@pytest.mark.parametrize("mode", list(TABLE_MODES))
def test_iceberg_row_level_commit_carries_session_snapshot_properties(spark, tmp_path, mode, command):
    table_name = "iceberg_row_level_session_properties"
    location = tmp_path / table_name
    _create_table(spark, table_name, location, mode)
    try:
        planned = _current_snapshot(location)

        settings = {RUN_ID: "run-1", EXPECTED_SNAPSHOT: str(planned.snapshot_id)}
        with _session_conf(spark, settings):
            spark.sql(ROW_LEVEL_COMMANDS[command].format(table=table_name))

        committed = _current_snapshot(location)
        assert committed.parent_snapshot_id == planned.snapshot_id
        assert committed.summary.additional_properties["hoist.run-id"] == "run-1"

        # The settings end with the session configuration.
        spark.sql(f"INSERT INTO {table_name} VALUES (9L, 'C', 90L)")
        assert "hoist.run-id" not in _current_snapshot(location).summary.additional_properties
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


@pytest.mark.parametrize("mode", list(TABLE_MODES))
def test_iceberg_row_level_command_refuses_a_stale_session_expected_snapshot(spark, tmp_path, mode):
    table_name = "iceberg_row_level_stale_session_snapshot"
    location = tmp_path / table_name
    _create_table(spark, table_name, location, mode)
    try:
        stale = _current_snapshot(location)
        spark.sql(f"INSERT INTO {table_name} VALUES (4L, 'C', 40L)")
        current = _current_snapshot(location)
        rows_before = _rows(spark, table_name)
        metadata_before = set((location / "metadata").glob("*.metadata.json"))

        settings = {RUN_ID: "run-2", EXPECTED_SNAPSHOT: str(stale.snapshot_id)}
        expected_error = rf"expected snapshot {stale.snapshot_id} does not match current snapshot {current.snapshot_id}"
        with _session_conf(spark, settings), pytest.raises(Exception, match=expected_error):
            spark.sql(ROW_LEVEL_COMMANDS["merge"].format(table=table_name))

        assert _current_snapshot(location).snapshot_id == current.snapshot_id
        assert set((location / "metadata").glob("*.metadata.json")) == metadata_before
        assert _rows(spark, table_name) == rows_before
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")


def test_iceberg_write_option_overrides_the_session_snapshot_property(spark, tmp_path):
    table_name = "iceberg_write_session_properties"
    location = tmp_path / table_name
    _create_table(spark, table_name, location, "copy-on-write")
    try:
        schema = "id BIGINT, category STRING, value BIGINT"
        with _session_conf(spark, {RUN_ID: "session-run", JOB: "session-job"}):
            (
                spark.createDataFrame([(5, "C", 50)], schema=schema)
                .writeTo(table_name)
                .option("snapshot-property.hoist.run-id", "write-run")
                .append()
            )

        properties = _current_snapshot(location).summary.additional_properties
        assert properties["hoist.run-id"] == "write-run"
        assert properties["hoist.job"] == "session-job"
    finally:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
