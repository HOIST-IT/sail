Feature: Iceberg DELETE reports the rows it deletes

  Background:
    Given variable location for temporary directory iceberg_delete
    Given final statement
      """
      DROP TABLE IF EXISTS iceberg_delete_table
      """

  Scenario: DELETE rewrites only candidate files and reports the deleted rows
    Given statement template
      """
      CREATE TABLE iceberg_delete_table (id INT, value STRING)
      USING iceberg
      LOCATION {{ location.uri }}
      """
    Given statement
      """
      INSERT INTO iceberg_delete_table VALUES (1, 'remove'), (2, 'keep-first-file')
      """
    Given statement
      """
      INSERT INTO iceberg_delete_table VALUES (10, 'keep-second-file')
      """
    Given remember current iceberg data manifest paths
    When query
      """
      DELETE FROM iceberg_delete_table WHERE id = 1
      """
    Then query result
      | count |
      | 1     |
    Then iceberg snapshot operation is overwrite
    Then iceberg current data manifests reuse 1 remembered paths
    Then iceberg snapshot count is 3
    When query
      """
      SELECT * FROM iceberg_delete_table ORDER BY id
      """
    Then query result ordered
      | id | value            |
      | 2  | keep-first-file  |
      | 10 | keep-second-file |

  Scenario: DELETE retains rows where the predicate is false or null
    Given statement template
      """
      CREATE TABLE iceberg_delete_table (id INT, score INT)
      USING iceberg
      LOCATION {{ location.uri }}
      """
    Given statement
      """
      INSERT INTO iceberg_delete_table VALUES (1, NULL), (2, 10), (3, 1)
      """
    When query
      """
      DELETE FROM iceberg_delete_table WHERE score > 5
      """
    Then query result
      | count |
      | 1     |
    When query
      """
      SELECT * FROM iceberg_delete_table ORDER BY id
      """
    Then query result collected ordered
      | id | score |
      | 1  | NULL  |
      | 3  | 1     |

  Scenario: DELETE without WHERE removes all rows from metadata
    Given statement template
      """
      CREATE TABLE iceberg_delete_table (id INT)
      USING iceberg
      LOCATION {{ location.uri }}
      """
    Given statement
      """
      INSERT INTO iceberg_delete_table VALUES (1), (2), (3)
      """
    When query
      """
      DELETE FROM iceberg_delete_table
      """
    Then query result
      | count |
      | 3     |
    Then iceberg snapshot operation is delete
    Then iceberg snapshot count is 2
    When query
      """
      SELECT * FROM iceberg_delete_table
      """
    Then query result
      | id |

  Scenario: DELETE with no candidate files is a validated no-op
    Given statement template
      """
      CREATE TABLE iceberg_delete_table (id INT)
      USING iceberg
      LOCATION {{ location.uri }}
      """
    Given statement
      """
      INSERT INTO iceberg_delete_table VALUES (1), (2), (3)
      """
    When query
      """
      DELETE FROM iceberg_delete_table WHERE id = 999
      """
    Then query result
      | count |
      | 0     |
    Then iceberg snapshot count is 1
    Then iceberg snapshot operation is append
    When query
      """
      SELECT * FROM iceberg_delete_table ORDER BY id
      """
    Then query result ordered
      | id |
      | 1  |
      | 2  |
      | 3  |
