---
title: Python
rank: 3
---

# Python Data Sources

The Python data source allows you to extend the `SparkSession.read` and `DataFrame.write` APIs to support custom formats and external system integrations.
It optionally supports Arrow for zero-copy data exchange between the Python process and the Sail execution engine. This gives you flexibility in data source implementations without incurring performance penalties.

You can define a Python class that inherits from the `pyspark.sql.datasource.DataSource` abstract class, and register it to the Spark session to create a custom data source that can be used in the standard PySpark API. The `DataSource` class provides methods for defining the name and schema of the data source, as well as methods for creating readers and writers.

Currently, Sail supports Python data sources for batch reading and writing.

## Classified Reader Failures

By default, Sail propagates a Python data source exception with its Python
traceback. A data source can instead declare a finite failure category when its
caller must distinguish a deterministic failure from a retryable one without
inspecting exception text. Define `__sail_data_source_failure_kind__` on the
exception class with one of these exact values:

- `"terminal"` becomes a Spark `AnalysisException`.
- `"transient"` becomes a Spark `SparkRuntimeException`.

For a declared failure, Sail replaces the Python message and traceback with a
constant category message before the error crosses Spark Connect. Unknown
values retain the default exception behavior.

```python
class RetryableReadError(TimeoutError):
    __sail_data_source_failure_kind__ = "transient"


class ContractReadError(ValueError):
    __sail_data_source_failure_kind__ = "terminal"
```

Use this protocol only for data-source-controlled exception classes. Do not put
record values, credentials, endpoints, or other runtime details in the category
attribute.

## Examples

<!--@include: ../../_common/spark-session.md-->

### Batch Reader

<<< @/../python/pysail/tests/spark/datasource/test_python_read.txt{python-console}

### Batch Arrow Reader

<<< @/../python/pysail/tests/spark/datasource/test_python_read_arrow.txt{python-console}

### More Examples

Please refer to the [Spark documentation](https://spark.apache.org/docs/latest/api/python/tutorial/sql/python_data_source.html) for more Python data source examples, including how to define a batch writer. We will also add more examples to this guide in the future.
