//! Error types for Python data source operations.
//!
//! Provides structured error types with context for debugging Python datasource issues.

use datafusion_common::DataFusionError;
use sail_common_datafusion::error::PythonDataSourceFailure;
use thiserror::Error;

const FAILURE_KIND_ATTRIBUTE: &str = "__sail_data_source_failure_kind__";

fn declared_failure_kind(error: &pyo3::PyErr) -> Option<PythonDataSourceFailure> {
    use pyo3::prelude::PyAnyMethods;
    use pyo3::types::{PyTuple, PyTupleMethods, PyType};

    pyo3::Python::attach(|py| {
        let type_type = py.get_type::<PyType>();
        let getattribute = type_type.getattr("__getattribute__").ok()?;
        let exception_type = error.get_type(py);
        let mro = getattribute
            .call1((&exception_type, "__mro__"))
            .ok()?
            .cast_into::<PyTuple>()
            .ok()?;

        for base in mro.iter() {
            let namespace = getattribute.call1((&base, "__dict__")).ok()?;
            let Ok(value) = namespace.get_item(FAILURE_KIND_ATTRIBUTE) else {
                continue;
            };
            let Ok(value) = value.extract::<String>() else {
                return None;
            };
            return match value.as_str() {
                "terminal" => Some(PythonDataSourceFailure::Terminal),
                "transient" => Some(PythonDataSourceFailure::Transient),
                _ => None,
            };
        }
        None
    })
}

/// Result type alias for Python data source operations.
#[expect(dead_code)]
pub type PythonDataSourceResult<T> = Result<T, PythonDataSourceError>;

/// Errors specific to Python data source operations.
#[derive(Debug, Error)]
pub enum PythonDataSourceError {
    /// Error from Python execution
    #[error("Python error: {0}")]
    PythonError(String),
    /// Schema validation error
    #[error("Schema error: {0}")]
    SchemaError(String),
    /// Version incompatibility
    #[error("Version error: {0}")]
    VersionError(String),
    /// Arrow conversion error
    #[error("Arrow conversion error: {0}")]
    ArrowError(String),
    /// DataFusion error
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] DataFusionError),
    /// Resource exhaustion (e.g., partition too large)
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    /// Application-declared failure with private Python details discarded.
    #[error("{0}")]
    DeclaredFailure(#[from] PythonDataSourceFailure),
}

impl PythonDataSourceError {
    /// Create a Python error with the given message.
    pub fn python(msg: impl Into<String>) -> Self {
        Self::PythonError(msg.into())
    }

    /// Create a schema error with the given message.
    pub fn schema(msg: impl Into<String>) -> Self {
        Self::SchemaError(msg.into())
    }

    /// Create a version error with the given message.
    pub fn version(msg: impl Into<String>) -> Self {
        Self::VersionError(msg.into())
    }

    /// Create an Arrow conversion error with the given message.
    pub fn arrow(msg: impl Into<String>) -> Self {
        Self::ArrowError(msg.into())
    }

    /// Create a resource exhausted error with the given message.
    pub fn resource_exhausted(msg: impl Into<String>) -> Self {
        Self::ResourceExhausted(msg.into())
    }
}

/// Context for Python datasource operations, used for enhanced error reporting.
///
/// This struct captures the datasource name and current operation to provide
/// better error messages when Python operations fail.
#[derive(Debug, Clone)]
pub struct PythonDataSourceContext {
    /// Name of the datasource being operated on
    pub datasource_name: String,
    /// Current operation (e.g., "schema", "partitions", "read")
    pub operation: &'static str,
}

impl PythonDataSourceContext {
    /// Create a new context for error reporting.
    pub fn new(datasource_name: impl Into<String>, operation: &'static str) -> Self {
        Self {
            datasource_name: datasource_name.into(),
            operation,
        }
    }

    /// Wrap an error message with context information.
    pub fn wrap_error(&self, msg: impl Into<String>) -> PythonDataSourceError {
        PythonDataSourceError::python(format!(
            "[{}::{}] {}",
            self.datasource_name,
            self.operation,
            msg.into()
        ))
    }

    /// Wrap a Python error with context information, preserving traceback.
    pub fn wrap_py_error(&self, e: pyo3::PyErr) -> PythonDataSourceError {
        match declared_failure_kind(&e) {
            Some(failure) => failure.into(),
            None => self.wrap_error(format_py_error_with_traceback(e)),
        }
    }
}

impl From<PythonDataSourceError> for DataFusionError {
    fn from(e: PythonDataSourceError) -> Self {
        DataFusionError::External(Box::new(e))
    }
}

/// Format a Python error with its traceback for better debugging.
///
/// This extracts the full Python traceback when available, making it much
/// easier to debug Python datasource errors.
pub fn format_py_error_with_traceback(e: pyo3::PyErr) -> String {
    use pyo3::types::PyTracebackMethods;

    pyo3::Python::attach(|py| {
        let traceback = e
            .traceback(py)
            .and_then(|tb| tb.format().ok())
            .unwrap_or_default();

        if traceback.is_empty() {
            e.to_string()
        } else {
            format!("{}\nTraceback:\n{}", e, traceback)
        }
    })
}

impl From<pyo3::PyErr> for PythonDataSourceError {
    fn from(e: pyo3::PyErr) -> Self {
        match declared_failure_kind(&e) {
            Some(failure) => failure.into(),
            None => Self::python(format_py_error_with_traceback(e)),
        }
    }
}

/// Convert PyO3 error to DataFusion error, preserving traceback.
///
/// This is a shared helper to avoid duplicating this conversion pattern
/// across multiple modules (stream.rs, executor.rs, arrow_utils.rs, etc.).
pub fn py_err(e: pyo3::PyErr) -> DataFusionError {
    match declared_failure_kind(&e) {
        Some(failure) => PythonDataSourceError::from(failure).into(),
        None => DataFusionError::External(Box::new(std::io::Error::other(
            format_py_error_with_traceback(e),
        ))),
    }
}

/// Import cloudpickle from PySpark.
///
/// Uses `pyspark.cloudpickle` which is bundled with PySpark, avoiding
/// the need for a separate cloudpickle package installation.
pub fn import_cloudpickle(
    py: pyo3::Python<'_>,
) -> Result<pyo3::Bound<'_, pyo3::types::PyModule>, DataFusionError> {
    py.import("pyspark.cloudpickle").map_err(|e| {
        DataFusionError::Execution(format!(
            "Failed to import pyspark.cloudpickle: {}. \
            PySpark must be installed to use Python data sources.",
            e
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::ffi::c_str;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods};

    #[expect(clippy::unwrap_used)]
    fn declared_error(kind: &str) -> PyErr {
        Python::initialize();
        Python::attach(|py| {
            let namespace = PyDict::new(py);
            namespace.set_item("failure_kind", kind).unwrap();
            py.run(
                c_str!(
                    "class DeclaredError(RuntimeError):\n    __sail_data_source_failure_kind__ = failure_kind\n"
                ),
                Some(&namespace),
                None,
            )
            .unwrap();
            let exception_type = namespace.get_item("DeclaredError").unwrap().unwrap();
            let value = exception_type.call1(("private Python detail",)).unwrap();
            PyErr::from_value(value)
        })
    }

    fn assert_declared_failure_is_constant(kind: &str, message: &str) -> Result<(), String> {
        match py_err(declared_error(kind)) {
            DataFusionError::External(error) => {
                assert_eq!(error.to_string(), message);
                assert!(!error.to_string().contains("private Python detail"));
                let Some(source) = error.source() else {
                    return Err("classified marker source was not preserved".to_string());
                };
                assert_eq!(source.to_string(), message);
                assert!(source.source().is_none());
                Ok(())
            }
            other => Err(format!("expected external error, got {other:?}")),
        }
    }

    #[test]
    fn test_terminal_declared_failure_preserves_only_finite_marker() -> Result<(), String> {
        assert_declared_failure_is_constant(
            "terminal",
            "Python data source reported a terminal failure",
        )
    }

    #[test]
    fn test_transient_declared_failure_preserves_only_finite_marker() -> Result<(), String> {
        assert_declared_failure_is_constant(
            "transient",
            "Python data source reported a transient failure",
        )
    }
}
