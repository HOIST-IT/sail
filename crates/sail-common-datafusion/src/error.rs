use std::collections::HashSet;
use std::fmt;
use std::fmt::Formatter;

use datafusion::arrow::error::ArrowError;
use datafusion_common::DataFusionError;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An error with a cause received from a remote worker.
#[derive(Debug, Clone, Error)]
#[error("remote error: {cause}")]
pub struct RemoteError {
    pub cause: CommonErrorCause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PythonFailureKind {
    Terminal,
    Transient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PythonDataSourceFailure {
    #[error("Python data source reported a terminal failure")]
    Terminal,
    #[error("Python data source reported a transient failure")]
    Transient,
}

impl PythonDataSourceFailure {
    pub fn kind(self) -> PythonFailureKind {
        match self {
            Self::Terminal => PythonFailureKind::Terminal,
            Self::Transient => PythonFailureKind::Transient,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PythonErrorCause {
    pub summary: String,
    pub traceback: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_kind: Option<PythonFailureKind>,
}

/// A trait to extract Python error cause from a generic error.
///
/// The implementation should inspect the provided error without
/// recursively inspecting its inner errors.
///
/// This trait avoids direct dependency on `PyO3` for this crate.
/// Since this crate is used by many other crates, we can therefore
/// avoid unnecessary recompilation when switching Python environments.
pub trait PythonErrorCauseExtractor {
    fn extract(error: &(dyn std::error::Error + 'static)) -> Option<PythonErrorCause>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CommonErrorCause {
    Unknown(String),
    Internal(String),
    NotImplemented(String),
    InvalidArgument(String),
    Io(String),
    Python(PythonErrorCause),
    ArrowCast(String),
    ArrowMemory(String),
    ArrowParse(String),
    ArrowCompute(String),
    ArrowIpc(String),
    ArrowCDataInterface(String),
    ArrowDivideByZero(String),
    ArrowArithmeticOverflow(String),
    ArrowDictionaryKeyOverflow(String),
    ArrowRunEndIndexOverflow(String),
    ArrowOffsetOverflow(String),
    FormatCsv(String),
    FormatJson(String),
    FormatParquet(String),
    FormatAvro(String),
    Plan(String),
    Schema(String),
    Configuration(String),
    Execution(String),
    DeltaTable(String),
}

impl fmt::Display for CommonErrorCause {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            CommonErrorCause::Unknown(x)
            | CommonErrorCause::Internal(x)
            | CommonErrorCause::NotImplemented(x)
            | CommonErrorCause::InvalidArgument(x)
            | CommonErrorCause::Io(x)
            | CommonErrorCause::ArrowCast(x)
            | CommonErrorCause::ArrowMemory(x)
            | CommonErrorCause::ArrowParse(x)
            | CommonErrorCause::ArrowCompute(x)
            | CommonErrorCause::ArrowIpc(x)
            | CommonErrorCause::ArrowCDataInterface(x)
            | CommonErrorCause::ArrowDivideByZero(x)
            | CommonErrorCause::ArrowArithmeticOverflow(x)
            | CommonErrorCause::ArrowDictionaryKeyOverflow(x)
            | CommonErrorCause::ArrowRunEndIndexOverflow(x)
            | CommonErrorCause::ArrowOffsetOverflow(x)
            | CommonErrorCause::FormatCsv(x)
            | CommonErrorCause::FormatJson(x)
            | CommonErrorCause::FormatParquet(x)
            | CommonErrorCause::FormatAvro(x)
            | CommonErrorCause::Plan(x)
            | CommonErrorCause::Schema(x)
            | CommonErrorCause::Configuration(x)
            | CommonErrorCause::Execution(x)
            | CommonErrorCause::DeltaTable(x) => write!(f, "{x}"),
            CommonErrorCause::Python(PythonErrorCause { summary, .. }) => {
                write!(f, "{summary}")
            }
        }
    }
}

impl CommonErrorCause {
    fn build<Py: PythonErrorCauseExtractor>(
        error: &(dyn std::error::Error + 'static),
        seen: &mut HashSet<*const dyn std::error::Error>,
    ) -> Self {
        let ptr = error as *const _;
        if seen.contains(&ptr) {
            return Self::Unknown("circular error reference".to_string());
        }
        seen.insert(ptr);
        if let Some(e) = error.downcast_ref::<ArrowError>() {
            return match e {
                ArrowError::NotYetImplemented(x) => Self::NotImplemented(x.clone()),
                ArrowError::ExternalError(e) => Self::build::<Py>(e.as_ref(), seen),
                ArrowError::CastError(x) => Self::ArrowCast(x.clone()),
                ArrowError::MemoryError(x) => Self::ArrowMemory(x.clone()),
                ArrowError::ParseError(x) => Self::ArrowParse(x.clone()),
                ArrowError::SchemaError(x) => Self::Schema(x.clone()),
                ArrowError::ComputeError(x) => Self::ArrowCompute(x.clone()),
                ArrowError::DivideByZero => Self::ArrowDivideByZero("divide by zero".to_string()),
                ArrowError::ArithmeticOverflow(x) => Self::ArrowArithmeticOverflow(x.clone()),
                ArrowError::CsvError(x) => Self::FormatCsv(x.clone()),
                ArrowError::JsonError(x) => Self::FormatJson(x.clone()),
                ArrowError::IoError(message, e) => Self::Io(format!("{message}: {e}")),
                ArrowError::IpcError(x) => Self::ArrowIpc(x.clone()),
                ArrowError::InvalidArgumentError(x) => Self::InvalidArgument(x.clone()),
                ArrowError::ParquetError(x) => Self::FormatParquet(x.clone()),
                ArrowError::AvroError(x) => Self::FormatAvro(x.clone()),
                ArrowError::CDataInterface(x) => Self::ArrowCDataInterface(x.clone()),
                ArrowError::DictionaryKeyOverflowError => {
                    Self::ArrowDictionaryKeyOverflow("dictionary key overflow".to_string())
                }
                ArrowError::RunEndIndexOverflowError => {
                    Self::ArrowRunEndIndexOverflow("run-end index overflow".to_string())
                }
                ArrowError::OffsetOverflowError(x) => {
                    Self::ArrowOffsetOverflow(format!("offset overflow: {x}"))
                }
            };
        }

        if let Some(e) = error.downcast_ref::<DataFusionError>() {
            return match e {
                DataFusionError::ArrowError(e, _) => Self::build::<Py>(e, seen),
                DataFusionError::ParquetError(e) => Self::FormatParquet(e.to_string()),
                DataFusionError::ObjectStore(e) => Self::Io(e.to_string()),
                DataFusionError::IoError(e) => Self::Io(e.to_string()),
                DataFusionError::NotImplemented(x) => Self::NotImplemented(x.clone()),
                DataFusionError::Internal(x) => Self::Internal(x.clone()),
                DataFusionError::Plan(x) => Self::Plan(x.clone()),
                DataFusionError::Configuration(x) => Self::Configuration(x.clone()),
                DataFusionError::SchemaError(e, _) => Self::Schema(e.to_string()),
                DataFusionError::Execution(x) => Self::Execution(x.clone()),
                DataFusionError::ExecutionJoin(e) => Self::Execution(e.to_string()),
                DataFusionError::ResourcesExhausted(x) => Self::Execution(x.clone()),
                DataFusionError::External(e) => Self::build::<Py>(e.as_ref(), seen),
                DataFusionError::Context(_, e) => Self::build::<Py>(e, seen),
                DataFusionError::Substrait(x) => Self::Unknown(x.clone()),
                DataFusionError::Diagnostic(_, e) => Self::build::<Py>(e.as_ref(), seen),
                DataFusionError::Collection(errors) => match errors.first() {
                    None => Self::Unknown("empty error collection".to_string()),
                    Some(e) => Self::build::<Py>(e, seen),
                },
                DataFusionError::Shared(e) => Self::build::<Py>(e.as_ref(), seen),
                DataFusionError::Ffi(x) => Self::Unknown(x.clone()),
            };
        }

        if let Some(failure) = error.downcast_ref::<PythonDataSourceFailure>() {
            return Self::Python(PythonErrorCause {
                summary: failure.to_string(),
                traceback: None,
                failure_kind: Some(failure.kind()),
            });
        }

        if let Some(cause) = Py::extract(error) {
            return Self::Python(cause);
        }

        if let Some(e) = error.downcast_ref::<RemoteError>() {
            return e.cause.clone();
        }

        if let Some(e) = error.source() {
            Self::build::<Py>(e, seen)
        } else {
            Self::Unknown(error.to_string())
        }
    }

    /// Recursively traverse the error source and determine the best error cause
    /// from the innermost error.
    pub fn new<Py: PythonErrorCauseExtractor>(error: &(dyn std::error::Error + 'static)) -> Self {
        Self::build::<Py>(error, &mut HashSet::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_failure_kind_round_trips() -> Result<(), Box<dyn std::error::Error>> {
        let cause = CommonErrorCause::Python(PythonErrorCause {
            summary: "Python data source reported a terminal failure".to_string(),
            traceback: None,
            failure_kind: Some(PythonFailureKind::Terminal),
        });

        let encoded = serde_json::to_string(&cause)?;
        assert!(encoded.contains(r#""failureKind":"terminal""#));
        let decoded: CommonErrorCause = serde_json::from_str(&encoded)?;
        let CommonErrorCause::Python(decoded) = decoded else {
            return Err(std::io::Error::other("expected Python error cause").into());
        };
        assert_eq!(decoded.failure_kind, Some(PythonFailureKind::Terminal));
        Ok(())
    }

    #[test]
    fn test_legacy_python_cause_defaults_failure_kind() -> Result<(), Box<dyn std::error::Error>> {
        let encoded = r#"{"python":{"summary":"legacy","traceback":null}}"#;
        let decoded: CommonErrorCause = serde_json::from_str(encoded)?;
        let CommonErrorCause::Python(decoded) = decoded else {
            return Err(std::io::Error::other("expected Python error cause").into());
        };
        assert_eq!(decoded.failure_kind, None);
        Ok(())
    }
}
