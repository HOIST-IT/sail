use datafusion_common::{DataFusionError, Result};
use futures::StreamExt;
use object_store::ObjectStoreExt;
use sail_catalog::error::CatalogError;

use crate::io::StoreContext;
use crate::table::metadata_loader::metadata_file_version_from_path;

/// List all metadata files in the table's `metadata/` prefix that correspond to the
/// given version number.
pub async fn metadata_files_for_version(
    store_ctx: &StoreContext,
    version: i32,
) -> Result<Vec<String>> {
    let prefix = object_store::path::Path::from("metadata/");
    let mut stream = store_ctx.prefixed.list(Some(&prefix));
    let mut matches = Vec::new();
    while let Some(meta) = stream.next().await {
        let meta = meta.map_err(|e| DataFusionError::External(Box::new(e)))?;
        if metadata_file_version_from_path(meta.location.as_ref()) == Some(version) {
            matches.push(meta.location.to_string());
        }
    }
    Ok(matches)
}

/// Who created a metadata file that a create-only write reported as already existing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExistingMetadataFile {
    /// The file holds exactly the bytes the write sent, so the write landed.
    Written,
    /// The file holds other bytes, so another writer created it.
    Concurrent,
}

/// Find out who created the metadata file at `path` after a create-only write of `written`
/// reported that the file already exists.
///
/// An object store retries a create-only put whose response was lost, for example after a
/// server error. When the first attempt landed, the retry reports that the file already
/// exists although this write created it. Only equal bytes prove that the file is this
/// write's own: its content names snapshot and manifest files that only this write produced.
///
/// When the file cannot be read, including when it is no longer there, its creator is
/// unknown. That is a [`CatalogError::CommitStateUnknown`] naming the path, and the caller
/// must not delete anything, because the file may be this write's own and already visible.
pub(crate) async fn reconcile_existing_metadata_file(
    store_ctx: &StoreContext,
    path: &object_store::path::Path,
    written: &[u8],
) -> Result<ExistingMetadataFile> {
    let existing = match store_ctx.prefixed.get(path).await {
        Ok(result) => result.bytes().await,
        Err(error) => Err(error),
    };
    match existing {
        Ok(existing) if existing.as_ref() == written => Ok(ExistingMetadataFile::Written),
        Ok(_) => Ok(ExistingMetadataFile::Concurrent),
        Err(error) => {
            let location = store_ctx
                .prefix_path
                .parts()
                .chain(path.parts())
                .collect::<object_store::path::Path>();
            Err(DataFusionError::External(Box::new(
                CatalogError::CommitStateUnknown(format!(
                    "the create-only write of Iceberg metadata file {location} reported that the \
                     file already exists, and the file could not be read to tell whether this \
                     write created it: {error}; no files were removed"
                )),
            )))
        }
    }
}
