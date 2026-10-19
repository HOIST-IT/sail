use bytes::Bytes;
use datafusion_common::{DataFusionError, Result, internal_err};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use url::Url;

use crate::io::StoreContext;
use crate::spec::snapshots::MAIN_BRANCH;
use crate::spec::{MetadataLog, SnapshotLog, TableMetadata, TableUpdate};
use crate::table::metadata_loader::{
    encode_metadata_file, metadata_file_extension_from_properties, table_metadata_location,
    write_version_hint,
};
use crate::utils::metadata::{
    ExistingMetadataFile, commit_state_unknown, failed_metadata_write, metadata_file_location,
    metadata_files_for_version, reconcile_existing_metadata_file,
};

#[derive(Clone, Copy)]
pub(crate) enum MetadataFileStyle {
    Versioned,
    Unique,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MetadataWriteOutcome {
    Written,
    Conflict,
}

/// A prepared metadata object. Catalog pointer publication remains the caller's responsibility.
pub(crate) struct MetadataFileCommit {
    path: Path,
    location: String,
    bytes: Bytes,
    version: i32,
    style: MetadataFileStyle,
    version_hint: String,
}

impl MetadataFileCommit {
    pub(crate) fn prepare(
        table_url: &Url,
        metadata: &TableMetadata,
        version: i32,
        style: MetadataFileStyle,
    ) -> Result<Self> {
        let json = metadata
            .to_json()
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        let extension = metadata_file_extension_from_properties(&metadata.properties)?;
        let (file, version_hint) = match style {
            MetadataFileStyle::Versioned => (
                format!("metadata/v{version}{extension}"),
                version.to_string(),
            ),
            MetadataFileStyle::Unique => {
                let name = format!("{version:05}-{}{extension}", uuid::Uuid::new_v4());
                (format!("metadata/{name}"), name)
            }
        };
        let location = table_metadata_location(table_url, &file)?;
        let bytes = encode_metadata_file(&file, &json)
            .map_err(|error| DataFusionError::External(Box::new(error)))?;
        Ok(Self {
            path: Path::from(file),
            location,
            bytes: Bytes::from(bytes),
            version,
            style,
            version_hint,
        })
    }

    pub(crate) fn location(&self) -> &str {
        &self.location
    }

    /// The metadata file path relative to the table location.
    pub(crate) fn file(&self) -> &str {
        self.path.as_ref()
    }

    /// Only definite conflicts may be retried.
    ///
    /// A versioned file publishes the commit as soon as it exists, so once its write may
    /// have been sent, every error is an unknown commit state and nothing is removed. A
    /// uniquely named file is published only by a later catalog pointer update, which the
    /// caller never sends after an error, so a put error other than `AlreadyExists` is a
    /// plain error for it.
    pub(crate) async fn write(&self, store: &StoreContext) -> Result<MetadataWriteOutcome> {
        match store
            .prefixed
            .put_opts(
                &self.path,
                self.bytes.clone().into(),
                object_store::PutOptions {
                    mode: object_store::PutMode::Create,
                    ..Default::default()
                },
            )
            .await
        {
            Ok(_) => {}
            // A retried put whose first attempt landed reports its own file as existing. The
            // file is read back before the write is treated as a lost race, and an unreadable
            // file fails as an unknown commit state so that the caller removes nothing.
            Err(error @ object_store::Error::AlreadyExists { .. }) => {
                match reconcile_existing_metadata_file(store, &self.path, &self.bytes).await? {
                    ExistingMetadataFile::Written => {
                        log::warn!(
                            "Metadata file {} for version {} was reported as existing but holds \
                             the bytes this commit sent, so the write landed: {error}",
                            self.path,
                            self.version
                        );
                    }
                    ExistingMetadataFile::Concurrent => {
                        return Ok(MetadataWriteOutcome::Conflict);
                    }
                }
            }
            Err(error) => {
                return Err(match self.style {
                    MetadataFileStyle::Versioned => failed_metadata_write(store, &self.path, error),
                    MetadataFileStyle::Unique => DataFusionError::External(Box::new(error)),
                });
            }
        }
        if matches!(self.style, MetadataFileStyle::Versioned) {
            let version_files = match metadata_files_for_version(store, self.version).await {
                Ok(files) => files,
                Err(error) => {
                    return Err(commit_state_unknown(format!(
                        "Iceberg metadata file {} was written, and listing metadata version {} \
                         to check for a concurrent writer failed: {error}; no files were removed",
                        metadata_file_location(store, &self.path),
                        self.version
                    )));
                }
            };
            let concurrent_files = version_files
                .iter()
                .filter(|file| *file != self.path.as_ref())
                .map(|file| metadata_file_location(store, &Path::from(file.as_str())).to_string())
                .collect::<Vec<_>>();
            if !concurrent_files.is_empty() {
                match store.prefixed.delete(&self.path).await {
                    Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
                    Err(error) => {
                        return Err(commit_state_unknown(format!(
                            "Iceberg metadata file {} was written beside {} for metadata \
                             version {}, and removing it failed: {error}; no other files were \
                             removed",
                            metadata_file_location(store, &self.path),
                            concurrent_files.join(", "),
                            self.version
                        )));
                    }
                }
                return Ok(MetadataWriteOutcome::Conflict);
            }
        }
        Ok(MetadataWriteOutcome::Written)
    }

    pub(crate) async fn write_hint(&self, store: &StoreContext) {
        write_version_hint(&store.prefixed, &self.version_hint).await;
    }
}

/// Apply snapshot actions and record the previous metadata version exactly once.
pub(crate) fn apply_snapshot_updates(
    metadata: &mut TableMetadata,
    updates: &[TableUpdate],
    previous_metadata_file: &str,
) -> Result<()> {
    let previous_timestamp = metadata.last_updated_ms;
    let timestamp_ms = crate::utils::timestamp::monotonic_timestamp_ms();
    for update in updates {
        match update {
            TableUpdate::AddSnapshot { snapshot } => {
                metadata.last_sequence_number = metadata
                    .last_sequence_number
                    .max(snapshot.sequence_number());
                if let Some(added_rows) = snapshot.added_rows {
                    metadata.advance_next_row_id(added_rows);
                }
                metadata.snapshots.push(snapshot.clone());
            }
            TableUpdate::SetSnapshotRef {
                ref_name,
                reference,
            } => {
                if ref_name == MAIN_BRANCH {
                    metadata.current_snapshot_id = Some(reference.snapshot_id);
                    metadata.snapshot_log.push(SnapshotLog {
                        timestamp_ms,
                        snapshot_id: reference.snapshot_id,
                    });
                }
                metadata.refs.insert(ref_name.clone(), reference.clone());
            }
            _ => return internal_err!("Unsupported Iceberg snapshot metadata update"),
        }
    }
    metadata.last_updated_ms = timestamp_ms;
    metadata.metadata_log.push(MetadataLog {
        timestamp_ms: previous_timestamp,
        metadata_file: previous_metadata_file.to_string(),
    });
    Ok(())
}
