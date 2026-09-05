// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BTreeMap;

use datafusion_common::{Result, plan_err};
use sail_common_datafusion::datasource::OptionLayer;

use crate::error::{DataSourceError, DataSourceResult};
use crate::spec::snapshots::Summary;

const SNAPSHOT_PROPERTY_PREFIX: &str = "snapshot-property.";

pub(crate) fn parse_optional_expected_snapshot_id(
    key: &str,
    value: &str,
) -> DataSourceResult<Option<i64>> {
    if value.is_empty() {
        Ok(None)
    } else {
        parse_expected_snapshot_id(key, value)
    }
}

pub(crate) fn parse_expected_snapshot_id(key: &str, value: &str) -> DataSourceResult<Option<i64>> {
    let snapshot_id = value
        .bytes()
        .all(|byte| byte.is_ascii_digit())
        .then(|| value.parse::<i64>().ok())
        .flatten()
        .filter(|snapshot_id| *snapshot_id > 0)
        .ok_or_else(|| DataSourceError::InvalidOption {
            key: key.to_string(),
            value: value.to_string(),
            cause: None,
        })?;
    Ok(Some(snapshot_id))
}

const RESERVED_SNAPSHOT_SUMMARY_KEYS: &[&str] = &[
    "operation",
    "added-data-files",
    "deleted-data-files",
    "total-data-files",
    "added-delete-files",
    "added-equality-delete-files",
    "removed-equality-delete-files",
    "added-position-delete-files",
    "removed-position-delete-files",
    "added-dvs",
    "removed-dvs",
    "removed-delete-files",
    "total-delete-files",
    "added-records",
    "deleted-records",
    "total-records",
    "added-files-size",
    "removed-files-size",
    "total-files-size",
    "added-position-deletes",
    "removed-position-deletes",
    "total-position-deletes",
    "added-equality-deletes",
    "removed-equality-deletes",
    "total-equality-deletes",
    "deleted-duplicate-files",
    "changed-partition-count",
    "partition-summaries-included",
    "wap.id",
    "published-wap-id",
    "source-snapshot-id",
    "replace-partitions",
    "manifests-created",
    "manifests-replaced",
    "manifests-kept",
    "entries-processed",
    "engine-name",
    "engine-version",
    "app-id",
    "iceberg-version",
];

fn strip_snapshot_property_prefix(key: &str) -> Option<&str> {
    key.get(..SNAPSHOT_PROPERTY_PREFIX.len())
        .filter(|prefix| prefix.eq_ignore_ascii_case(SNAPSHOT_PROPERTY_PREFIX))
        .map(|_| &key[SNAPSHOT_PROPERTY_PREFIX.len()..])
}

fn validate_snapshot_property_key(key: &str) -> Result<()> {
    if key.is_empty() || key.trim().is_empty() {
        return plan_err!("Iceberg snapshot property key cannot be empty");
    }
    if key.trim() != key {
        return plan_err!(
            "Iceberg snapshot property key `{key}` cannot have leading or trailing whitespace"
        );
    }
    if RESERVED_SNAPSHOT_SUMMARY_KEYS
        .iter()
        .any(|reserved| key.eq_ignore_ascii_case(reserved))
        || key
            .get(.."partitions.".len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("partitions."))
    {
        return plan_err!("Iceberg snapshot property key `{key}` is reserved");
    }
    Ok(())
}

pub(crate) fn validate_snapshot_properties(properties: &[(String, String)]) -> Result<()> {
    let mut previous_key: Option<&str> = None;
    for (key, _) in properties {
        validate_snapshot_property_key(key)?;
        if let Some(previous) = previous_key {
            if previous == key {
                return plan_err!("duplicate Iceberg snapshot property key `{key}`");
            }
            if previous > key.as_str() {
                return plan_err!("Iceberg snapshot properties are not in canonical key order");
            }
        }
        previous_key = Some(key);
    }
    Ok(())
}

pub(crate) fn extract_snapshot_properties(
    options: Vec<OptionLayer>,
) -> Result<(Vec<OptionLayer>, Vec<(String, String)>)> {
    let mut properties = BTreeMap::new();
    let mut clean_options = Vec::with_capacity(options.len());

    for layer in options {
        match layer {
            OptionLayer::OptionList { items } => {
                let mut clean_items = Vec::with_capacity(items.len());
                for (key, value) in items {
                    let Some(property_key) = strip_snapshot_property_prefix(&key) else {
                        clean_items.push((key, value));
                        continue;
                    };
                    validate_snapshot_property_key(property_key)?;
                    if let Some(previous_value) = properties.get(property_key) {
                        return plan_err!(
                            "duplicate Iceberg snapshot property key `{property_key}` has conflicting values `{previous_value}` and `{value}`"
                        );
                    }
                    properties.insert(property_key.to_string(), value);
                }
                clean_options.push(OptionLayer::OptionList { items: clean_items });
            }
            other => clean_options.push(other),
        }
    }

    let properties = properties.into_iter().collect::<Vec<_>>();
    validate_snapshot_properties(&properties)?;
    Ok((clean_options, properties))
}

pub(crate) fn insert_snapshot_properties(
    summary: &mut Summary,
    properties: &[(String, String)],
) -> Result<()> {
    validate_snapshot_properties(properties)?;
    for (key, value) in properties {
        if let Some(engine_value) = summary.additional_properties.get(key) {
            return plan_err!(
                "Iceberg snapshot property key `{key}` collides with engine summary value `{engine_value}` instead of caller value `{value}`"
            );
        }
    }
    summary
        .additional_properties
        .extend(properties.iter().cloned());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Operation;

    fn direct(items: &[(&str, &str)]) -> OptionLayer {
        OptionLayer::OptionList {
            items: items
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect(),
        }
    }

    #[test]
    fn extraction_is_case_insensitive_only_for_the_prefix_and_preserves_bytes() {
        let inherited = OptionLayer::TablePropertyList {
            items: vec![(
                "snapshot-property.inherited".to_string(),
                "must-not-propagate".to_string(),
            )],
        };
        let options = vec![
            inherited.clone(),
            direct(&[
                ("SnApShOt-PrOpErTy.Hoist.ID", " Value "),
                ("snapshot-property.hoist.id", ""),
            ]),
        ];

        let (clean, properties) = extract_snapshot_properties(options).expect("valid properties");

        assert_eq!(
            properties,
            vec![
                ("Hoist.ID".to_string(), " Value ".to_string()),
                ("hoist.id".to_string(), String::new()),
            ]
        );
        assert_eq!(clean[0], inherited);
        assert_eq!(clean[1], direct(&[]));
    }

    #[test]
    fn extraction_rejects_empty_whitespace_reserved_and_duplicate_suffixes() {
        for key in [
            "snapshot-property.",
            "snapshot-property. ",
            "snapshot-property. leading",
            "snapshot-property.trailing ",
            "snapshot-property.OpErAtIoN",
            "snapshot-property.Partitions.region",
            "snapshot-property.total-position-deletes",
            "snapshot-property.total-equality-deletes",
        ] {
            let error = extract_snapshot_properties(vec![direct(&[(key, "value")])])
                .expect_err("invalid snapshot property key must fail");
            assert!(
                error.to_string().contains("cannot") || error.to_string().contains("reserved"),
                "unexpected error for {key}: {error}"
            );
        }

        let error = extract_snapshot_properties(vec![
            direct(&[("snapshot-property.hoist.id", "one")]),
            direct(&[("SNAPSHOT-PROPERTY.hoist.id", "two")]),
        ])
        .expect_err("byte-identical suffix duplicates must fail");
        assert!(error.to_string().contains("duplicate"));
    }

    #[test]
    fn validation_requires_canonical_byte_order() {
        let error = validate_snapshot_properties(&[
            ("z".to_string(), "first".to_string()),
            ("A".to_string(), "second".to_string()),
        ])
        .expect_err("noncanonical order must fail");
        assert!(error.to_string().contains("canonical key order"));

        validate_snapshot_properties(&[
            ("A".to_string(), "first".to_string()),
            ("z".to_string(), "second".to_string()),
        ])
        .expect("byte-sorted properties are canonical");
    }

    #[test]
    fn insertion_rejects_engine_collision_without_overwriting_it() {
        let mut summary = Summary::new(Operation::Append);
        summary
            .additional_properties
            .insert("hoist.id".to_string(), "engine".to_string());

        let error = insert_snapshot_properties(
            &mut summary,
            &[("hoist.id".to_string(), "caller".to_string())],
        )
        .expect_err("engine collision must fail");

        assert!(error.to_string().contains("collides"));
        assert_eq!(
            summary.additional_properties.get("hoist.id"),
            Some(&"engine".to_string())
        );
    }

    #[test]
    fn expected_snapshot_id_parser_distinguishes_absence_from_explicit_invalid_values() {
        assert_eq!(
            parse_optional_expected_snapshot_id("caller-expected-snapshot-id", "")
                .expect("generated default is absence"),
            None
        );
        assert_eq!(
            parse_expected_snapshot_id("expected-snapshot-id", "42")
                .expect("positive decimal snapshot id"),
            Some(42)
        );
        for value in ["", " ", "0", "-1", "+1", "1 ", "1.0", "１２"] {
            assert!(
                parse_expected_snapshot_id("expected-snapshot-id", value).is_err(),
                "explicit value {value:?} must fail"
            );
        }
    }
}
