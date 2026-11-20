<!--
SPDX-FileCopyrightText: 2026 HOIST IT B.V.
SPDX-License-Identifier: Apache-2.0
-->

# DataFusion physical-plan override

Source: the crates.io `datafusion-physical-plan` 55.3.0 archive. Its Apache
licence and notice are retained unchanged.

`src/spill/spill_pool.rs` carries the production changes from Apache
DataFusion commit `09a2aff72d88c16499ae3266a943a4cd9841b618`
(PR [24891](https://github.com/apache/datafusion/pull/24891)). The reader
checks other open files when the oldest file has caught up with its writer.
It still waits for pending disk reads to preserve single-writer order.

The local regression `test_reader_drains_second_open_file` holds one file
out of the writer queue to reproduce two concurrent open files deterministically.
It requires both batches before either writer is dropped.

Run the vendored library tests separately from Sail:

```sh
cargo test --manifest-path vendor/datafusion-physical-plan/Cargo.toml --locked --lib
```

Remove the root Cargo patch and this directory once the chosen DataFusion
release includes the fix. Update the workspace lock at the same time.
