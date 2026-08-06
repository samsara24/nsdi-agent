# Legacy Exploration Archive

Historical scripts, `saved_methods/`, reports, and `outputs/` are retained under `../../before/` as exploration evidence. They are not imported by `rca_framework/` and are not treated as the new method.

- `source_data_manifest.json` freezes the name, byte size, and SHA-256 digest of every original file under `data/`.
- Original sensitive cases remain only under `data/`; the v2 preparation command never writes there.
- Existing output directories remain unchanged. New method artifacts are written under `artifacts/`.
- Historical result numbers must be cited as exploration results, not as v2 results.
