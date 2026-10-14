# Anonymization protocol

How the shipped `examples/sample_data/` files were de-identified before they
left the study machine, and how to re-run the same protocol on your own data
(`python -m lumbarmri.deidentify`).

**Scope note.** This is metadata/header/name de-identification, **not**
defacing. Voxel content is preserved bit-for-bit by design (check D4). Do not
use this module for head MRI, where facial surface may be reconstructed from
the volume.

## Layer 1 — metadata dictionary scrubbing

Before writing, every `MetaDataDictionary` key is removed unless it is on the
technical whitelist (geometry and quantisation fields only: size, spacing,
origin, direction, pixel type and related IO keys).

Always dropped, in addition to the whitelist rule:

| group | examples |
|---|---|
| free-text keys | `descrip`, `aux_file`, `intent_name`, `db_name`, `ITK_FileNotes` |
| patient / institution patterns | 7 regex families covering patient name/ID/birth, study/series IDs, institution, physician, comments |

## Layer 2 — NIfTI-1 header text-field zeroing

After writing, five fixed text fields of the 348-byte NIfTI-1 header are
forced to zero **at the byte level**, so the guarantee does not depend on
writer behaviour:

| field | offset (bytes) | length (bytes) |
|---|---|---|
| `data_type` | 4 | 10 |
| `db_name` | 14 | 18 |
| `descrip` | 148 | 80 |
| `aux_file` | 228 | 24 |
| `intent_name` | 328 | 16 |

Measured on the shipped study files these fields were *already* all zero;
the byte patch is a safety net ("belt and braces"). Output `.nii.gz` is
re-compressed deterministically (gzip mtime = 0) so runs are reproducible.

## Layer 3 — filename anonymisation

Real numeric IDs are replaced by sequential `case_NN` labels through a JSON
mapping file (`{ "<real id>": "case_NN", ... }`):

- replacement order is longest-ID-first, so overlapping numeric stems
  cannot produce collisions;
- mask suffixes are unified to `_label` (`_roi` → `_label`), the
  toolkit-wide naming convention (see `docs/ADAPTING_TO_YOUR_DATA.md`);
- the result must match `^case_\d{2}_(image|ME_label|P_label)\.nii\.gz$`.

**Mapping-file policy.** The real-ID ↔ case mapping is kept **outside the
repository**, always. `.gitignore` excludes `anonymization_map.*`; never
commit it, attach it to an issue, or paste it into documentation.

## Verification (`verify_dir`, four checks)

| check | rule |
|---|---|
| **D1** | every `.nii.gz` header text field is all zero |
| **D2** | every file name matches the anonymous pattern above |
| **D3** | no forbidden token anywhere: exact real IDs (from the local map) are byte-searched in *whole* files; structural patterns (pseudonym-style `MRY…` tokens, the study table directory token, the CJK study word) are regex-scanned in text files and in imaging-file *header regions* only — scanning raw voxel bytes with short regexes would create false positives |
| **D4** | pixel-array MD5 and geometry (size / spacing / origin / direction) identical before vs after de-identification, compared per file pair |

Run against a candidate directory:

```bash
python -m lumbarmri.deidentify check --dir examples/sample_data --src SRC --map /path/to/anonymization_map.json
python -m lumbarmri.deidentify verify   # synthetic self-check of the checks themselves
```

## Record for the shipped sample data

- 3 cases × 3 files (image + two ROI labels) = 9 files, ~31 MB.
- D1–D4: **PASS** on all 9 files at inclusion time; per-file pixel MD5 is
  recorded in `examples/sample_data/deidentify_manifest.json`.
- Repository-wide grep for the forbidden tokens: 0 hits.
- `git ls-files` imaging inventory: exactly those 9 files.
- Inclusion authorised (ethics / data-sharing confirmation obtained before
  the `data(sample)` commit).

## Limitations

- Voxel data are untouched; anything encoded *inside* the image content
  (burned-in annotations) is out of scope — visually inspect slices before
  sharing.
- D3 pattern scanning cannot detect a free-text identifier of arbitrary
  form; the strongest guarantee is Layer 1 (drop unknown metadata) plus D1.
- The protocol assumes NIfTI (`.nii.gz`). Other containers (DICOM, NRRD)
  need their own scrubbers.
