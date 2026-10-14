# Push checklist

Run this matrix once before the first `git push`, and again after any later
change to code, configs, docs or sample data. Every command is run from the
repository root. Expected results are stated inline — anything else is a
blocker.

## 1. Cleanliness

- [ ] `git status` — clean working tree (no untracked leftovers).
- [ ] `git log --stat` — no commit touches `outputs/`, build artefacts or
      large ad-hoc data; imaging inventory in git is exactly the 9 files of
      `examples/sample_data/`:
      `git ls-files | grep -E '\.nii(\.gz)?$'` → 9 lines.
- [ ] No real study IDs / study-specific column names anywhere in tracked
      files (the de-identification forbidden-token scan over the repo text
      must report 0 hits — `python -m lumbarmri.deidentify check` does this
      for a directory; for the repo, grep the tracked text files).
- [ ] The ID mapping file is **not** tracked:
      `git ls-files | grep -i anonymization_map` → 0 lines
      (`.gitignore` already excludes `anonymization_map.*`).

## 2. Tests and self-checks

- [ ] `python -m pytest` — all green (38 tests as of the last release).
- [ ] Seven module self-checks, all PASS:
      ```bash
      python -m lumbarmri.preprocess verify
      python -m lumbarmri.normalization verify
      python -m lumbarmri.gmm_segmentation verify
      python -m lumbarmri.zsplit verify
      python -m lumbarmri.bic_lasso verify
      python -m lumbarmri.synth verify
      python -m lumbarmri.deidentify verify
      ```
- [ ] `python examples/run_demo.py` — ends with
      `demo finished …, all steps OK`, BIC-LASSO baseline comparison
      `Overall: PASS`, and both λ selections agreeing on the variable set.

## 3. Data release checks (only if sample data changed)

- [ ] `python -m lumbarmri.deidentify check --dir examples/sample_data --src SRC --map MAP`
      → D1–D4 all PASS (`SRC`/`MAP` stay on the study machine).
- [ ] `examples/sample_data/deidentify_manifest.json` updated and consistent
      with the committed files (names + pixel MD5s).
- [ ] Ethics / data-sharing authorisation confirmed for every case included.

## 4. Documentation

- [ ] README quick-start commands copy-pasted and run at least once
      (`python examples/run_demo.py` is the smoke test).
- [ ] Links resolve: `docs/ALGORITHM_NOTES.md`, `docs/ANONYMIZATION.md`,
      `docs/ADAPTING_TO_YOUR_DATA.md`, `examples/README.md`.
- [ ] `LICENSE` present (MIT).

## 5. Publish

```bash
git remote add origin git@github.com:<you>/lumbar-gmm-fmr-toolkit.git
git push -u origin main
```

After pushing (manual, on GitHub):

- [ ] Repository is **public** if that is the intent; default branch `main`.
- [ ] File browser shows `examples/sample_data/` with 9 files and nothing
      else large (repo size sane; no Git LFS needed for ~31 MB).
- [ ] README renders correctly on the repo home page.
- [ ] A release/tag is optional; if tagged, the tag message points at the
      frozen conventions in `docs/ALGORITHM_NOTES.md`.

## Rollback

Sample data is the only hard-to-undo part of the history. If a privacy
concern ever appears: `git rm -r --cached examples/sample_data`, commit,
and rewrite history **before** others have cloned
(`git filter-repo --path examples/sample_data --invert-paths`), then force
push. Prefer preventing this: run section 3 religiously.
