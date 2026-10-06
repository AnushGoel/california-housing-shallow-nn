# Changelog

All notable changes are recorded here. Versions follow [semantic versioning](https://semver.org).

## 2.1.1 (2026-10-07)

### Added
- Results of the first full training run: artifacts with fingerprint `f0d7955fc3a6`, the generated `docs/RESULTS.md`, figures and the README summary.
- Final interpretations in the notebook, written from the executed outputs.
- `results/` with the executed notebook and its PDF, so every output can be read on GitHub.

### Changed
- The README badges use relative links, so the repository works under any account name without edits.

### Fixed
- `.gitattributes` keeps artifact files byte for byte, so their checksums survive a commit made on Windows.
- The spatial cross-validation now clusters block groups with a NumPy implementation of k-means, because scikit-learn's `KMeans` crashes on some Windows installations with an outdated `threadpoolctl`.

## 2.1.0 (2026-10-06)

### Added
- Split-conformal prediction intervals for every prediction, with test-set coverage reported in the notebook and the app.
- Exact Shapley explanations ("why this price") against a typical block group.
- Batch scoring with input validation, out-of-distribution flags and drift statistics (PSI and KS), with CSV upload and download.
- A REST API (FastAPI), Docker images for the dashboard and the API, and a `docker-compose.yml`.
- A command line: `python -m housing_app info | validate | predict | explain | score | drift | report | publish`.
- An evidence report generator that writes `docs/RESULTS.md`, small SVG figures and the README summary from the artifacts.
- A SHA-256 manifest for the artifacts, verified on every load; gzip-compressed artifacts; a repository size budget in CI.
- Notebook built from plain-text sources, with optional rewording "editions" kept out of git.
- Methodology, evaluation, architecture, model card, datasheet and reference documents.
- Packaging (`pip install -e .` gives a `housing-app` command), pre-commit hooks, Dependabot and issue templates.

### Changed
- Model labels in the app are now "Final model" and "Tuned champion".

## 2.0.0

- Streamlit dashboard with nine pages, S3-compatible storage for saved scenarios and scores, a guessing game,
  unit and app tests, and GitHub Actions.
- Notebook section on stronger testing: paired bootstrap, random and spatial cross-validation, slice tests.

## 1.0.0

- Research notebook: controlled experiments on width and activation, seed robustness, random-search tuning and a
  single held-out test evaluation.
