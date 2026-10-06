The research notebook's export step writes the trained artifacts here: `results.json`, `histories.json.gz`,
`dataset.csv.gz`, `test_predictions.csv.gz`, `model_final.npz`, `model_tuned.npz` and `manifest.json` (SHA-256 checksums).

Commit those files (the `runs/` cache is ignored), then run `python -m housing_app report` to regenerate
`docs/RESULTS.md`. `python -m housing_app validate` checks the checksums at any time.
