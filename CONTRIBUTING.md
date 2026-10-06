# Contributing

Thanks for taking an interest. The project is small enough that one rule covers most of it: every claim in the
documentation must be reproducible from the code and the artifacts.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

## Where things live

| You want to change... | Edit | Then run |
|---|---|---|
| the research notebook | `notebooks/source/part*.txt` | `python scripts/build_notebook.py` |
| the trained results | run the built notebook from `notebooks/` | `python -m housing_app report` |
| serving logic (predictions, intervals, explanations, drift) | `housing_app/service.py` and the module it calls | `pytest` |
| a dashboard page | `housing_app/views/<page>.py`; charts in `housing_app/charts.py` | `streamlit run app.py` |
| the API | `api/main.py` | `uvicorn api.main:app --reload` |

The notebook is generated from plain-text sources so that changes are reviewable as text. Never edit the `.ipynb`
directly; the pre-commit hook strips its outputs. The one executed copy that is kept, with its PDF, lives in `results/`
and is replaced after each full run.

## Storage policy

The repository stays small on purpose. Artifacts are gzip-compressed and listed in `artifacts/manifest.json`;
`python scripts/check_size.py` enforces per-folder budgets in CI, and the pre-commit hook rejects files over 1.5 MB.
Figures in `docs/figures/` are SVG with text kept as text.

## Before opening a pull request

- `pytest` passes and `ruff check .` is clean.
- If the artifacts changed, `python -m housing_app report` has been run and the regenerated `docs/RESULTS.md` is committed.
  CI checks this with `python -m housing_app report --check`.
- New statistical claims in the documentation cite their source (APA 7th edition) and point to the evidence.
