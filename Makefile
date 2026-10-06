# Common tasks:  make <target>   (one-line recipes, so no tab characters are needed)
.PHONY: help setup notebook test lint app api report check-report size docker clean
help: ; @echo "targets: setup notebook test lint app api report check-report size docker clean"
setup: ; pip install -r requirements-dev.txt && pre-commit install
notebook: ; python scripts/build_notebook.py
test: ; pytest -q
lint: ; ruff check .
app: ; streamlit run app.py
api: ; uvicorn api.main:app --reload
report: ; python -m housing_app report
check-report: ; python -m housing_app report --check
size: ; python scripts/check_size.py
docker: ; docker compose up --build
clean: ; rm -rf .pytest_cache .ruff_cache user_data && find . -name __pycache__ -prune -exec rm -rf {} +
