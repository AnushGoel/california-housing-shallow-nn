import io
import json

import pandas as pd

from housing_app.cli import main


def run(*args):
    buffer = io.StringIO()
    code = main(list(args), out=buffer)
    return code, buffer.getvalue()


def test_predict_and_explain_print_json(artifacts_dir):
    code, text = run("--artifacts", str(artifacts_dir), "predict", "--MedInc", "6.5", "--json")
    assert code == 0 and json.loads(text)["prediction"] > 0
    code, text = run("--artifacts", str(artifacts_dir), "explain", "--MedInc", "6.5")
    assert code == 0 and "location" in text


def test_validate_info_score_and_drift(artifacts_dir, tmp_path):
    assert run("--artifacts", str(artifacts_dir), "validate")[0] == 0
    assert "fingerprint" in run("--artifacts", str(artifacts_dir), "info")[1]
    src = tmp_path / "in.csv"
    pd.read_csv(artifacts_dir / "test_predictions.csv.gz").head(25).to_csv(src, index=False)
    code, _ = run("--artifacts", str(artifacts_dir), "score", str(src), "-o", str(tmp_path / "out.csv"))
    assert code == 0 and {"prediction", "lower", "upper", "out_of_distribution"} <= set(pd.read_csv(tmp_path / "out.csv").columns)
    code, text = run("--artifacts", str(artifacts_dir), "drift", str(src))
    assert code == 0 and "PSI" in text


def test_missing_artifacts_give_a_clear_error(tmp_path):
    assert run("--artifacts", str(tmp_path / "nothing"), "info")[0] == 2
