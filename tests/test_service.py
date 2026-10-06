import numpy as np
import pandas as pd
import pytest

from housing_app.service import PredictionService


def test_batch_predictions_have_ordered_intervals_and_flags(bundle):
    service = PredictionService(bundle)
    sample = bundle.predictions[bundle.features].head(40)
    bad = pd.DataFrame([{**sample.iloc[0].to_dict(), "MedInc": "not a number"}])
    scored, problems = service.predict_frame(pd.concat([sample, bad], ignore_index=True), level=0.9)
    good = scored[scored["valid"]]
    assert len(good) == 40 and problems and np.isnan(scored["prediction"].iloc[-1])
    assert (good["lower"] <= good["prediction"]).all() and (good["prediction"] <= good["upper"]).all()
    assert scored["out_of_distribution"].dtype == bool


def test_single_prediction_fills_missing_inputs_and_rejects_unknown_ones(bundle):
    service = PredictionService(bundle)
    result = service.predict_one({"MedInc": 8.0})
    assert result["inputs"]["HouseAge"] == pytest.approx(float(service.typical["HouseAge"]))
    with pytest.raises(ValueError, match="unknown feature"):
        service.predict_one({"Bedrooms": 3})


def test_explanation_adds_up_to_the_prediction(bundle):
    result = PredictionService(bundle).explain({"MedInc": 9.0, "Latitude": 37.8, "Longitude": -122.4})
    total = result["baseline"] + sum(c["contribution"] for c in result["contributions"])
    assert total == pytest.approx(result["prediction"], abs=1e-6)
    assert [c["feature"] for c in result["contributions"]][-1] == "location"


def test_drift_detects_a_simulated_income_shift(bundle):
    service = PredictionService(bundle)
    sample = bundle.dataset[bundle.dataset["split"] == "test"][bundle.features]
    shifted = sample.assign(MedInc=sample["MedInc"] * 1.6)
    table, _ = service.drift(shifted)
    assert table.iloc[0]["feature"] == "MedInc" and table.iloc[0]["status"] != "stable"
