import math

import numpy as np
import pytest

from housing_app.model import ACTIVATIONS, ShallowNet, erf


def tiny(activation="relu", mode="raw"):
    rng = np.random.default_rng(0)
    return ShallowNet(W1=rng.normal(size=(3, 4)), b1=rng.normal(size=4), W2=rng.normal(size=(4, 1)), b2=np.array([0.5]),
                      activation=activation, feature_mode=mode, log_idx=np.array([1]),
                      scaler_mean=np.array([1.0, 0.5, -2.0]), scaler_scale=np.array([2.0, 1.0, 4.0]),
                      feature_names=("a", "b", "c"))


def test_predict_matches_a_manual_forward_pass_for_every_activation():
    x = np.array([[3.0, 2.0, 0.0], [1.0, 0.5, -2.0]])
    for name, f in ACTIVATIONS.items():
        net = tiny(name)
        z = ((x - net.scaler_mean) / net.scaler_scale) @ net.W1 + net.b1
        np.testing.assert_allclose(net.predict(x), (f(z) @ net.W2 + net.b2).ravel(), rtol=1e-12, atol=1e-12)


def test_log_mode_transforms_only_the_listed_columns():
    x = np.array([[3.0, 2.0, 0.0]])
    raw, logged = tiny(mode="raw").inputs(x), tiny(mode="log").inputs(x)
    assert float(logged[0, 0]) == pytest.approx(float(raw[0, 0]))
    assert float(logged[0, 2]) == pytest.approx(float(raw[0, 2]))
    assert float(logged[0, 1]) == pytest.approx(math.log1p(2.0) - 0.5)


def test_erf_approximation_is_accurate():
    grid = np.linspace(-5, 5, 2001)
    exact = np.array([math.erf(v) for v in grid])
    assert float(np.max(np.abs(erf(grid) - exact))) < 2e-7


def test_unit_contributions_add_up_to_the_prediction():
    net, x = tiny("tanh"), np.array([[0.3, 1.2, -1.0]])
    assert float(net.contributions(x).sum() + net.b2[0]) == pytest.approx(float(net.predict(x)[0]))


def test_wrong_number_of_features_is_rejected():
    with pytest.raises(ValueError, match="expected 3 features"):
        tiny().predict(np.ones((1, 5)))


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        tiny("swishy")


def test_exported_models_reproduce_the_keras_predictions(bundle):
    for label, net in bundle.models.items():
        assert net.self_check() < 1e-4, label
