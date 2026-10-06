import numpy as np
import pytest

from housing_app.explain import shapley_values


def test_linear_model_gets_exact_weight_times_change():
    w = np.array([2.0, -1.0, 0.5])
    f = lambda X: X @ w + 3.0  # noqa: E731
    x, base = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0])
    np.testing.assert_allclose(shapley_values(f, x, base), w * (x - base), atol=1e-12)


def test_efficiency_and_dummy_properties_for_a_nonlinear_model():
    f = lambda X: np.tanh(X[:, 0] * X[:, 1]) + X[:, 1] ** 2  # noqa: E731   (feature 2 is unused)
    x, base = np.array([0.7, -1.2, 5.0]), np.array([0.1, 0.3, -2.0])
    phi = shapley_values(f, x, base)
    assert abs(phi.sum() - (f(x[None])[0] - f(base[None])[0])) < 1e-12
    assert abs(phi[2]) < 1e-12


def test_grouped_players_share_one_value():
    f = lambda X: X[:, 0] * X[:, 1] + X[:, 2]  # noqa: E731
    x, base = np.array([2.0, 3.0, 1.0]), np.zeros(3)
    phi = shapley_values(f, x, base, groups=[[0, 1], [2]])
    np.testing.assert_allclose(phi, [6.0, 1.0], atol=1e-12)


def test_too_many_players_are_refused():
    with pytest.raises(ValueError):
        shapley_values(lambda X: X.sum(axis=1), np.zeros(13), np.zeros(13))
