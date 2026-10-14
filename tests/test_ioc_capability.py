"""The IOC fitter needs a capability, not a compiled robotics problem."""
import numpy as np

from rei.equations import StationarityTermContribution, estimate_stationarity_weights


class Observation:
    n_total = 1

    def set_point(self, x):
        self.point = float(x[0])

    def required_list(self, required=None):
        return []

    def term_contributions(self, *, required=None):
        return [
            StationarityTermContribution(
                term_index=i, name=f"target_{i}", attrs={},
                gradient=np.array([self.point - target]),
                residual_norm=abs(self.point - target),
                weighted_residual_norm=abs(self.point - target),
            )
            for i, target in enumerate([0.0, 2.0])
        ]


def test_fit_without_runtime_or_backend():
    result = estimate_stationarity_weights(Observation(), p=[0.5])
    np.testing.assert_allclose(result["weights"], [0.75, 0.25], atol=1e-10)
    assert result["identifiability"]["unique_stationary_weights"]
    assert "backend" not in result
