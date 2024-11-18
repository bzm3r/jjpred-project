import scipy.stats as sps
import numpy as np


def generate_demand_distribution(
    peak_month: float, kappa: float
) -> np.ndarray:
    """Generate a model demand distribution, centered at given ``peak_month``
    and with width controlled by ``kappa``."""
    n = 12
    scale = n / (2 * np.pi)

    distribution = sps.vonmises.cdf(
        (np.arange(n) + 1),
        kappa=kappa,
        loc=peak_month,
        scale=scale,
    ) - sps.vonmises.cdf(
        np.arange(n),
        kappa=kappa,
        loc=peak_month,
        scale=scale,
    )

    assert np.abs(np.sum(distribution) - 1.0) < 1e-16

    return distribution


def get_probability_within(
    distribution: np.ndarray, center: int, delta: int
) -> float:
    """Get probability within ``delta`` months of ``center``."""

    b = ((center - 1) - delta + 1) % 12
    sum = 0.0
    for n in range(2 * delta + 2):
        x = (n + b) % 12
        sum += distribution[x]

    return sum
