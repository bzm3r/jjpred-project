import polars as pl
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from math import ceil
from typing import NamedTuple


class GeneratedSales(NamedTuple):
    """Generated model sales data."""

    demand_factors: np.ndarray
    """Factors making up the demand distribution used to generate these sales."""
    growth_factors: np.ndarray
    """Factors making up the growth aspect used to generate these sales."""
    noise_factors: np.ndarray
    """Factors making up the noise aspect used to generate these sales"""
    demand_sales: np.ndarray
    """Sales due to demand distribution."""
    growth_sales: np.ndarray
    """Sales due to growth."""
    noise_sales: np.ndarray
    """Sales due to noise."""

    def as_df(self) -> pl.DataFrame:
        """Return a Polars dataframe representation of the generated sales."""
        series = []

        data_dict = self._asdict()
        for k, s in data_dict.items():
            series.append(pl.Series(k, s))

        return (
            pl.DataFrame(series)
            .with_row_index(name="month", offset=0)
            .with_columns(year=(pl.col.month / 12).floor().cast(pl.Int16()))
            .with_columns((pl.col.month % 12) + 1, min_sales=0.0)
            .with_columns(
                sales=pl.max_horizontal(
                    pl.col.demand_sales
                    + pl.col.growth_sales
                    + pl.col.noise_sales,
                    pl.col.min_sales,
                ),
            )
            .drop("min_sales")
            .with_columns(
                total_sales=pl.col.sales.sum().over("year"),
            )
            .with_columns(
                demand_ratio=pl.col.sales / pl.col.total_sales,
            )
            .select(
                ["year", "month"]
                + list(
                    x for x in data_dict.keys() if x not in ["demand_factors"]
                )
                + ["demand_factors", "demand_ratio", "sales", "total_sales"]
            )
        )


def extend_demand_distribution(
    demand_distribution: np.ndarray, len: int
) -> npt.NDArray[np.float64]:
    """Elongate or shorten the demand distribution to be ``len`` months long.

    If ``len`` is larger than 12, the demand distribution is repeated. Otherwise
    it is shortened.
    """
    if len > demand_distribution.shape[0]:
        tile_size = ceil(len / demand_distribution.shape[0])
        tiled = np.tile(demand_distribution, tile_size)
        return tiled[:len]
    else:
        return demand_distribution[:len]


@dataclass
class NoiseFactorOfSales:
    factor: float


@dataclass
class NoiseConstantValue:
    constant: float


@dataclass
class NoiseNone: ...


@dataclass
class GrowthConstantWithTime:
    factor: float


@dataclass
class GrowthConstantWithSalesQty:
    factor: float


def generate_sales(
    rng: np.random.Generator,
    peak_sales: int,
    n_months: int,
    demand_factors: npt.NDArray[np.float64],
    growth_factor: GrowthConstantWithTime | GrowthConstantWithSalesQty,
    noise: NoiseConstantValue | NoiseFactorOfSales | NoiseNone,
) -> GeneratedSales:
    expected_total_sales = peak_sales / float(np.max(demand_factors))

    demand_factors = extend_demand_distribution(demand_factors, n_months)
    print(f"{expected_total_sales=}")

    demand_sales = expected_total_sales * demand_factors
    print(f"{demand_sales=}")

    growth_factors = np.zeros(n_months, dtype=np.float64)
    if isinstance(growth_factor, GrowthConstantWithSalesQty):
        cumulative_demand_sales = np.cumsum(
            np.concatenate((np.zeros(shape=1), demand_sales), axis=0)
        )
        print(f"{cumulative_demand_sales=}")
        for ix in range(1, n_months):
            # growth_factors[ix] = (growth_factors[ix - 1] + 1) * (
            #     growth_factor + 1
            # ) - 1
            # growth_factors[ix] = (
            #     growth_factors[ix - 1] * growth_factor + 2 * growth_factor + 1 - 1
            # )
            # growth_factors[ix] = (
            #     growth_factors[ix - 1] * growth_factor + 2 * growth_factor
            # )
            growth_factors[ix] = (
                cumulative_demand_sales[ix] / 100
            ) * growth_factor.factor
    elif isinstance(growth_factor, GrowthConstantWithTime):
        for ix in range(1, n_months):
            growth_factors[ix] = (growth_factors[ix - 1] + 1) * (
                growth_factor.factor + 1
            ) - 1
            # growth_factors[ix] = (
            #     growth_factors[ix - 1] * growth_factor.factor
            #     + 2 * growth_factor.factor
            #     + 1
            #     - 1
            # )
            # growth_factors[ix] = (
            #     growth_factors[ix - 1] * growth_factor.factor
            #     + 2 * growth_factor.factor
            # )
        print(f"{growth_factors=}")

    print(f"{growth_factors=}")
    growth_sales = np.zeros(n_months, dtype=np.float64)
    for ix in range(1, n_months):
        growth_sales[ix] = demand_sales[ix] * growth_factors[ix]

    if isinstance(noise, NoiseFactorOfSales):
        noise_factors = np.zeros(n_months, dtype=np.float64)
        for ix in range(1, n_months):
            noise_factors[ix] = (2 * rng.random() - 1) * noise.factor

        noise_sales = np.zeros(n_months, dtype=np.float64)
        for ix in range(1, n_months):
            noise_sales[ix] = demand_sales[ix] * noise_factors[ix]
    elif isinstance(noise, NoiseConstantValue):
        noise_sales = np.zeros(n_months, dtype=np.float64)
        for ix in range(1, n_months):
            noise_sales[ix] = (2 * rng.random() - 1) * noise.constant

        noise_factors = noise_sales / demand_sales
    elif isinstance(noise, NoiseNone):
        noise_sales = np.zeros(n_months, dtype=np.float64)
        noise_factors = np.zeros(n_months, dtype=np.float64)

    return GeneratedSales(
        demand_factors,
        growth_factors,
        noise_factors,
        demand_sales,
        growth_sales,
        noise_sales,
    )
