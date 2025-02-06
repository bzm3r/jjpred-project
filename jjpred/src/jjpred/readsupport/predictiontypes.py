"""Functions to read from the ``prediction_types.csv`` file.

This CSV has a format like:

.. code::

  season	month	prediction_type
      SS	  1	  PO
      FW	  1	  E
      AS	  1	  PO
      SS	  2	  PO
      FW	  2	  E
      AS	  2	  PO
      SS	  3	  PO
      FW	  3	  E
      AS	  3	  PO
      SS	  4	  CE
      FW	  4	  E
      AS	  4	  CE
"""

from __future__ import annotations

from pathlib import Path
import sys
from jjpred.analysisdefn import FbaRevDefn
from jjpred.datagroups import ALL_SKU_IDS
from jjpred.database import DataBase
from jjpred.predictiontypes import InputPredictionType, PredictionType
import polars as pl

from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.seasons import Season
from jjpred.skuinfo import get_all_sku_currentness_info
from jjpred.utils.datetime import Date, DateLike
from jjpred.utils.polars import find_dupes


PREDICTION_TYPES_SCHEMA: dict[str, pl.DataType] = {
    "season": Season.polars_type(),
    "dispatch_month": pl.Int8(),
    "prediction_type": InputPredictionType.polars_type(),
}
""":py:class:`pl.Schema` for the prediction type information dataframe."""


def gen_prediction_types_path(
    prediction_types_input_meta: str | DateLike | None,
) -> Path:
    if prediction_types_input_meta is None:
        meta = ""
    else:
        try:
            prediction_types_input_meta = Date.from_datelike(
                prediction_types_input_meta
            ).fmt_default()
        except ValueError:
            pass
        meta = f"_{prediction_types_input_meta}"

    return Path(f"prediction_types{meta}.csv")


def read_prediction_types(
    analysis_id_or_database: FbaRevDefn | DataBase,
    prediction_types_input_meta: str | DateLike | None,
) -> pl.DataFrame:
    """Read the prediction types CSV for a given analysis ID.

    This CSV has a format like::

    .. code::

      season	dispatch_month	prediction_type
          SS	  1	                PO
          FW	  1	                E
          AS	  1	                PO
          SS	  2	                PO
          FW	  2	                E
          AS	  2	                PO
          SS	  3	                PO
          FW	  3	                E
          AS	  3	                PO
          SS	  4	                CE
          FW	  4	                E
          AS	  4	                CE
    """

    prediction_types = (
        pl.read_csv(
            ANALYSIS_INPUT_FOLDER.joinpath(
                gen_prediction_types_path(prediction_types_input_meta)
            )
        )
        .cast(
            PREDICTION_TYPES_SCHEMA  # type: ignore
        )
        .cast({"prediction_type": PredictionType.polars_type()})
    )
    all_sku_info = (
        get_all_sku_currentness_info(analysis_id_or_database)
        .filter(pl.col.is_active)
        .join(prediction_types, on="season", how="left")
    )

    if len(all_sku_info.filter(pl.col.prediction_type.is_null())) > 0:
        sys.displayhook(all_sku_info.filter(pl.col.prediction_type.is_null()))

    find_dupes(
        all_sku_info, ALL_SKU_IDS + ["dispatch_month"], raise_error=True
    )

    return all_sku_info
