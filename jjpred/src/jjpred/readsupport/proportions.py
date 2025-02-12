"""Read information about J&J website sales ought to be proportioned between
CA/Not-CA and East/West."""

from __future__ import annotations

from pathlib import Path
from analysis_tools.utils import get_analysis_defn_and_db
from jjpred.analysisdefn import JJWebDefn
from jjpred.database import DataBase
import polars as pl

from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER
from jjpred.readsupport.utils import cast_standard
from jjpred.utils.datetime import Date, DateLike


def gen_jjweb_proportions_path(
    proportions_date: DateLike,
) -> Path:
    return Path(
        f"JJ_orders_proportion_{
            Date.from_datelike(proportions_date).format_as(r'%Y-%m-%d')
        }.csv"
    )


def read_jjweb_proportions(
    analysis_defn_or_db: JJWebDefn | DataBase,
) -> pl.DataFrame:
    """Read the CA/East sales proportions for the J&J website."""

    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)

    assert isinstance(analysis_defn, JJWebDefn)

    proportions_path = gen_jjweb_proportions_path(
        analysis_defn.website_proportions_split_date
    )

    raw_proportions = pl.read_csv(
        ANALYSIS_INPUT_FOLDER.joinpath(proportions_path)
    )

    rename_dict = dict(
        (x, x.split("_")[0].lower()) for x in raw_proportions.columns
    )

    proportions = cast_standard(
        [db.meta_info.all_sku],
        raw_proportions.rename(rename_dict),
    )

    return proportions
