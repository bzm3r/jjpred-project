"""Where different input strategies are defined for use in analyses."""

from __future__ import annotations
from calendar import Month

import polars as pl

from analysis_tools.utils import get_analysis_defn_and_db
from jjpred.analysisdefn import RefillDefn
from jjpred.channel import Channel
from jjpred.database import DataBase
from jjpred.inputstrategy import InputStrategy, SimpleTimePeriod

from jjpred.utils.datetime import Date


def get_default_current_period_dict(
    analysis_defn_or_db: RefillDefn | DataBase,
) -> dict[str, SimpleTimePeriod]:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
    assert isinstance(analysis_defn, RefillDefn)

    category_current_period_start_end = (
        db.meta_info.active_sku.select("category", "season")
        .unique()
        .with_columns(
            fw_start=pl.date(
                analysis_defn.current_seasons.FW + 2000,
                int(analysis_defn.fw_start_month),
                1,
            ),
            ss_start=pl.date(
                analysis_defn.current_seasons.SS + 2000,
                int(analysis_defn.ss_start_month),
                1,
            ),
        )
        .with_columns(
            start_date=pl.when(pl.col.season.eq("SS"))
            .then(pl.col.ss_start)
            .when(pl.col.season.eq("FW"))
            .then(pl.col.fw_start)
            .when(pl.col.season.eq("AS"))
            .then(pl.max_horizontal(pl.col.ss_start, pl.col.fw_start))
        )
        .with_columns(
            end_date=analysis_defn.latest_date.as_polars_date(),
        )
        .with_columns(
            start_date=pl.when(pl.col.end_date.lt(pl.col.start_date))
            .then(pl.col.start_date.dt.offset_by("-1y"))
            .otherwise(pl.col.start_date)
        )
    )

    current_period_dict = {
        str(x["category"]): SimpleTimePeriod(x["start_date"], x["end_date"])
        for x in category_current_period_start_end.to_dicts()
    }

    return current_period_dict


def get_last_year_as_current_period_dict(
    analysis_defn_or_db: RefillDefn | DataBase,
) -> dict[str, SimpleTimePeriod]:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
    assert isinstance(analysis_defn, RefillDefn)

    assert analysis_defn.dispatch_date >= Date.from_ymd(
        2025, Month.SEPTEMBER, 1
    )

    category_current_period_start_end = (
        db.meta_info.active_sku.select("category", "season")
        .unique()
        .with_columns(
            start_date=analysis_defn.date.as_polars_date()
            .dt.month_start()
            .dt.offset_by("-1y"),
        )
        .with_columns(
            end_date=pl.col.start_date.dt.offset_by("1y"),
        )
    )

    current_period_dict = {
        str(x["category"]): SimpleTimePeriod(x["start_date"], x["end_date"])
        for x in category_current_period_start_end.to_dicts()
    }

    return current_period_dict


def get_strategy_from_library(
    analysis_defn_or_db: RefillDefn | DataBase,
    current_period_overrides: dict[str, SimpleTimePeriod] | None = None,
) -> list[InputStrategy]:
    analysis_defn, db = get_analysis_defn_and_db(analysis_defn_or_db)
    assert isinstance(analysis_defn, RefillDefn)

    current_period_dict = (
        get_default_current_period_dict(db)
        if current_period_overrides is None
        else current_period_overrides
    )

    strategy = [
        InputStrategy(
            Channel.from_str(channel),
            analysis_defn.reference_categories,
            current_period_dict,
            analysis_defn.per_channel_reference_channels,
        )
        for channel in [
            "amazon.com",
            "amazon.ca",
            "janandjul.com",
            "jjweb ca east",
            "wholesale",
            "amazon.co.uk",
            "amazon.de",
        ]
    ]

    return strategy
