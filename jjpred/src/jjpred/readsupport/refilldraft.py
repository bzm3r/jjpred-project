"""Functions to read from the refill draft dispatch plan Excel file."""

from __future__ import annotations

from enum import Enum, unique
from pathlib import Path
import polars as pl
import polars.selectors as cs

from jjpred.analysisdefn import FbaRevDefn
from jjpred.channel import Channel
from jjpred.countryflags import CountryFlags
from jjpred.globalvariables import DISPATCH_CUTOFF_QTY
from jjpred.readsupport.utils import (
    NA_FBA_SHEET,
    cast_standard,
    parse_channels,
)
from jjpred.readsupport.marketing import read_config
from jjpred.sku import Sku
from jjpred.structlike import MemberType
from jjpred.utils.datetime import Date
from jjpred.utils.fileio import (
    delete_or_read_df,
    gen_support_info_path,
    read_meta_info,
    write_df,
)
from jjpred.utils.polars import vstack_to_unified, sanitize_excel_extraction
from jjpred.utils.typ import normalize_as_list, as_polars_type
from jjpred.globalpaths import ANALYSIS_INPUT_FOLDER


@unique
class DispatchType(str, Enum):
    """:py:class:`Enum` enumerating relevant sheets from a refill draft plan."""

    ORIGINAL_DRAFT_FROM_MAIN_PROGRAM = NA_FBA_SHEET
    ORIGINAL_PLAN_FROM_REFILL_DRAFT = "Original_plan"
    FINAL_US = "US"
    FINAL_CA = "CA"


def read_raw_dispatch_from_excel(
    excel_path: Path,
    dispatch_type: DispatchType,
) -> pl.DataFrame:
    """Read an excel file containing dispatch information."""
    if (
        dispatch_type == DispatchType.FINAL_CA
        or dispatch_type == DispatchType.FINAL_US
    ):
        use_columns = ["SKU-Merchant", "Final Replenish Details in PCS"]
        rename_map = {
            "SKU-Merchant": "sku",
            "Final Replenish Details in PCS": "dispatch",
        }

        return sanitize_excel_extraction(
            pl.read_excel(
                excel_path,
                sheet_name=dispatch_type.value,
                read_options={"use_columns": use_columns},
            ),
        ).rename(rename_map)

    use_columns = []
    intermediate_names = []
    coerce_dtypes = []

    if dispatch_type == DispatchType.ORIGINAL_PLAN_FROM_REFILL_DRAFT:
        sheet_headers = pl.read_excel(
            excel_path,
            sheet_name=dispatch_type.value,
        ).columns
    else:
        sheet_headers = pl.read_excel(
            excel_path,
            sheet_name=dispatch_type.value,
            read_options={
                "header_row": 1,
                "n_rows": 1,
            },
        ).columns

    for col in sheet_headers:
        intermediate = None
        if col.strip().lower() == "flag":
            intermediate = "flag"
        else:
            for x in [
                "merchant",
                ["abnormal", "flag"],
                ["final", "replenish", "details"],
            ]:
                if all([y in col.lower() for y in normalize_as_list(x)]):
                    if x == "merchant":
                        intermediate = "sku"
                    elif "abnormal" in x:
                        intermediate = "abnormal_flag"
                    elif "final" in x:
                        if col.lower().endswith("_1"):
                            intermediate = "Amazon.ca"
                        else:
                            intermediate = "Amazon.com"

        if intermediate is not None:
            assert intermediate not in intermediate_names
            intermediate_names.append(intermediate)
            use_columns.append(col)
            coerce_dtypes.append("string")

    if dispatch_type == DispatchType.ORIGINAL_PLAN_FROM_REFILL_DRAFT:
        raw_df = pl.read_excel(
            excel_path,
            sheet_name=dispatch_type.value,
            read_options={"use_columns": use_columns},
        )
    else:
        raw_df = pl.read_excel(
            excel_path,
            sheet_name=NA_FBA_SHEET,
            read_options={
                "header_row": 1,
                "use_columns": use_columns,
            },
        )

    return (
        sanitize_excel_extraction(
            raw_df.rename(
                dict(zip(use_columns, intermediate_names, strict=True))
            )
        )
        .filter(pl.col("sku").is_not_null())
        .with_columns(
            pl.col("abnormal_flag").fill_null("NO_FLAG"),
            pl.col("flag").fill_null("NO_FLAG"),
        )
    )


def read_final_dispatch(
    analysis_defn: FbaRevDefn,
    target_channels: list[str | Channel] = ["amazon.com", "amazon.ca"],
) -> pl.DataFrame:
    """Read the dispatch from a ``REFILL DRAFT PLAN`` Excel file."""
    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")

    refill_draft_date = Date.from_datelike(
        analysis_defn.get_refill_draft_date()
    )
    refill_draft_path = ANALYSIS_INPUT_FOLDER.joinpath(
        f"REFILL DRAFT PLAN {refill_draft_date.fmt_flat()}.xlsx"
    )

    dispatch_df = pl.DataFrame()

    for channel in target_channels:
        ch = Channel.parse(channel)
        dispatch_type = (
            DispatchType.FINAL_US
            if ch.country_flag == CountryFlags.US
            else DispatchType.FINAL_CA
        )
        dispatch_df = vstack_to_unified(
            dispatch_df,
            cast_standard(
                [active_sku_info, channel_info],
                read_raw_dispatch_from_excel(
                    refill_draft_path,
                    dispatch_type,
                ).with_columns(**ch.to_columns()),
            ),
        )

    return dispatch_df


def create_dispatch_info(
    analysis_defn: FbaRevDefn,
    save_path: Path,
    original_dispatch: pl.DataFrame,
    final_dispatch: pl.DataFrame,
    overwrite: bool = True,
) -> pl.DataFrame:
    active_sku_info = read_meta_info(analysis_defn, "active_sku")
    channel_info = read_meta_info(analysis_defn, "channel")

    channel_dispatch_cols = ["Amazon.com", "Amazon.ca"]

    id_cols = [
        x
        for x in ["sku", "abnormal_flag", "flag"]
        if x in original_dispatch.columns
    ]

    # inventory flags (e.g. "XORO INV. LOWER THAN 12") are stored in the
    # dispatch columns. So we need to extract them out.
    inv_flags = parse_channels(
        original_dispatch.with_columns(
            pl.when(
                pl.col(x)
                .cast(pl.String())
                .str.strip_chars()
                .cast(pl.Int64(), strict=False)
                .is_null()
            )
            .then(pl.col(x))
            .otherwise(None)
            for x in channel_dispatch_cols
        )
        .unpivot(
            index=id_cols,
            variable_name="channel",
            value_name="inv_flag",
        )
        .with_columns(pl.col("inv_flag").fill_null("NO_FLAG")),
    ).drop("raw_channel")

    # inventory flags (e.g. "XORO INV. LOWER THAN 12") are stored in the same
    # column as the dispatch; in the last step we extracted these out. Now we
    # want to extract actual dispatch values out.
    dispatch_values = parse_channels(
        original_dispatch.with_columns(
            pl.when(
                ~(
                    pl.col(x)
                    .cast(pl.String())
                    .str.strip_chars()
                    .cast(pl.Int64(), strict=False)
                    .is_null()
                )
            )
            .then(pl.col(x))
            .otherwise(None)
            .cast(pl.Int64())
            .fill_null(0)
            for x in channel_dispatch_cols
        ).unpivot(
            index=id_cols,
            variable_name="channel",
            value_name="calc_dispatch",
        )
    ).drop("raw_channel")

    # channel_enum = pl.Enum(sorted(channel_dispatch_cols))
    # channel_map = parse_channels(
    #     pl.DataFrame(pl.Series("channel", channel_dispatch_cols))
    #     .with_columns(
    #         struct_channel=pl.col("channel").map_elements(
    #             Channel.map_polars,
    #             return_dtype=Channel.intermediate_polars_type_struct(),
    #         )
    #     )
    #     .unnest("struct_channel")
    # ).drop("raw_channel")

    # join the inventory flags and dispatch values into a combined table (each
    # has its own column now)
    original_dispatch = (
        # when a_sku in main program is adjusted to be the same as a_sku (or
        # vice versa), then this causes unique-ness issues that need to be resolved)
        inv_flags.unique()
        .join(
            dispatch_values.unique(),
            on=id_cols + ["channel"],
            how="left",
            validate="1:1",
            nulls_equal=True,
        )
        # .join(channel_map, on="channel", validate="m:1", nulls_equal=True)
        .drop("channel")
    )

    # check to see if there are SKUs in the actual dispatch results which we did
    # not read from the Master SKU file
    missing_from_active_sku = [
        x
        for x in original_dispatch["sku"]
        if x
        not in as_polars_type(active_sku_info["sku"].dtype, pl.Enum).categories
    ]
    assert (
        len(
            original_dispatch.filter(
                pl.col("sku").is_in(missing_from_active_sku)
            ).filter(pl.col("calc_dispatch").gt(0))
        )
        == 0
    )

    original_dispatch = cast_standard(
        [active_sku_info, channel_info],
        original_dispatch.filter(
            ~pl.col("sku").is_in(missing_from_active_sku)
        ),
    )

    non_sku_str_columns = original_dispatch.select(
        cs.exclude("sku", cs.exclude(cs.string())),
    ).columns
    config_data = read_config(analysis_defn)
    original_dispatch = cast_standard(
        [active_sku_info],
        original_dispatch.with_columns(
            flag=pl.when(
                pl.col("flag").is_not_null()
                & pl.col("flag").str.len_chars().eq(0)
            )
            .then(None)
            .otherwise(pl.col("flag"))
        ).with_columns(
            pl.col(x).cast(
                pl.Enum(original_dispatch[x].unique().drop_nulls().sort())
            )
            for x in non_sku_str_columns
        ),
    ).join(
        active_sku_info.select(
            Sku.members(MemberType.META)
            + [
                "pause_plan",
                "season",
                "sku_year_history",
                "category_year_history",
                "sku_latest_year",
            ]
        ),
        on=["sku"],
        how="left",
        # on the LHS, each SKU has multiple entries per channel
        validate="m:1",
        nulls_equal=True,
    )

    original_dispatch = original_dispatch.join(
        config_data.refill,
        on=Sku.members(MemberType.META) + Channel.members(),
        how="left",
        validate="1:1",
        nulls_equal=True,
    )

    dispatch_info = (
        original_dispatch.join(
            final_dispatch,
            how="left",
            on=["sku"] + Channel.members(),
            nulls_equal=True,
        )
        .rename({"dispatch": "final_dispatch"})
        .with_columns(pl.col.final_dispatch.fill_null(0))
        .with_columns(
            manually_deleted=(
                pl.col.calc_dispatch.ge(DISPATCH_CUTOFF_QTY)
                & pl.col.final_dispatch.eq(0)
            ),
        )
        .with_columns(
            manually_altered=(
                ~pl.col.manually_deleted
                & pl.col.calc_dispatch.ne(pl.col.final_dispatch)
            ),
        )
    )

    id_cols = cs.expand_selector(
        dispatch_info,
        cs.exclude(
            "refill_request",
            "manually_altered",
            "manually_deleted",
            "calc_dispatch",
            "final_dispatch",
            cs.contains("inv_flag"),
        ),
    )

    # for country in channel_map["country_flag"]:
    #     country_filter = pl.col("country_flag").eq(country)
    #     country_info = original_dispatch.filter(country_filter).join(
    #         original_dispatch.filter(~country_filter).select(
    #             cs.exclude([x for x in id_cols if x != "sku"])
    #         ),
    #         on="sku",
    #         how="left",
    #         validate="1:1",
    #         suffix="_other",
    #         nulls_equal=True,
    #     )
    #     # sys.displayhook(with_final_dispatch)
    #     dispatch_info = concat_to_unified(dispatch_info, country_info)

    # sys.displayhook(dispatch_info_df)
    dispatch_info = dispatch_info.with_columns(
        dispatch_info.select(
            cs.exclude("country_flag", cs.exclude(cs.numeric()))
        ).fill_null(0)
    ).select(
        Sku.members(MemberType.META)
        + Channel.members()
        + sorted(
            [
                x
                for x in cs.expand_selector(dispatch_info, cs.contains("flag"))
                if "country" not in x
            ]
        )
        + sorted(
            cs.expand_selector(dispatch_info, cs.contains("refill_request"))
        )
        + sorted(cs.expand_selector(dispatch_info, cs.contains("manually_")))
        + sorted(cs.expand_selector(dispatch_info, cs.contains("dispatch")))
    )

    write_df(overwrite, save_path, dispatch_info)
    return dispatch_info


def read_dispatch_from_refill_draft(
    analysis_defn: FbaRevDefn,
    read_from_disk: bool = False,
    delete_if_exists: bool = False,
    target_channels: list[str | Channel] = ["amazon.ca", "amazon.com"],
    overwrite: bool = True,
) -> pl.DataFrame:
    save_path = gen_support_info_path(
        analysis_defn,
        "refill_draft_dispatch",
        analysis_defn.get_refill_draft_date(),
        source_name="refill_draft_plan",
    )

    refill_draft_date = Date.from_datelike(
        analysis_defn.get_refill_draft_date()
    )
    refill_draft_path = ANALYSIS_INPUT_FOLDER.joinpath(
        f"REFILL DRAFT PLAN {refill_draft_date.fmt_flat()}.xlsx"
    )

    if read_from_disk or delete_if_exists:
        result = delete_or_read_df(delete_if_exists, save_path)
        if result is not None:
            return result

    original_plan = read_raw_dispatch_from_excel(
        refill_draft_path,
        DispatchType.ORIGINAL_PLAN_FROM_REFILL_DRAFT,
    ).with_columns(
        pl.col("abnormal_flag").fill_null("NO_FLAG"),
        pl.col("flag").fill_null("NO_FLAG"),
    )

    final_dispatch = read_final_dispatch(
        analysis_defn, target_channels=target_channels
    )

    return create_dispatch_info(
        analysis_defn,
        save_path,
        original_plan,
        final_dispatch,
        overwrite=overwrite,
    )
