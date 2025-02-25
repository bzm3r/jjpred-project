"""Objects for defining the meta-information associated with an analysis,
in order to reproducibly run different analyses."""

from __future__ import annotations

import calendar
import polars as pl
from dataclasses import dataclass, field
import datetime
from functools import total_ordering
from typing import Self

from jjpred.inputstrategy import RefillType
from jjpred.scripting.dateoffset import (
    determine_main_program_compatible_start_end_dates,
)
from jjpred.seasons import Season
from jjpred.utils.datetime import (
    Date,
    DateLike,
    DateOffset,
    DateUnit,
    first_day,
)


@dataclass
class LatestDates:
    """Sometimes we may want to only consider historical data up to a particular
    month in a year.

    This data structure defines such time points."""

    sales_history_latest_date: Date
    """The latest date to use from historical data.

    For example, if historical data is available until ``2024-AUG-01``, but this
    variable is set to ``2024-MAR-01``, then historical data including and after
    ``2024-MAR-01`` will be ignored.

    Note that if a date like ``2024-AUG-15`` is given, then ``2024-AUG-01`` will
    be used as the value for this variable, because days are irrelevant."""
    demand_ratio_rolling_update_to: Date
    """The latest date to perform a rolling update to for demand (monthly)
    ratios."""

    def __init__(
        self,
        sales_history_latest_date: DateLike,
        demand_ratio_rolling_update_to: DateLike | None = None,
    ) -> None:
        self.sales_history_latest_date = first_day(sales_history_latest_date)
        if demand_ratio_rolling_update_to is not None:
            self.demand_ratio_rolling_update_to = first_day(
                Date.from_datelike(demand_ratio_rolling_update_to)
            )
        else:
            self.demand_ratio_rolling_update_to = (
                self.sales_history_latest_date
            )

    def latest(self) -> Date:
        if (
            self.demand_ratio_rolling_update_to
            < self.sales_history_latest_date
        ):
            return self.demand_ratio_rolling_update_to
        else:
            return self.sales_history_latest_date

    def __hash__(self) -> int:
        return hash(
            (
                self.sales_history_latest_date,
                self.demand_ratio_rolling_update_to,
            )
        )


def normalize_optional_datelike(date_like: DateLike | None) -> Date | None:
    if date_like is not None:
        return Date.from_datelike(date_like)
    else:
        return None


@total_ordering
@dataclass
class AnalysisDefn:
    """The ID of an analysis."""

    basic_descriptor: str = field(compare=False)
    """Name or short description."""

    date: Date = field(compare=False)
    """Date of analysis."""

    master_sku_date: Date = field(compare=False)
    """Date of associated Master SKU file."""

    sales_and_inventory_date: Date = field(compare=False)
    """Date of associated (historical) sales and current channel inventory
    file."""

    warehouse_inventory_date: Date = field(compare=False)
    """Date of associated warehouse inventory file."""

    config_date: Date | None = field(default=None, init=False, compare=False)
    """Date of associated marketed configuration file."""

    in_stock_ratio_date: Date | None = field(
        default=None, compare=False, init=False
    )
    """Date of associated in-stock ratio information file."""

    po_date: Date | None = field(default=None, compare=False, init=False)
    """Date of associated static PO data file."""

    latest_dates: LatestDates = field(init=False)
    """Latest dates for: default current period end dates and monthly ratio
    rolling update point."""

    extra_descriptor: str | None = field(default=None, init=False)
    """Extra description string."""

    _hash: int = field(init=False, compare=True)

    def __init__(
        self,
        basic_descriptor: str,
        date: DateLike,
        master_sku_date: DateLike,
        sales_and_inventory_date: DateLike,
        warehouse_inventory_date: DateLike,
        latest_dates: LatestDates | None = None,
        config_date: DateLike | None = None,
        in_stock_ratio_date: DateLike | None = None,
        po_date: DateLike | None = None,
        extra_descriptor: str | None = None,
    ):
        self.basic_descriptor = basic_descriptor
        self.date = Date.from_datelike(date)
        self.master_sku_date = Date.from_datelike(master_sku_date)
        self.sales_and_inventory_date = Date.from_datelike(
            sales_and_inventory_date
        )
        self.warehouse_inventory_date = Date.from_datelike(
            warehouse_inventory_date
        )

        self.config_date = normalize_optional_datelike(config_date)
        self.in_stock_ratio_date = normalize_optional_datelike(
            in_stock_ratio_date
        )
        self.po_date = normalize_optional_datelike(po_date)

        if latest_dates is None:
            self.latest_dates = LatestDates(self.date)
        else:
            self.latest_dates = latest_dates

        self.extra_descriptor = extra_descriptor

        self._hash = hash(
            (
                self.basic_descriptor,
                self.date,
                self.master_sku_date,
                self.sales_and_inventory_date,
                self.warehouse_inventory_date,
                self.config_date,
                self.in_stock_ratio_date,
                self.po_date,
                self.latest_dates,
                self.extra_descriptor,
            )
        )

    def __post_init__(self):
        self._hash = str(self).__hash__()

    def __str__(self) -> str:
        return f"{self.basic_descriptor}_analysis_date={str(self.date)}_{self.extra_descriptor}"

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return str(self) < str(other)

        return False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return str(self) == str(other)

        return False

    def get_field_if_available[T](
        self, field_name: str, field_type: type[T]
    ) -> T | None:
        field_value = None
        if hasattr(self, field_name):
            field_value = self.__getattribute__(field_name)
            if field_value is not None:
                assert isinstance(field_value, field_type)

        return field_value

    def get_website_sku_date(self) -> Date | None:
        return self.get_field_if_available("website_sku_date", Date)

    def tag_with_output_time(self) -> str:
        """Return a string identifying this analysis definition, along with a
        final part that states when this string was created."""
        output_date_time = datetime.datetime.now().strftime(r"%Y-%b-%d_%H%M%S")
        return self.tag() + f"_OUTPUT={output_date_time}"

    def tag(self) -> str:
        """Return a string identifying this analysis definition."""
        if self.extra_descriptor is not None:
            extra_descriptor = f"_{self.extra_descriptor}"
        else:
            extra_descriptor = ""
        return (
            f"{self.basic_descriptor}{extra_descriptor}_ANALYSIS={self.date}"
        )


@dataclass
class OutperformerSettings:
    """If this setting is active, outperforming SKUs have a long-term prediction
    made different from other SKUs."""

    active: bool
    """Whether outperformers should have a specialized long-term prediction."""

    prediction_offset: DateOffset = field(
        default_factory=lambda: DateOffset(3, DateUnit.MONTH)
    )
    """The offset applied to the dispatch date to mark the end of the prediction
    period."""

    active_months: dict[Season, list[int]] = field(
        default_factory=lambda: {
            Season.SS: [
                calendar.APRIL,
                calendar.MAY,
                calendar.JUNE,
                calendar.JULY,
                calendar.AUGUST,
            ],
            Season.FW: [
                calendar.OCTOBER,
                calendar.NOVEMBER,
                calendar.DECEMBER,
                calendar.JANUARY,
                calendar.FEBRUARY,
            ],
        }
    )
    """Specialized outperformer prediction is only performed for specific
    months based on the item's season. Otherwise, even if it is marked active,
    no long-term prediction is made.

    If an item is all-season, and no special definition is given for all-season
    items, then a combination of SS and FW definition is used.

    Note that currently this setting assumes we are working in the Northern Hemisphere."""


@dataclass
class ReservationInfo:
    polars_filter: pl.Expr | None
    reserve_to_date: Date

    def __init__(
        self, polars_filter: pl.Expr | None, reserve_to_date: DateLike
    ):
        self.polars_filter = polars_filter
        self.reserve_to_date = Date.from_datelike(reserve_to_date)


@dataclass
class RefillDefn(AnalysisDefn):
    """An analysis definition for review/refill of a channel. It is defined by a
    dispatch date (same as the prediction start date), and the date on which the
    analysis was conducted."""

    dispatch_date: Date
    """Date of dispatch. (Also the start of the prediction period.)

    It should be a Monday, but this check can be disabled by setting
    ``check_dispatch_date`` to ``False`` when initializing this class."""

    end_date: Date
    """Prediction period end date. The prediction period begins at the dispatch
    date and goes up to the end date."""

    dispatch_cutoff_qty: int
    """If a dispatch is calculated to be below the cutoff, no dispatch should be
    made."""

    prediction_type_meta_date: DateLike | None = field(compare=False)
    """Date of associated prediction type information file."""

    website_sku_date: Date | None = field(default=None, compare=False)
    """The website SKU file should be a table containing the list of SKUs that
    are sold on the website. It is used by the Master SKU reader to determine
    which SKUs listed in the Master SKU file are sold on the website."""

    jjweb_reserve_to_date: list[ReservationInfo] | None = field(
        default=None, compare=False
    )
    """The date up to which we will calculate reserved quantities based on J&J
    website PO predictions."""

    qty_box_date: Date | None = field(default=None, compare=False)
    """Date of the ``PO boxes and volume - All seasons`` Excel file to get the
    quantity per box information per category.

    If no date is given, then a file without a date will be looked for."""

    overperformer_settings: OutperformerSettings = field(
        default_factory=lambda: OutperformerSettings(False)
    )
    """If provided, outperformers will have a specialized long-term prediction
    done based based on the provided settings."""

    new_overrides_e: bool = field(default=True)
    """If ``True``, then categories marked new will use NE type prediction even
    if the category is supposed to use E type prediction."""

    enable_full_box_logic: bool = field(default=True)
    """Enable logic for rounding dispatches to the nearest full-box."""

    enable_low_current_period_isr_logic: bool = field(default=True)
    """Enable logic for using NE-type estimation for SKUs with low current
    perioda ISR."""

    warehouse_min_keep_qty: int = field(default=12)
    """Minimum quantity to keep in the warehouse for this dispatch."""

    def __init__(
        self,
        refill_description: str,
        analysis_date: DateLike,
        dispatch_date: DateLike,
        end_date: DateLike,
        master_sku_date: DateLike,
        sales_and_inventory_date: DateLike,
        warehouse_inventory_date: DateLike,
        config_date: DateLike,
        prediction_type_meta_date: DateLike | None,
        website_sku_date: DateLike | None = None,
        jjweb_reserve_to_date: list[ReservationInfo] | None = None,
        check_dispatch_date: bool = True,
        qty_box_date: DateLike | None = None,
        in_stock_ratio_date: DateLike | None = None,
        po_date: DateLike | None = None,
        outperformer_settings: OutperformerSettings = OutperformerSettings(
            False
        ),
        new_overrides_e: bool = True,
        enable_full_box_logic: bool = True,
        demand_ratio_rolling_update_to: DateLike | None = None,
        enable_low_current_period_isr_logic: bool = True,
        extra_descriptor: str | None = None,
        warehouse_min_keep_qty: int = 12,
        dispatch_cutoff_qty: int = 2,
    ):
        self.dispatch_date = Date.from_datelike(dispatch_date)
        self.end_date = Date.from_datelike(end_date)

        assert self.end_date > self.dispatch_date

        if (
            check_dispatch_date
            and calendar.day_name[self.dispatch_date.date.weekday()]
            != "Monday"
        ):
            raise ValueError(
                f"{self.dispatch_date.date} corresponds to weekday "
                f"{calendar.day_name[self.dispatch_date.date.weekday()]}"
                ", which is not a Monday. (To disable this check, set "
                "`check_dispatch_date=False`.)"
            )

        self.prediction_type_meta_date = (
            Date.from_datelike(prediction_type_meta_date)
            if prediction_type_meta_date is not None
            else None
        )

        self.overperformer_settings = outperformer_settings

        self.new_overrides_e = new_overrides_e

        self.enable_full_box_logic = enable_full_box_logic

        self.enable_low_current_period_isr_logic = (
            enable_low_current_period_isr_logic
        )

        if qty_box_date is not None:
            self.qty_box_date = Date.from_datelike(qty_box_date)

        self.warehouse_min_keep_qty = warehouse_min_keep_qty

        self.dispatch_cutoff_qty = dispatch_cutoff_qty

        self.website_sku_date = (
            Date.from_datelike(website_sku_date)
            if website_sku_date is not None
            else None
        )

        self.jjweb_reserve_to_date = jjweb_reserve_to_date

        super().__init__(
            refill_description,
            analysis_date,
            master_sku_date,
            sales_and_inventory_date,
            warehouse_inventory_date,
            config_date=config_date,
            in_stock_ratio_date=in_stock_ratio_date,
            po_date=po_date,
            latest_dates=LatestDates(
                self.dispatch_date, demand_ratio_rolling_update_to
            ),
            extra_descriptor=extra_descriptor,
        )

    def tag(self) -> str:
        return AnalysisDefn.tag(self) + f"_DISPATCH={self.dispatch_date}"

    def tag_with_output_time(self) -> str:
        """Return a string identifying this analysis definition, along with a
        final part that states when this string was created."""
        output_date_time = datetime.datetime.now().strftime(r"%Y-%b-%d_%H%M%S")
        return self.tag() + f"_OUTPUT={output_date_time}"


@dataclass
class FbaRevDefn(RefillDefn):
    """An analysis defn for an FBA review/refill. It is defined by a dispatch
    date (same as the prediction start date), and the date on which the
    analysis was conducted."""

    refill_type: RefillType = field(default=RefillType.WEEKLY, compare=False)
    """Type of FBA Refill to perform."""

    mon_sale_r_date: Date | None = field(default=None, compare=False)
    """Date of associated Historical sales data file (which contains the
    ``MonSaleR`` sheet.)"""

    mainprogram_date: Date | None = field(default=None, compare=False)
    """Date of associated Main Program Excel file (useful for comparing reuslts
    between main program and JJPRED Python program)."""

    refill_draft_date: Date | None = field(default=None, compare=False)
    """Date of associated refill draft plan file (useful for comparing results
    between main program and JJPRED Python program.)"""

    prediction_start_date_required_month_parts: int | None = field(
        default=None, compare=False
    )
    """Manual setting for number of month parts to use from dispatch date's
    month.

    This is sometimes necessary to make a closer comparison with the main
    program."""

    prediction_end_date_required_month_parts: int | None = field(
        default=None, compare=False
    )
    """Manual setting for number of month parts to use from the end date's month
    (determined by refill type).

    This is sometimes necessary to make a closer comparison with the main
    program."""

    match_main_program_month_fractions: bool = field(default=False)
    """Round month fractions to the nearest 25% multiple, in order to match how
    the main program calculates dates."""

    def __init__(
        self,
        analysis_date: DateLike,
        master_sku_date: DateLike,
        sales_and_inventory_date: DateLike,
        dispatch_date: DateLike,
        warehouse_inventory_date: DateLike,
        config_date: DateLike,
        prediction_type_meta_date: DateLike | None,
        refill_type: RefillType,
        check_dispatch_date: bool = True,
        website_sku_date: DateLike | None = None,
        jjweb_reserve_to_date: list[ReservationInfo] | None = None,
        qty_box_date: DateLike | None = None,
        mon_sale_r_date: DateLike | None = None,
        mainprogram_date: DateLike | None = None,
        refill_draft_date: DateLike | None = None,
        in_stock_ratio_date: DateLike | None = None,
        po_date: DateLike | None = None,
        outperformer_settings: OutperformerSettings = OutperformerSettings(
            False
        ),
        new_overrides_e: bool = True,
        enable_full_box_logic: bool = True,
        demand_ratio_rolling_update_to: DateLike | None = None,
        prediction_start_date_required_month_parts: int | None = None,
        prediction_end_date_required_month_parts: int | None = None,
        match_main_program_month_fractions: bool = False,
        enable_low_current_period_isr_logic: bool = True,
        warehouse_min_keep_qty: int = 12,
        dispatch_cutoff_qty: int = 2,
        extra_descriptor: str | None = None,
    ):
        self.refill_type = refill_type

        self.prediction_start_date_required_month_parts = (
            prediction_start_date_required_month_parts
        )
        self.prediction_end_date_required_month_parts = (
            prediction_end_date_required_month_parts
        )

        self.mon_sale_r_date = (
            Date.from_datelike(mon_sale_r_date)
            if mon_sale_r_date is not None
            else None
        )

        self.mainprogram_date = (
            Date.from_datelike(mainprogram_date)
            if mainprogram_date is not None
            else None
        )

        self.refill_draft_date = (
            Date.from_datelike(refill_draft_date)
            if refill_draft_date is not None
            else None
        )

        self.match_main_program_month_fractions = (
            match_main_program_month_fractions
        )

        # determine start/end dates that are compatible with the main program
        # TODO: this should be changed to just use the actual start/end dates,
        # once we are no longer comparing with the main program
        end_date = refill_type.end_date(dispatch_date)
        if self.match_main_program_month_fractions:
            dispatch_date, end_date = (
                determine_main_program_compatible_start_end_dates(
                    dispatch_date,
                    end_date,
                    start_date_required_month_parts=prediction_start_date_required_month_parts,
                    end_date_required_month_parts=prediction_end_date_required_month_parts,
                )
            )

        super().__init__(
            "fba_rev",
            analysis_date=analysis_date,
            dispatch_date=dispatch_date,
            end_date=end_date,
            master_sku_date=master_sku_date,
            sales_and_inventory_date=sales_and_inventory_date,
            warehouse_inventory_date=warehouse_inventory_date,
            config_date=config_date,
            prediction_type_meta_date=prediction_type_meta_date,
            website_sku_date=website_sku_date,
            jjweb_reserve_to_date=jjweb_reserve_to_date,
            check_dispatch_date=check_dispatch_date,
            qty_box_date=qty_box_date,
            in_stock_ratio_date=in_stock_ratio_date,
            po_date=po_date,
            outperformer_settings=outperformer_settings,
            new_overrides_e=new_overrides_e,
            enable_full_box_logic=enable_full_box_logic,
            demand_ratio_rolling_update_to=demand_ratio_rolling_update_to,
            enable_low_current_period_isr_logic=enable_low_current_period_isr_logic,
            extra_descriptor=extra_descriptor,
            warehouse_min_keep_qty=warehouse_min_keep_qty,
            dispatch_cutoff_qty=dispatch_cutoff_qty,
        )

    @classmethod
    def new_comparison_analysis(
        cls,
        analysis_date: DateLike,
        dispatch_date: DateLike,
        config_date: DateLike,
        prediction_type_meta_date: DateLike | None,
        real_analysis_date: DateLike,
        refill_type: RefillType,
        check_dispatch_date: bool = True,
        prediction_start_date_required_month_parts: int | None = None,
        prediction_end_date_required_month_parts: int | None = None,
        extra_descriptor: str | None = None,
        in_stock_ratio_date: DateLike | None = None,
        po_date: DateLike | None = None,
        outperformer_settings: OutperformerSettings = OutperformerSettings(
            False
        ),
        new_overrides_e: bool = True,
        demand_ratio_rolling_update_to: DateLike | None = None,
        enable_full_box_logic: bool = True,
        enable_low_current_period_isr_logic: bool = False,
        match_main_program_month_fractions: bool = True,
    ) -> Self:
        """Create an analysis definition for an FBA review that is meant to
        compare against a real analysis.

        Typically, the real analysis date corresponds with the date of files
        like such as master SKU file, sales and inventory file, warehouse
        inventory file, ``MonSaleR`` sheet file, main program file, and refill
        draft plan.

        If you need finer control over these dates, use the standard
        initialization method for the class, or use files which have date set to
        the real analysis date."""

        master_sku_date = real_analysis_date
        sales_and_inventory_date = real_analysis_date
        warehouse_inventory_date = real_analysis_date

        return cls(
            analysis_date,
            master_sku_date,
            sales_and_inventory_date,
            dispatch_date,
            warehouse_inventory_date,
            config_date,
            prediction_type_meta_date,
            refill_type,
            check_dispatch_date=check_dispatch_date,
            mon_sale_r_date=real_analysis_date,
            mainprogram_date=real_analysis_date,
            refill_draft_date=real_analysis_date,
            prediction_start_date_required_month_parts=prediction_start_date_required_month_parts,
            prediction_end_date_required_month_parts=prediction_end_date_required_month_parts,
            extra_descriptor=extra_descriptor,
            new_overrides_e=new_overrides_e,
            in_stock_ratio_date=in_stock_ratio_date,
            po_date=po_date,
            outperformer_settings=outperformer_settings,
            enable_full_box_logic=enable_full_box_logic,
            demand_ratio_rolling_update_to=demand_ratio_rolling_update_to,
            match_main_program_month_fractions=match_main_program_month_fractions,
            enable_low_current_period_isr_logic=enable_low_current_period_isr_logic,
        )

    def _get_date(self, date_name: str) -> Date:
        date = self.__dict__.get(date_name)
        if date is not None and isinstance(date, Date):
            return date
        else:
            raise ValueError(
                f"{self.__class__.__qualname__} has: {date_name}={date}"
            )

    def get_mon_sale_r_date(self) -> Date:
        return self._get_date("mon_sale_r_date")

    def get_mainprogram_date(self) -> Date:
        return self._get_date("mainprogram_date")

    def get_refill_draft_date(self) -> Date:
        return self._get_date("refill_draft_date")


@dataclass
class JJWebDefn(RefillDefn):
    website_proportions_split_date: Date = field(init=False)
    """Until we receive data that already includes splits, we will need to split
    janandjul.com Sales/PO data into."""

    def __init__(
        self,
        analysis_date: DateLike,
        dispatch_date: DateLike,
        end_date: DateLike,
        website_sku_date: DateLike,
        master_sku_date: DateLike,
        sales_and_inventory_date: DateLike,
        warehouse_inventory_date: DateLike,
        config_date: DateLike,
        prediction_type_meta_date: DateLike | None,
        proportion_split_date: DateLike,
        jjweb_reserve_to_date: list[ReservationInfo] | None = None,
        check_dispatch_date: bool = True,
        qty_box_date: DateLike | None = None,
        in_stock_ratio_date: DateLike | None = None,
        po_date: DateLike | None = None,
        outperformer_settings: OutperformerSettings = OutperformerSettings(
            False
        ),
        new_overrides_e: bool = True,
        enable_full_box_logic: bool = True,
        demand_ratio_rolling_update_to: DateLike | None = None,
        enable_low_current_period_isr_logic: bool = True,
        extra_descriptor: str | None = None,
    ):
        self.website_proportions_split_date = Date.from_datelike(
            proportion_split_date
        )

        super().__init__(
            "jjweb",
            analysis_date=analysis_date,
            dispatch_date=dispatch_date,
            end_date=end_date,
            master_sku_date=master_sku_date,
            sales_and_inventory_date=sales_and_inventory_date,
            warehouse_inventory_date=warehouse_inventory_date,
            config_date=config_date,
            prediction_type_meta_date=prediction_type_meta_date,
            website_sku_date=website_sku_date,
            jjweb_reserve_to_date=jjweb_reserve_to_date,
            check_dispatch_date=check_dispatch_date,
            qty_box_date=qty_box_date,
            in_stock_ratio_date=in_stock_ratio_date,
            po_date=po_date,
            outperformer_settings=outperformer_settings,
            new_overrides_e=new_overrides_e,
            enable_full_box_logic=enable_full_box_logic,
            demand_ratio_rolling_update_to=demand_ratio_rolling_update_to,
            enable_low_current_period_isr_logic=enable_low_current_period_isr_logic,
            extra_descriptor=extra_descriptor,
            warehouse_min_keep_qty=12,
            dispatch_cutoff_qty=0,
        )
