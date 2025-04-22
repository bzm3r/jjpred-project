"""Helpers for offsetting dates so that they match up with how the main program
Excel file divides up months.

The Excel file divides the month up into 4 parts.

So, for example, if we the prediction starting date is Sep. 8, we will want 3
parts of the September, and will have to shift prediction date we start from in
order to get (approximately) 75% of September.

Similar considerations exist for the ending date: using the Sep. 8 example, we
will want 1 week out of September, i.e. 25% of the days out of September.
"""

from __future__ import annotations

from calendar import monthrange
from math import ceil
from jjpred.utils.datetime import (
    Date,
    DateLike,
    DateOffset,
    DateUnit,
    first_day_next_month,
    first_day,
    offset_date,
)


def shift_date(
    actual_date: DateLike,
    is_start_date: bool,
    required_month_parts: int | None = None,
) -> Date:
    print(f"shifting: {actual_date}, {is_start_date=}")
    actual_date = Date.from_datelike(actual_date)

    _, days_in_month = monthrange(actual_date.year, actual_date.month)

    if is_start_date:
        actual_days = (first_day_next_month(actual_date) - actual_date).days
    else:
        actual_days = (actual_date - first_day(actual_date)).days

    if required_month_parts is None:
        actual_weeks = actual_days / 7.0
        month_parts = round(actual_weeks)
        print(f"{actual_days=}")
        print(f"{actual_weeks=}")
        print(f"number of month parts: {month_parts}")
    else:
        month_parts = required_month_parts

    if required_month_parts is not None:
        required_days = ceil((month_parts / 4) * days_in_month)
    else:
        required_days = round((month_parts / 4) * days_in_month)

    if required_days > days_in_month:
        required_days = days_in_month

    print()

    print(f"number of days from month before date shift: {actual_days}")
    print(f"required number of days: {required_days}")
    if is_start_date:
        correction = actual_days - required_days
    else:
        correction = required_days - actual_days
    print(f"correction: {correction}")

    shifted_date = offset_date(
        actual_date,
        DateOffset(correction, DateUnit.DAY),
    )

    if is_start_date:
        num_days_after_shift = (
            first_day_next_month(shifted_date) - shifted_date
        ).days
    else:
        num_days_after_shift = (
            shifted_date - first_day(shifted_date)
        ).days + 1

    print(
        f"number of days from month after date shift: {num_days_after_shift}"
    )
    print()

    return shifted_date


def determine_main_program_compatible_start_end_dates(
    actual_start_date: DateLike,
    actual_end_date: DateLike,
    start_date_required_month_parts: int | None = None,
    end_date_required_month_parts: int | None = None,
) -> tuple[Date, Date]:
    """Determine start/end dates, shifted in order to match up the how the main
    Excel program divides up a month in order to predict demand."""
    actual_start_date = Date.from_datelike(actual_start_date)
    print(f"{actual_start_date=}")

    actual_end_date = Date.from_datelike(actual_end_date)
    print(f"{actual_end_date=}")

    start_date = shift_date(
        actual_start_date,
        True,
        required_month_parts=start_date_required_month_parts,
    )
    end_date = shift_date(
        actual_end_date,
        False,
        required_month_parts=end_date_required_month_parts,
    )
    print(f"{start_date=}")
    print(f"{end_date=}")

    return start_date, end_date
