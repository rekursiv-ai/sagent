"""Holiday dates against published ones, windows, overlap choice, and icons."""

from __future__ import annotations

from datetime import date
from random import Random
from typing import TYPE_CHECKING, cast

import pytest

from sagent.tui import holiday
from sagent.tui.holiday import Icon, holiday_icon, holidays, icon_rows


if TYPE_CHECKING:
    from collections.abc import Callable


# From the `holidays` package 0.107 (China, India, Iran, Israel, Singapore) and
# `ephem` 4.2 (UTC equinoxes and solstices), 2026-2035.
PUBLISHED = {
    "lunar_new_year": "02-17 02-06 01-26 02-13 02-03 01-23 02-11 01-31 02-19 02-08",
    "holi": "03-04 03-22 03-11 03-01 03-20 03-09 03-27 03-16 03-05 03-24",
    "nowruz": "03-21 03-21 03-20 03-20 03-21 03-21 03-20 03-20 03-21 03-21",
    "spring": "03-20 03-20 03-20 03-20 03-20 03-20 03-20 03-20 03-20 03-20",
    "passover": "04-02 04-22 04-11 03-31 04-18 04-08 03-27 04-14 04-04 04-24",
    "vaisakhi": "04-14 04-14 04-13 04-14 04-14 04-14 04-13 04-14 04-14 04-14",
    "summer": "06-21 06-21 06-20 06-21 06-21 06-21 06-20 06-21 06-21 06-21",
    "fall": "09-23 09-23 09-22 09-22 09-22 09-23 09-22 09-22 09-22 09-23",
    "mid_autumn": "09-25 09-15 10-03 09-22 09-12 10-01 09-19 09-08 09-27 09-16",
    "rosh_hashanah": "09-12 10-02 09-21 09-10 09-28 09-18 09-06 09-24 09-14 10-04",
    "diwali": "11-08 10-29 10-17 11-05 10-26 11-14 11-02 10-22 11-10 10-30",
    "hanukkah": "12-05 12-25 12-13 12-02 12-21 12-10 11-28 12-17 12-07 12-26",
    "winter": "12-21 12-22 12-21 12-21 12-21 12-22 12-21 12-21 12-21 12-22",
}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_computed_dates_match_published_dates(name: str) -> None:
    starts = cast("Callable[[int], tuple[date, ...]]", getattr(holiday, name))
    published = [
        date.fromisoformat(f"{year}-{month_day}")
        for year, month_day in zip(
            range(2026, 2036),
            PUBLISHED[name].split(),
            strict=True,
        )
    ]
    assert [starts(day.year)[0] for day in published] == published


@pytest.mark.parametrize(
    ("k", "julian_day"),
    [
        # `ephem` 4.2; the truncated series is good to about 4 minutes.
        (323, 2_461_089.000_75),
        (323.5, 2_461_102.984_60),
        (417, 2_463_864.069_16),
        (437.5, 2_464_469.684_44),
    ],
)
def test_moon_phases_land_within_minutes(k: float, julian_day: float) -> None:
    assert holiday._moon_phase(k) == pytest.approx(julian_day, abs=0.005)


def test_moon_phase_series_is_unchanged() -> None:
    # Meeus example 49.a gives 2443192.65118 with all 25 terms; seven land
    # 25 seconds away. Pinning the outputs guards each coefficient.
    phases = [holiday._moon_phase(k) for k in (-283, 323, 323.5, 417, 437.5)]
    assert phases == pytest.approx(
        [
            2_443_192.651_473_429,
            2_461_089.000_905_377_3,
            2_461_102.987_284_493_6,
            2_463_864.072_095_820_7,
            2_464_469.685_235_688,
        ],
        rel=0,
        abs=1e-7,
    )


def test_equinox_late_in_the_utc_day_keeps_its_utc_date() -> None:
    assert holiday.spring(2044) == (date(2044, 3, 19),)


@pytest.mark.parametrize(
    ("year", "index", "julian_day"),
    [(2026, 0, 2_461_120.115_20), (2030, 3, 2_462_857.339_79)],
)
def test_seasons_land_within_minutes(year: int, index: int, julian_day: float) -> None:
    assert holiday._season(year, index) == pytest.approx(julian_day, abs=0.005)


def test_rosh_hashanah_after_a_382_day_year_waits_a_day() -> None:
    # 2005 is the only such year in 1990-2099 (`holidays` 0.107 agrees).
    assert holiday.rosh_hashanah(2005) == (date(2005, 10, 4),)


def test_easter_and_thanksgiving() -> None:
    assert holiday.easter(2026) == (date(2026, 4, 5),)
    assert holiday.easter(2027) == (date(2027, 3, 28),)
    assert holiday.easter(2038) == (date(2038, 4, 25),)
    assert holiday.thanksgiving(2026) == (date(2026, 11, 26),)
    assert holiday.thanksgiving(2027) == (date(2027, 11, 25),)


def test_announced_dates_cover_both_eids_in_one_year() -> None:
    assert holiday.eid_al_fitr(2033) == (date(2033, 1, 3), date(2033, 12, 23))
    assert holiday.eid_al_adha(2026) == (date(2026, 5, 27),)
    assert holiday.vesak(2031) == (date(2031, 6, 4),)


def test_arithmetic_calendar_estimates_eid_outside_the_announced_dates() -> None:
    # One day from the announced 2026-03-21: announcements follow sightings.
    assert holiday._islamic(2026, month=10, day=1) == (date(2026, 3, 20),)
    assert holiday.eid_al_fitr(2036) == (date(2036, 11, 20),)
    assert holiday.eid_al_adha(2036) == (date(2036, 2, 8),)


def test_full_moon_estimates_vesak_outside_the_announced_dates() -> None:
    assert holiday.vesak(2036) == (date(2036, 5, 10),)


def _icon_names(day: date, *, draws: int = 1) -> set[str]:
    rng = Random(0)  # noqa: S311 -- Test draws, not cryptography.
    return {
        icon.name
        for _ in range(draws)
        if (icon := holiday_icon(day, rng=rng)) is not None
    }


def _names_of(holiday_name: str) -> set[str]:
    (match,) = (h for h in holidays() if h.name == holiday_name)
    return {icon.name for icon in match.icons}


def test_every_icon_is_11_by_12_with_a_color_for_every_pixel() -> None:
    assert len(holidays()) == 27
    for each in holidays():
        for icon in each.icons:
            assert len(icon.pixels) == 12, icon.name
            assert {len(row) for row in icon.pixels} == {11}, icon.name
            assert set("".join(icon.pixels)) - {"."} <= set(icon.palette), icon.name


@pytest.mark.parametrize(
    ("day", "holiday_name"),
    [
        # Halloween 2026 is a Saturday: Friday opens the window.
        (date(2026, 10, 29), None),
        (date(2026, 10, 30), "Halloween"),
        (date(2026, 10, 31), "Halloween"),
        (date(2026, 11, 1), None),
        # Valentine's Day 2028 is a Monday: the window spans the weekend.
        (date(2028, 2, 10), None),
        (date(2028, 2, 11), "Valentine's Day"),
        (date(2028, 2, 12), "Valentine's Day"),
        (date(2028, 2, 14), "Valentine's Day"),
        # Hanukkah 2026 lasts eight days from December 5.
        (date(2026, 12, 12), "Hanukkah"),
        (date(2026, 12, 13), None),
    ],
)
def test_window_runs_from_the_weekday_before_through_the_last_day(
    day: date,
    holiday_name: str | None,
) -> None:
    expected = set[str]() if holiday_name is None else _names_of(holiday_name)
    assert _icon_names(day, draws=20) == expected


def test_overlapping_holidays_are_all_chosen() -> None:
    # March 20, 2026: spring begins; Nowruz and Eid al-Fitr fall on the 21st.
    expected = (
        _names_of("First day of spring")
        | _names_of("Nowruz")
        | _names_of("Eid al-Fitr")
    )
    assert _icon_names(date(2026, 3, 20), draws=200) == expected


def test_kwanzaa_runs_into_the_new_year() -> None:
    expected = _names_of("Kwanzaa") | _names_of("New Year's")
    assert _icon_names(date(2027, 1, 1), draws=200) == expected


def test_icon_rows_pack_two_pixels_per_cell() -> None:
    icon = Icon(
        name="test",
        palette={"R": "#ff0000", "B": "#0000ff"},
        pixels=("R.RR.", ".BRB."),
    )
    (row,) = icon_rows(icon)
    assert row.plain == "▀▄█▀ "
    styles = [str(span.style) for span in row.spans]
    assert styles == ["#ff0000", "#0000ff", "#ff0000", "#ff0000 on #0000ff"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
