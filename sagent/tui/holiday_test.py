"""Holiday dates against published ones, windows, overlap choice, and icons."""

from __future__ import annotations

from datetime import date, timedelta
from random import Random
from typing import TYPE_CHECKING, cast

import time

import pytest

from sagent.tui import holiday
from sagent.tui.holiday import Icon, holiday_icon, holidays, icon_rows


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


@pytest.fixture
def zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Set the process's local time zone, as a user's machine would have it."""

    def to(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield to
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def los_angeles(zone: Callable[[str], None]) -> None:
    zone("America/Los_Angeles")


# What US observers kept in 2026. Sources: timeanddate.com/holidays/us/2026
# (Lunar New Year, Holi, Nowruz, Eid al-Fitr, Easter, Eid al-Adha, seasons,
# Juneteenth, Thanksgiving), fiqhcouncil.org (Eid al-Fitr), un.org (Vesak),
# hebcal.com (Hanukkah begins at sundown Dec 4), almanac.com (fall Sep 22),
# and US temple calendars (Diwali Nov 8, Vaisakhi Apr 14, Mid-Autumn Sep 25).
US_2026 = {
    "New Year's": "01-01",
    "Lunar New Year": "02-17",
    "Valentine's Day": "02-14",
    "Mario Day": "03-10",
    "Pi Day": "03-14",
    "St. Patrick's Day": "03-17",
    "Holi": "03-03",
    "Nowruz": "03-20",
    "First day of spring": "03-20",
    "Easter": "04-05",
    "Vaisakhi": "04-14",
    "Eid al-Fitr": "03-20",
    "Vesak": "05-01",
    "Juneteenth": "06-19",
    "First day of summer": "06-21",
    "Independence Day": "07-04",
    "Eid al-Adha": "05-27",
    "First day of fall": "09-22",
    "Mid-Autumn Festival": "09-25",
    "Halloween": "10-31",
    "Diwali": "11-08",
    "Thanksgiving": "11-26",
    "Hanukkah": "12-04",
    "First day of winter": "12-21",
    "Christmas": "12-25",
    "Kwanzaa": "12-26",
}


@pytest.mark.usefixtures("los_angeles")
def test_each_holiday_starts_on_its_us_date_in_2026() -> None:
    assert {
        each.name: each.starts(2026)[0].strftime("%m-%d") for each in holidays()
    } == US_2026


# From the `holidays` package 0.107 (China, India, Israel), 2026-2035. Hanukkah
# is its 25 Kislev less a day: the first candle is lit the evening before.
PUBLISHED = {
    "lunar_new_year": "02-17 02-06 01-26 02-13 02-03 01-23 02-11 01-31 02-19 02-08",
    "vaisakhi": "04-14 04-14 04-13 04-14 04-14 04-14 04-13 04-14 04-14 04-14",
    "mid_autumn": "09-25 09-15 10-03 09-22 09-12 10-01 09-19 09-08 09-27 09-16",
    "diwali": "11-08 10-29 10-17 11-05 10-26 11-14 11-02 10-22 11-10 10-30",
    "hanukkah": "12-04 12-24 12-12 12-01 12-20 12-09 11-27 12-16 12-06 12-25",
}

# `ephem` 4.2 equinoxes and solstices, as local dates, 2026-2035.
LOCAL_SEASONS = {
    "America/Los_Angeles": {
        "spring": "03-20 03-20 03-19 03-20 03-20 03-20 03-19 03-20 03-20 03-20",
        "summer": "06-21 06-21 06-20 06-20 06-21 06-21 06-20 06-20 06-20 06-21",
        "fall": "09-22 09-22 09-22 09-22 09-22 09-22 09-22 09-22 09-22 09-22",
        "winter": "12-21 12-21 12-21 12-21 12-21 12-21 12-20 12-21 12-21 12-21",
    },
    "America/New_York": {
        "spring": "03-20 03-20 03-19 03-20 03-20 03-20 03-19 03-20 03-20 03-20",
        "summer": "06-21 06-21 06-20 06-20 06-21 06-21 06-20 06-20 06-21 06-21",
        "fall": "09-22 09-23 09-22 09-22 09-22 09-23 09-22 09-22 09-22 09-23",
        "winter": "12-21 12-21 12-21 12-21 12-21 12-21 12-21 12-21 12-21 12-21",
    },
}


def _first_starts(name: str, *, years: range) -> list[str]:
    starts = cast("Callable[[int], tuple[date, ...]]", getattr(holiday, name))
    return [starts(year)[0].strftime("%m-%d") for year in years]


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_computed_dates_match_published_dates(name: str) -> None:
    assert _first_starts(name, years=range(2026, 2036)) == PUBLISHED[name].split()


@pytest.mark.parametrize(
    ("zone_name", "season"),
    [(z, s) for z, seasons in LOCAL_SEASONS.items() for s in seasons],
)
def test_seasons_start_on_the_local_date(
    zone: Callable[[str], None],
    zone_name: str,
    season: str,
) -> None:
    zone(zone_name)
    expected = LOCAL_SEASONS[zone_name][season].split()
    assert _first_starts(season, years=range(2026, 2036)) == expected


@pytest.mark.parametrize("zone_name", ["America/New_York", "America/Los_Angeles"])
def test_vesak_is_the_un_day_of_vesak(
    zone: Callable[[str], None],
    zone_name: str,
) -> None:
    # timeanddate.com/holidays/un/vesak-day, read 2026-10-30. Buddhist councils
    # pick the full moon, so no zone's "first full moon of May" matches: 2015
    # was June 1.
    zone(zone_name)
    un = (
        "05-23 05-13 05-02 05-20 05-08 05-27 05-17 05-06 05-25 05-13 06-01 05-20"
        " 05-10 05-29 05-19 05-07 05-26 05-16 05-05 05-23 05-12 05-01 05-20 05-08"
        " 05-27 05-27"
    )
    assert _first_starts("vesak", years=range(2005, 2031)) == un.split()


def test_after_the_un_dates_vesak_is_mays_first_full_moon_in_utc() -> None:
    # Of every fixed zone, UTC agrees with the most UN years, 20 of 26.
    assert holiday.vesak(2031) == (date(2031, 5, 7),)


@pytest.mark.parametrize(
    ("zone_name", "year_2018"),
    [("America/New_York", "03-02"), ("America/Los_Angeles", "03-01")],
)
def test_holi_falls_on_the_users_own_date(
    zone: Callable[[str], None],
    zone_name: str,
    year_2018: str,
) -> None:
    # timeanddate.com/holidays/us/hindu-holi, 2008-2030.
    zone(zone_name)
    us = [
        "03-21",
        "03-11",
        "02-28",
        "03-19",
        "03-08",
        "03-27",
        "03-16",
        "03-05",
        "03-23",
        "03-12",
        "03-02",
        "03-21",
        "03-09",
        "03-28",
        "03-18",
        "03-07",
        "03-25",
        "03-14",
        "03-03",
        "03-22",
        "03-11",
        "02-28",
        "03-19",
    ]
    us[2018 - 2008] = year_2018
    assert _first_starts("holi", years=range(2008, 2031)) == us


def test_mid_autumn_follows_the_month_holding_the_equinox() -> None:
    # 2082: the new moon falls on the equinox's own day, so its month is the
    # eighth. 2033: China's calendar places a leap month that keeps it in
    # September, against the rule.
    assert holiday.mid_autumn(2082) == (date(2082, 10, 6),)
    assert holiday.mid_autumn(2033) == (date(2033, 9, 8),)


@pytest.mark.parametrize(
    ("k", "julian_day"),
    [
        # `ephem` 4.2, in universal time.
        (323, 2_461_089.000_75),
        (323.5, 2_461_102.984_60),
        (417, 2_463_864.069_16),
        (437.5, 2_464_469.684_44),
    ],
)
def test_moon_phases_land_within_half_a_minute(k: float, julian_day: float) -> None:
    jde = holiday._moon_phase(k)
    ut = jde - holiday._delta_t_sec(jde) / 86_400
    assert ut == pytest.approx(julian_day, abs=0.000_35)


@pytest.mark.parametrize(
    ("k", "julian_day"),
    [
        (-1200, 2_416_113.392_246_785),
        (0, 2_451_550.260_257_791_3),
        (323.5, 2_461_102.985_492_775),
        (1200, 2_486_987.127_214_118_4),
        (2400.5, 2_522_437.973_625_188),
        (-12_000, 2_097_182.914_411_113),
        (12000.5, 2_805_932.468_359_847_6),
    ],
)
def test_every_term_of_the_moon_phase_series_is_pinned(
    k: float,
    julian_day: float,
) -> None:
    assert holiday._moon_phase(k) == pytest.approx(julian_day, rel=0, abs=1e-9)


def test_the_moon_phase_series_reproduces_meeus_example_49a() -> None:
    # "Astronomical Algorithms" (1998), example 49.a: the new moon of
    # 1977 February, JDE 2443192.65118, with every periodic term.
    assert holiday._moon_phase(-283) == pytest.approx(2_443_192.651_18, abs=1e-5)


@pytest.mark.parametrize(
    ("year", "seconds"),
    [(2000.0, 63.86), (2010.0, 66.70), (2100.0, 202.74)],
)
def test_delta_t_follows_espenak_and_meeus(year: float, seconds: float) -> None:
    # eclipse.gsfc.nasa.gov/SEhelp/deltatpoly2004.html: one polynomial per era.
    jde = 2_451_545.0 + (year - 2000) * 365.25
    assert holiday._delta_t_sec(jde) == pytest.approx(seconds, abs=0.01)


def test_mid_autumn_2089_is_chinas_date() -> None:
    # The `holidays` package 0.107. The new moon begins 2089-09-04 23:57
    # China time; seven terms put it six minutes later, past midnight.
    assert holiday.mid_autumn(2089) == (date(2089, 9, 18),)


@pytest.mark.usefixtures("los_angeles")
def test_an_equinox_late_in_the_local_day_keeps_its_local_date() -> None:
    # 2044-03-19 23:23 UTC is 16:23 the same day in Los Angeles.
    assert holiday.spring(2044) == (date(2044, 3, 19),)


@pytest.mark.parametrize(
    ("year", "index", "julian_day"),
    [(2026, 0, 2_461_120.115_20), (2030, 3, 2_462_857.339_79)],
)
def test_seasons_land_within_minutes(year: int, index: int, julian_day: float) -> None:
    assert holiday._season(year, index) == pytest.approx(julian_day, abs=0.005)


def test_hanukkah_after_a_382_day_year_waits_a_day() -> None:
    # 5766 (2005) is the only such Hebrew year whose Tishrei falls in
    # 1990-2099; its first candle was lit at sundown on December 25.
    assert holiday.hanukkah(2005) == (date(2005, 12, 25),)


def test_easter_and_thanksgiving() -> None:
    assert holiday.easter(2026) == (date(2026, 4, 5),)
    assert holiday.easter(2027) == (date(2027, 3, 28),)
    assert holiday.easter(2038) == (date(2038, 4, 25),)
    assert holiday.thanksgiving(2026) == (date(2026, 11, 26),)
    assert holiday.thanksgiving(2027) == (date(2027, 11, 25),)


# fiqhcouncil.org/calendar, read 2026-10-30. Eid al-Fitr is 1 Shawwal; Eid
# al-Adha is the day after Saudi Arabia's Day of Arafah, 1 Dhul-Hijjah + 9.
SHAWWAL_1 = (
    "2026-03-20 2027-03-09 2028-02-26 2029-02-14 2030-02-04 2031-01-24 2032-01-14 "
    "2033-01-02 2033-12-23 2034-12-12 2035-12-01 2036-11-19 2037-11-08 2038-10-29 "
    "2039-10-19 2040-10-07 2041-09-26 2042-09-15 2043-09-04 2044-08-24 2045-08-14"
)
DHUL_HIJJAH_1 = (
    "2026-05-18 2027-05-07 2028-04-26 2029-04-15 2030-04-04 2031-03-24 2032-03-13 "
    "2033-03-02 2034-02-20 2035-02-09 2036-01-29 2037-01-17 2038-01-07 2038-12-27 "
    "2039-12-17 2040-12-05 2041-11-25 2042-11-14 2043-11-03 2044-10-22 2045-10-12"
)


def test_eids_follow_the_fiqh_council_of_north_america() -> None:
    fitr = [date.fromisoformat(d) for d in SHAWWAL_1.split()]
    adha = [date.fromisoformat(d) + timedelta(days=9) for d in DHUL_HIJJAH_1.split()]
    years = range(2026, 2046)
    assert [d for y in years for d in holiday.eid_al_fitr(y)] == fitr
    assert [d for y in years for d in holiday.eid_al_adha(y)] == adha


def test_the_arithmetic_calendar_takes_over_after_the_announced_dates() -> None:
    # It agrees with the council's 2026 date; past 2045 it is all there is.
    assert holiday._islamic(2026, month=10, day=1) == (date(2026, 3, 20),)
    assert holiday.eid_al_fitr(2046) == holiday._islamic(2046, month=10, day=1)
    assert holiday.eid_al_adha(2046) == holiday._islamic(2046, month=12, day=10)


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


def test_every_holiday_has_two_11_by_12_icons_with_a_color_for_every_pixel() -> None:
    for each in holidays():
        assert len(each.icons) == 2, each.name
        for icon in each.icons:
            assert len(icon.pixels) == 12, icon.name
            assert {len(row) for row in icon.pixels} == {11}, icon.name
            assert set("".join(icon.pixels)) - {"."} <= set(icon.palette), icon.name


def _luminance(color: str) -> float:
    """WCAG 2 relative luminance of ``#rrggbb``."""
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.039_28 else cast(float, ((c + 0.055) / 1.055) ** 2.4)
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _edge(icon: Icon) -> list[str]:
    """Return the colors of painted pixels that touch a clear pixel or the border."""
    rows = ["." * 13, *(f".{row}." for row in icon.pixels), "." * 13]
    return [
        icon.palette[rows[y][x]]
        for y in range(1, 13)
        for x in range(1, 12)
        if rows[y][x] != "."
        and "." in rows[y - 1][x] + rows[y + 1][x] + rows[y][x - 1] + rows[y][x + 1]
    ]


@pytest.mark.parametrize("background", ["#ffffff", "#000000"])
def test_every_icon_shows_on_light_and_dark_terminals(background: str) -> None:
    # A shape shows by its outline: half the pixels on it must differ from the
    # background by 1.5:1, or it dissolves, as a white ghost on a white page.
    dim: list[str] = []
    for each in holidays():
        for icon in each.icons:
            lums = [_luminance(c) for c in _edge(icon)]
            bg = _luminance(background)
            visible = [
                (max(lum, bg) + 0.05) / (min(lum, bg) + 0.05) >= 1.5 for lum in lums
            ]
            if sum(visible) < len(visible) / 2:
                dim.append(icon.name)
    assert dim == []


@pytest.mark.parametrize(
    ("day", "holiday_name"),
    [
        # Halloween 2026 is a Saturday: Friday opens the window.
        (date(2026, 10, 29), None),
        (date(2026, 10, 30), "Halloween"),
        (date(2026, 10, 31), "Halloween"),
        (date(2026, 11, 1), None),
        # Valentine's Day 2028 is a Monday: a workday, so only that day.
        (date(2028, 2, 11), None),
        (date(2028, 2, 13), None),
        (date(2028, 2, 14), "Valentine's Day"),
        (date(2028, 2, 15), None),
        # Hanukkah 2026 begins at sundown Friday, December 4, through the
        # eighth night, December 12.
        (date(2026, 12, 3), None),
        (date(2026, 12, 4), "Hanukkah"),
        (date(2026, 12, 12), "Hanukkah"),
        (date(2026, 12, 13), None),
        # Hanukkah 2031 begins at sundown Tuesday, December 9.
        (date(2031, 12, 9), "Hanukkah"),
    ],
)
def test_window_runs_from_the_last_workday_through_the_last_day(
    day: date,
    holiday_name: str | None,
) -> None:
    expected = set[str]() if holiday_name is None else _names_of(holiday_name)
    assert _icon_names(day, draws=20) == expected


@pytest.mark.usefixtures("los_angeles")
def test_overlapping_holidays_are_all_chosen() -> None:
    # March 20, 2026: spring begins, and with it Nowruz; Eid al-Fitr too.
    expected = (
        _names_of("First day of spring")
        | _names_of("Nowruz")
        | _names_of("Eid al-Fitr")
    )
    assert _icon_names(date(2026, 3, 20), draws=200) == expected


def test_kwanzaa_runs_into_the_new_year() -> None:
    expected = _names_of("Kwanzaa") | _names_of("New Year's")
    assert _icon_names(date(2027, 1, 1), draws=200) == expected


def test_an_asset_error_names_the_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "holidays.yaml").write_text("- name: a\n  name: b\n")

    def here(package: str) -> Path:
        del package
        return tmp_path

    monkeypatch.setattr(holiday, "files", here)
    holidays.cache_clear()
    try:
        with pytest.raises(ValueError, match=r"holidays\.yaml, line 2: the key 'name'"):
            holidays()
    finally:
        holidays.cache_clear()


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
