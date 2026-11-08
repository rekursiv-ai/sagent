"""Holiday dates against published ones, windows, overlap choice, and icons."""

from __future__ import annotations

from datetime import date, timedelta
from random import Random
from typing import TYPE_CHECKING, Literal, cast

import itertools
import time

import pytest

from sagent.lib.userdirs import base_dir_variables
from sagent.tui import holiday
from sagent.tui.holiday import Icon, holidays, icon_rows, todays_holiday


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path


@pytest.fixture
def zone(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Set the process's local time zone, as a user's machine would have it."""
    isolated = base_dir_variables()
    try:
        with monkeypatch.context() as local:

            def to(name: str) -> None:
                local.setenv("TZ", name)
                time.tzset()

            yield to
    finally:
        time.tzset()
        assert base_dir_variables() == isolated


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
    "spring": "03-20",
    "Easter": "04-05",
    "Vaisakhi": "04-14",
    "Eid al-Fitr": "03-20",
    "Vesak": "05-01",
    "Juneteenth": "06-19",
    "summer": "06-21",
    "Independence Day": "07-04",
    "Eid al-Adha": "05-27",
    "fall": "09-22",
    "Mid-Autumn Festival": "09-25",
    "Halloween": "10-31",
    "Diwali": "11-08",
    "Thanksgiving": "11-26",
    "Hanukkah": "12-04",
    "winter": "12-21",
    "Christmas": "12-25",
    "Kwanzaa": "12-26",
}


@pytest.mark.usefixtures("los_angeles")
def test_each_holiday_falls_on_its_us_date_in_2026() -> None:
    assert {
        each.name: each.dates(2026)[0].strftime("%m-%d") for each in holidays()
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


def _first_dates(name: str, *, years: range) -> list[str]:
    dates = cast("Callable[[int], tuple[date, ...]]", getattr(holiday, name))
    return [dates(year)[0].strftime("%m-%d") for year in years]


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_computed_dates_match_published_dates(name: str) -> None:
    assert _first_dates(name, years=range(2026, 2036)) == PUBLISHED[name].split()


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
    assert _first_dates(season, years=range(2026, 2036)) == expected


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
    assert _first_dates("vesak", years=range(2005, 2031)) == un.split()


def test_after_the_un_dates_vesak_is_mays_first_full_moon_in_utc() -> None:
    # Of every fixed zone, UTC agrees with the most UN years, 20 of 26.
    assert holiday.vesak(2031) == (date(2031, 5, 7),)
    # `ephem` 4.2: full moons at 2045-05-01 05:52 UTC, and 2075-05-30 02:38
    # after one on April 30.
    assert holiday.vesak(2045) == (date(2045, 5, 1),)
    assert holiday.vesak(2075) == (date(2075, 5, 30),)


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
    assert _first_dates("holi", years=range(2008, 2031)) == us


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


@pytest.mark.parametrize("zone_name", ["America/Los_Angeles", "Pacific/Kiritimati"])
def test_mid_autumn_is_chinas_wherever_the_user_is(
    zone: Callable[[str], None],
    zone_name: str,
) -> None:
    zone(zone_name)
    assert _first_dates("mid_autumn", years=range(2026, 2036)) == (
        PUBLISHED["mid_autumn"].split()
    )
    assert holiday.mid_autumn(2082) == (date(2082, 10, 6),)
    # `lunardate` 0.2: here the equinox falls a day before China's.
    assert holiday.mid_autumn(1938) == (date(1938, 10, 8),)


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
        found[1].name
        for _ in range(draws)
        if (found := todays_holiday(day, rng=rng)) is not None
    }


def _holiday_icons(holiday_name: str) -> set[str]:
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
        # Halloween 2026 is a Saturday: its weekend, Friday to Sunday.
        (date(2026, 10, 29), None),
        (date(2026, 10, 30), "Halloween"),
        (date(2026, 10, 31), "Halloween"),
        (date(2026, 11, 1), "Halloween"),
        (date(2026, 11, 2), None),
        # Easter is seen at work on Monday, not on Good Friday: Saturday to Monday.
        (date(2026, 4, 3), None),
        (date(2026, 4, 4), "Easter"),
        (date(2026, 4, 5), "Easter"),
        (date(2026, 4, 6), "Easter"),
        (date(2026, 4, 7), None),
        # Valentine's Day 2028 is a Monday: a workday, so only that day.
        (date(2028, 2, 11), None),
        (date(2028, 2, 13), None),
        (date(2028, 2, 14), "Valentine's Day"),
        (date(2028, 2, 15), None),
        # Hanukkah 2026 begins at sundown Friday, December 4: that day only.
        (date(2026, 12, 3), None),
        (date(2026, 12, 4), "Hanukkah"),
        (date(2026, 12, 5), None),
        # Kwanzaa begins Saturday, December 26, 2026, and is gone by New Year's.
        (date(2026, 12, 27), "Kwanzaa"),
        (date(2026, 12, 28), None),
        (date(2027, 1, 1), "New Year's"),
    ],
)
def test_window_is_the_day_or_its_whole_weekend(
    day: date,
    holiday_name: str | None,
) -> None:
    expected = set[str]() if holiday_name is None else _holiday_icons(holiday_name)
    assert _icon_names(day, draws=20) == expected


@pytest.mark.usefixtures("los_angeles")
def test_overlapping_holidays_are_all_chosen(monkeypatch: pytest.MonkeyPatch) -> None:
    # March 20, 2026: spring begins, and with it Nowruz; Eid al-Fitr too.
    expected = (
        _holiday_icons("spring")
        | _holiday_icons("Nowruz")
        | _holiday_icons("Eid al-Fitr")
    )
    choices = iter(
        index for pair in itertools.product(range(3), range(2)) for index in pair
    )
    sizes: list[int] = []

    def choose[T](options: Sequence[T]) -> T:
        sizes.append(len(options))
        return options[next(choices)]

    rng = Random(0)  # noqa: S311 -- Test draws, not cryptography.
    monkeypatch.setattr(rng, "choice", choose)
    found = [todays_holiday(date(2026, 3, 20), rng=rng) for _ in range(6)]
    pairs = [pair for pair in found if pair is not None]
    assert {icon.name for _, icon in pairs} == expected
    assert sizes == [3, 2] * 6
    # Each icon comes with the name of the holiday it belongs to.
    assert all(icon.name in _holiday_icons(name) for name, icon in pairs)


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


@pytest.fixture(autouse=True)
def _fresh_holidays() -> Iterator[None]:
    """Read the asset each test, so no test sees rules another test cached."""
    holidays.cache_clear()
    yield
    holidays.cache_clear()


def test_diwali_is_the_first_evening_of_the_kartika_new_moon_s_lunar_day() -> None:
    # The `holidays` package 0.107, India, 2001-2035. In 2024 the new moon came at
    # 18:17 India time on November 1, but its lunar day began at 15:54 on October
    # 31, so October 31's evening was the first within it.
    published = (
        "11-14 11-04 10-25 11-12 11-01 10-21 11-09 10-28 10-17 11-05 10-26 11-13 11-03"
        " 10-23 11-11 10-30 10-19 11-07 10-27 11-14 11-04 10-24 11-12 10-31 10-20 11-08"
        " 10-29 10-17 11-05 10-26 11-14 11-02 10-22 11-10 10-30"
    )
    assert _first_dates("diwali", years=range(2001, 2036)) == published.split()


def test_the_2024_lunar_day_begins_when_ephem_says() -> None:
    # `ephem` 4.2: the moon came within 12 degrees of the sun at JD 2460614.933378
    # UT, 15:54 India time; over 1990-2100 every Kartika start lands within 2 min.
    jde = holiday._tithi_start(holiday._moon_phase(307))
    ut = jde - holiday._delta_t_sec(jde) / 86_400
    assert ut == pytest.approx(2_460_614.933_378, abs=3 / 1440)
    # Solved to the root: a ten-millionth of a degree is under a second.
    assert jde == pytest.approx(2_460_614.934_618_226_3, rel=0, abs=1e-7)


@pytest.mark.parametrize(
    ("ut", "degrees"),
    [
        # `ephem` 4.2, in universal time.
        (2_451_545.0, 302.949_04),
        (2_460_615.5, 354.166_73),
        (2_470_000.25, 287.115_57),
        (2_480_000.75, 157.511_26),
    ],
)
def test_the_moon_s_lead_on_the_sun_is_within_a_hundredth_of_a_degree(
    ut: float,
    degrees: float,
) -> None:
    jde = ut + holiday._delta_t_sec(ut) / 86_400
    assert holiday._elongation(jde) == pytest.approx(degrees, abs=0.01)


@pytest.mark.parametrize(
    ("jde", "degrees"),
    [
        (2_451_545.0, 302.937_369_408_123_86),
        (2_460_615.5, 354.149_671_989_362_1),
        (2_415_020.0, 345.655_320_807_709_36),
        (2_488_070.0, 243.307_634_797_587_53),
        (2_469_807.5, 97.919_435_445_277_49),
    ],
)
def test_every_term_of_the_elongation_series_is_pinned(
    jde: float,
    degrees: float,
) -> None:
    assert holiday._elongation(jde) == pytest.approx(degrees, rel=0, abs=1e-9)


def test_india_keeps_utc_plus_five_thirty() -> None:
    assert holiday._india().utcoffset(None) == timedelta(hours=5, minutes=30)


# Computed, pinned: the years where a slip in the Lahiri ayanamsa's epoch or
# drift first moves the sun's entry into sidereal Aries across midnight.
@pytest.mark.parametrize(
    ("year", "day"),
    [(1950, date(1950, 4, 13)), (2075, date(2075, 4, 14)), (2114, date(2114, 4, 16))],
)
def test_vaisakhi_follows_the_ayanamsa_s_drift(year: int, day: date) -> None:
    assert holiday.vaisakhi(year) == (day,)


def test_lunar_new_year_matches_china_each_year_from_1990_to_2099() -> None:
    # The `holidays` package 0.107, China. Each year from 1990, and from 2008.
    published = (
        "01-27 02-15 02-04 01-23 02-10 01-31 02-19 02-07 01-28 02-16 02-05 01-24 02-12"
        " 02-01 01-22 02-09 01-29 02-18 02-07 01-26 02-14 02-03 01-23 02-10 01-31 02-19"
        " 02-08 01-28 02-16 02-05 01-25 02-12 02-01 01-22 02-10 01-29 02-17 02-06 01-26"
        " 02-13 02-03 01-23 02-11 01-31 02-19 02-08 01-28 02-15 02-04 01-24 02-12 02-01"
        " 01-22 02-10 01-30 02-17 02-06 01-26 02-14 02-02 01-23 02-11 02-01 02-19 02-08"
        " 01-28 02-15 02-04 01-24 02-12 02-02 01-21 02-09 01-29 02-17 02-05 01-26 02-14"
        " 02-03 01-23 02-11 01-31 02-19 02-07 01-27 02-15 02-05 01-24 02-12 02-02 01-22"
        " 02-09 01-29 02-17 02-06 01-26 02-14 02-03 01-24 02-10 01-30 02-18 02-07 01-27"
        " 02-15 02-05 01-25 02-12 02-01 01-21"
    )
    assert _first_dates("lunar_new_year", years=range(1990, 2100)) == published.split()


# Dateutil 2.9's `easter`: a year from each place a computus constant shows.
@pytest.mark.parametrize(
    "easter_day",
    [date(1590, 4, 22), date(1700, 4, 11), date(1707, 4, 24), date(3165, 4, 18)],
)
def test_easter_matches_an_independent_computus(easter_day: date) -> None:
    assert holiday.easter(easter_day.year) == (easter_day,)


def test_thanksgiving_is_a_thursday_in_its_fourth_week_every_year() -> None:
    days = [holiday.thanksgiving(year)[0] for year in range(1950, 2050)]
    assert {(d.weekday(), 22 <= d.day <= 28) for d in days} == {(3, True)}


# Rosh Hashanah, hebcal.com; a year never begins on a Sunday, Wednesday or Friday.
@pytest.mark.parametrize(
    ("hebrew_year", "new_year"),
    [
        (5784, date(2023, 9, 16)),
        (5785, date(2024, 10, 3)),
        (5786, date(2025, 9, 23)),
        (5787, date(2026, 9, 12)),
        (5788, date(2027, 10, 2)),
    ],
)
def test_the_hebrew_new_year_is_rosh_hashanah(hebrew_year: int, new_year: date) -> None:
    assert date.fromordinal(holiday._hebrew_new_year(hebrew_year)) == new_year


def test_every_hebrew_year_has_a_lawful_length_and_first_day() -> None:
    starts = [holiday._hebrew_new_year(y) for y in range(5700, 5900)]
    lengths = {b - a for a, b in itertools.pairwise(starts)}
    assert lengths == {353, 354, 355, 383, 384, 385}
    assert {date.fromordinal(s).weekday() for s in starts} == {0, 1, 3, 5}


def test_the_arithmetic_islamic_years_leap_on_the_standard_eleven_of_thirty() -> None:
    # Reingold and Dershowitz, "Calendrical Calculations": years 2, 5, 7, 10, 13,
    # 16, 18, 21, 24, 26, 29 of each cycle have 355 days. 2019's 1 Muharram began
    # 1441, year 1 of its cycle.
    starts = sorted(
        d for year in range(2019, 2050) for d in holiday._islamic(year, month=1, day=1)
    )
    leap = [
        n + 1
        for n, (a, b) in enumerate(itertools.pairwise(starts))
        if b - a == timedelta(355)
    ]
    assert leap == [2, 5, 7, 10, 13, 16, 18, 21, 24, 26, 29]


@pytest.mark.parametrize(
    ("month", "day", "first"),
    [
        (10, 1, date(2020, 5, 24)),
        (12, 10, date(2020, 7, 31)),
        (9, 1, date(2020, 4, 24)),
    ],
)
def test_each_islamic_month_starts_after_those_before_it(
    month: int,
    day: int,
    first: date,
) -> None:
    # 1441 AH as kept, which the tabular calendar matches: Eid al-Fitr, Eid
    # al-Adha, and the first of Ramadan.
    assert holiday._islamic(2020, month=month, day=day) == (first,)


@pytest.mark.parametrize(
    ("year", "dates"),
    [
        (2026, (date(2026, 1, 1),)),
        (2027, (date(2027, 2, 2),)),
        (2028, (date(2028, 3, 3),)),
        (2025, ()),
        (2029, ()),
    ],
)
def test_listed_dates_hold_from_the_first_year_to_the_last(
    year: int,
    dates: tuple[date, ...],
) -> None:
    listed = ("2026-01-01", "2027-02-02", "2028-03-03")
    assert holiday._listed(year, dates=listed, otherwise=lambda _: ()) == dates


@pytest.mark.parametrize(
    ("day", "workday", "today"),
    [
        # Sunday, January 2 reaches back to Friday, December 31.
        (date(2028, 1, 2), "friday", date(2027, 12, 31)),
        # Saturday, December 30 reaches on to Monday, January 1.
        (date(2028, 12, 30), "monday", date(2029, 1, 1)),
    ],
)
def test_a_weekend_holiday_reaches_into_the_next_or_last_year(
    day: date,
    workday: Literal["friday", "monday"],
    today: date,
) -> None:
    def dates(year: int) -> tuple[date, ...]:
        return (day,) if year == day.year else ()

    event = holiday.Holiday(name="x", dates=dates, icons=(), workday=workday)
    assert holiday._in_window(event, today)


def test_a_fixed_date_is_its_month_and_day_and_a_name_its_rule() -> None:
    assert holiday._rule("07-04")(2026) == (date(2026, 7, 4),)
    assert holiday._rule("easter") is holiday.easter
    with pytest.raises(ValueError, match="unpack"):
        holiday._rule("nope")


def _delta_t_2004(t: float) -> float:
    # eclipse.gsfc.nasa.gov/SEhelp/deltatpoly2004.html, 1986-2005, t = y - 2000.
    return (
        63.86
        + 0.3345 * t
        - 0.060_374 * t**2
        + 0.001_727_5 * t**3
        + 0.000_651_814 * t**4
        + 0.000_023_735_99 * t**5
    )


@pytest.mark.parametrize(
    ("year", "seconds"),
    [
        (1986.0, _delta_t_2004(-14)),
        (2004.5, _delta_t_2004(4.5)),
        # 2005 and 2050 open the next polynomials.
        (2005.0, 62.92 + 0.322_17 * 5 + 0.005_589 * 25),
        (2050.0, -20 + 32 * 2.3**2 - 0.5628 * 100),
    ],
)
def test_delta_t_uses_each_eras_polynomial_from_its_first_year(
    year: float,
    seconds: float,
) -> None:
    jde = 2_451_545.0 + (year - 2000) * 365.25
    assert holiday._delta_t_sec(jde) == pytest.approx(seconds, rel=0, abs=1e-9)


# Meeus table 27.C, years 1000-3000: at 3000 and 1000 the millennia y are 1 and -1.
@pytest.mark.parametrize(
    ("index", "terms"),
    [
        (0, (2_451_623.809_84, 365_242.374_04, 0.051_69, -0.004_11, -0.000_57)),
        (1, (2_451_716.567_67, 365_241.626_03, 0.003_25, 0.008_88, -0.000_30)),
        (2, (2_451_810.217_15, 365_242.017_67, -0.115_75, 0.003_37, 0.000_78)),
        (3, (2_451_900.059_52, 365_242.740_49, -0.062_23, -0.008_23, 0.000_32)),
    ],
)
def test_each_season_follows_every_term_of_meeus_table(
    index: int,
    terms: tuple[float, float, float, float, float],
) -> None:
    a0, a1, a2, a3, a4 = terms
    assert holiday._season(3000, index) == pytest.approx(
        a0 + a1 + a2 + a3 + a4,
        rel=0,
        abs=1e-6,
    )
    assert holiday._season(1000, index) == pytest.approx(
        a0 - a1 + a2 - a3 + a4,
        rel=0,
        abs=1e-6,
    )


@pytest.mark.parametrize(
    "pixels",
    [("RR", "RR", "RR"), ("RR", "R")],
)
def test_an_icon_with_an_odd_row_or_a_short_one_is_refused(
    pixels: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="zip"):
        icon_rows(Icon(name="bad", palette={"R": "#ff0000"}, pixels=pixels))


def test_each_terminal_row_is_two_pixel_rows() -> None:
    icon = Icon(
        name="test",
        palette={"R": "#ff0000", "B": "#0000ff"},
        pixels=("R.", ".R", "B.", "BB"),
    )
    assert [r.plain for r in icon_rows(icon)] == ["▀▄", "█▄"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
