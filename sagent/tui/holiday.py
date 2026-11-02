"""Holidays, their dates, and the icons that replace the welcome logo then.

Each holiday falls on the date its US observers keep. Seasons, Nowruz, and
Holi follow the sky on the user's own calendar. Vesak and the Eids are
announced, by the UN and the Fiqh Council of North America. Lunar New Year and
the Mid-Autumn Festival follow China's calendar; Diwali and Vaisakhi, India's.
Hebrew dates are exact arithmetic. Astronomy follows Meeus, "Astronomical Algorithms"
(1998), ch. 27 and 49, accurate to seconds; ``holiday_test`` checks each rule
against published dates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from functools import cache, partial
from importlib.resources import files
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

import math

from treekle import from_plain

from sagent.lib.yaml import parse


if TYPE_CHECKING:
    from random import Random

    from rich.text import Text
else:
    from wrapt import lazy_import

    Text = lazy_import("rich.text", "Text")


@dataclass(frozen=True, slots=True, kw_only=True)
class Icon:
    """An 11x12 pixel picture; each pixel names a palette color, "." is clear."""

    name: str
    palette: dict[str, str]
    pixels: tuple[str, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Holiday:
    """A recurring holiday: its dates, the workday its weekends reach, its icons."""

    name: str
    dates: Callable[[int], tuple[date, ...]]
    icons: tuple[Icon, ...]
    workday: Literal["friday", "monday"] = "friday"


def holiday_icon(today: date, *, rng: Random) -> Icon | None:
    """Pick an icon of a random holiday showing ``today``, if any.

    A holiday shows on its day. On a Saturday or Sunday, it also shows on the
    rest of that weekend and its ``workday``, Friday or Monday, so it is seen
    at work.

    Args:
      today: The local date.
      rng: Chooses among overlapping holidays, then between their icons.

    Returns:
      icon: The chosen icon, or None outside every holiday.

    """
    current = [h for h in holidays() if _in_window(h, today)]
    return rng.choice(rng.choice(current).icons) if current else None


def icon_rows(icon: Icon) -> list[Text]:
    """Draw ``icon`` as rows of half blocks, two pixel rows per terminal row.

    Args:
      icon: The picture to draw.

    Returns:
      rows: One styled row per pair of pixel rows.

    """
    rows: list[Text] = []
    for upper, lower in zip(icon.pixels[::2], icon.pixels[1::2], strict=True):
        row = Text()
        for top, bottom in zip(upper, lower, strict=True):
            if top == bottom == ".":
                row.append(" ")
            elif bottom == ".":
                row.append("▀", style=icon.palette[top])
            elif top == ".":
                row.append("▄", style=icon.palette[bottom])
            elif top == bottom:
                row.append("█", style=icon.palette[top])
            else:
                row.append("▀", style=f"{icon.palette[top]} on {icon.palette[bottom]}")
        rows.append(row)
    return rows


@cache
def holidays() -> tuple[Holiday, ...]:
    """List every holiday, in calendar order, from ``assets/holidays.yaml``.

    Returns:
      holidays: Each holiday with its dates and its two icons.

    """
    asset = files(__package__) / "assets" / "holidays.yaml"
    text = asset.read_text(encoding="utf-8")
    return tuple(
        Holiday(
            name=entry["name"],
            dates=_rule(entry["dates"]),
            workday=entry.get("workday", "friday"),
            icons=tuple(
                Icon(
                    name=icon["name"],
                    palette=icon["palette"],
                    pixels=tuple(icon["pixels"].split()),
                )
                for icon in entry["icons"]
            ),
        )
        for entry in from_plain(parse(text, source=str(asset)), list[_HolidayEntry])
    )


def spring(year: int) -> tuple[date, ...]:
    """Return the March equinox's local date; Nowruz begins then too."""
    return (_civil_date(_season(year, 0), zone=None),)


def summer(year: int) -> tuple[date, ...]:
    """Return the June solstice's local date."""
    return (_civil_date(_season(year, 1), zone=None),)


def fall(year: int) -> tuple[date, ...]:
    """Return the September equinox's local date."""
    return (_civil_date(_season(year, 2), zone=None),)


def winter(year: int) -> tuple[date, ...]:
    """Return the December solstice's local date."""
    return (_civil_date(_season(year, 3), zone=None),)


def lunar_new_year(year: int) -> tuple[date, ...]:
    """Return the first new moon on or after January 21, in China time."""
    china = timezone(timedelta(hours=8))
    return (_new_or_full_moon(date(year, 1, 21), phase="new", zone=china),)


def mid_autumn(year: int) -> tuple[date, ...]:
    """Return day 15 of the lunar month holding the September equinox, China time.

    Args:
      year: The Gregorian year.

    Returns:
      days: The Mid-Autumn Festival's date.

    """
    # The "2033 problem": China's calendar places a leap month that makes the
    # eighth month the one before the equinox's, against the rule below.
    if year == 2033:
        return (date(2033, 9, 8),)
    china = timezone(timedelta(hours=8))
    equinox = _civil_date(_season(year, 2), zone=china)
    # pragma: no mutate start -- slack search starts: a day further finds the same moons.
    new_moon = _new_or_full_moon(equinox - timedelta(days=30), phase="new", zone=china)
    following = _new_or_full_moon(new_moon + timedelta(days=1), phase="new", zone=china)
    # pragma: no mutate end
    if following <= equinox:
        new_moon = following
    return (new_moon + timedelta(days=14),)


def holi(year: int) -> tuple[date, ...]:
    """Return the morning after the bonfire night of the Phalguna full moon."""
    # The full moon ends its lunar day. Ending after 17:00 local time, that day
    # holds the evening bonfire, so Holi is the next day: +7 h rolls over then.
    return (
        _new_or_full_moon(date(year, 2, 28), phase="full", zone=None, shift_hours=7),
    )


def diwali(year: int) -> tuple[date, ...]:
    """Return the first evening of prayers in the Kartika new moon's lunar day.

    Args:
      year: The Gregorian year.

    Returns:
      days: Diwali's date, India time, which US temples keep too.

    """
    # The first 18:00 India time at or after the lunar day begins: 6 h later, the
    # date has moved on unless that day's evening falls within it.
    india = _india()
    return (
        _first_phase(
            date(year, 10, 17),
            phase="new",
            day=lambda jde: _civil_date(_tithi_start(jde), zone=india, shift_hours=6),
        ),
    )


def vaisakhi(year: int) -> tuple[date, ...]:
    """Return the sun's entry into sidereal Aries (Lahiri), in India time."""
    ayanamsa_deg = 24.22 + 0.013_97 * (year - 2026)
    sun_deg_per_day = 0.985
    jde = _season(year, 0) + ayanamsa_deg / sun_deg_per_day
    return (_civil_date(jde, zone=_india()),)


def hanukkah(year: int) -> tuple[date, ...]:
    """Return the evening the first candle is lit, the eve of 25 Kislev."""
    hebrew_year = year + 3761
    new_year = _hebrew_new_year(hebrew_year)
    complete = (_hebrew_new_year(hebrew_year + 1) - new_year) % 10 == 5
    return (date.fromordinal(new_year + 30 + (30 if complete else 29) + 23),)


def vesak(year: int) -> tuple[date, ...]:
    """Return the UN's Day of Vesak, else the first full moon of May in UTC.

    Args:
      year: The Gregorian year.

    Returns:
      days: Vesak's date, one for the whole country.

    """
    # timeanddate.com/holidays/un/vesak-day. Buddhist councils pick the full
    # moon, so no zone's rule matches every year: 2015 was June 1. Past the
    # list, UTC agrees with the most UN years, 20 of 26.
    return _listed(
        year,
        dates=(
            "2005-05-23",
            "2006-05-13",
            "2007-05-02",
            "2008-05-20",
            "2009-05-08",
            "2010-05-27",
            "2011-05-17",
            "2012-05-06",
            "2013-05-25",
            "2014-05-13",
            "2015-06-01",
            "2016-05-20",
            "2017-05-10",
            "2018-05-29",
            "2019-05-19",
            "2020-05-07",
            "2021-05-26",
            "2022-05-16",
            "2023-05-05",
            "2024-05-23",
            "2025-05-12",
            "2026-05-01",
            "2027-05-20",
            "2028-05-08",
            "2029-05-27",
            "2030-05-27",
        ),
        otherwise=_vesak_estimate,
    )


def eid_al_fitr(year: int) -> tuple[date, ...]:
    """Return 1 Shawwal by the Fiqh Council of North America, else arithmetic.

    Args:
      year: The Gregorian year.

    Returns:
      days: Eid al-Fitr's dates; some years hold two.

    """
    # fiqhcouncil.org/calendar, which runs through 2045.
    return _listed(
        year,
        dates=(
            "2026-03-20",
            "2027-03-09",
            "2028-02-26",
            "2029-02-14",
            "2030-02-04",
            "2031-01-24",
            "2032-01-14",
            "2033-01-02",
            "2033-12-23",
            "2034-12-12",
            "2035-12-01",
            "2036-11-19",
            "2037-11-08",
            "2038-10-29",
            "2039-10-19",
            "2040-10-07",
            "2041-09-26",
            "2042-09-15",
            "2043-09-04",
            "2044-08-24",
            "2045-08-14",
        ),
        otherwise=partial(_islamic, month=10, day=1),
    )


def eid_al_adha(year: int) -> tuple[date, ...]:
    """Return the day after Arafah by the Fiqh Council of North America, else arithmetic.

    Args:
      year: The Gregorian year.

    Returns:
      days: Eid al-Adha's dates; some years hold two.

    """
    # fiqhcouncil.org/calendar: 1 Dhul-Hijjah plus nine days. The council
    # defers to Saudi Arabia's Day of Arafah, which its calendar anticipates.
    return _listed(
        year,
        dates=(
            "2026-05-27",
            "2027-05-16",
            "2028-05-05",
            "2029-04-24",
            "2030-04-13",
            "2031-04-02",
            "2032-03-22",
            "2033-03-11",
            "2034-03-01",
            "2035-02-18",
            "2036-02-07",
            "2037-01-26",
            "2038-01-16",
            "2039-01-05",
            "2039-12-26",
            "2040-12-14",
            "2041-12-04",
            "2042-11-23",
            "2043-11-12",
            "2044-10-31",
            "2045-10-21",
        ),
        otherwise=partial(_islamic, month=12, day=10),
    )


def easter(year: int) -> tuple[date, ...]:
    """Return Western Easter, by the anonymous Gregorian computus.

    Args:
      year: The Gregorian year.

    Returns:
      days: Easter Sunday's date.

    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    # pragma: no mutate start -- Meeus, "Astronomical Algorithms" ch. 8: f's 8 or 9 give one g to 4099, and 15 is -15 mod 30.
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    # pragma: no mutate end
    i, k = divmod(c, 4)
    j = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * j) // 451
    month, day = divmod(h + j - 7 * m + 114, 31)
    return (date(year, month, day + 1),)


def thanksgiving(year: int) -> tuple[date, ...]:
    """Return the fourth Thursday of November."""
    first = date(year, 11, 1)
    return (first + timedelta(days=(3 - first.weekday()) % 7 + 21),)


def _in_window(holiday: Holiday, today: date) -> bool:
    """Whether ``today`` is a holiday, or in the weekend and workday holding one."""
    first = 4 if holiday.workday == "friday" else 5  # Friday, else Saturday.
    for year in (today.year - 1, today.year, today.year + 1):
        for day in holiday.dates(year):
            opens = day - timedelta(days=day.weekday() - first)
            weekend = day.weekday() >= 5  # Saturday or Sunday.
            if today == day or (weekend and 0 <= (today - opens).days <= 2):
                return True
    return False


def _rule(dates: str) -> Callable[[int], tuple[date, ...]]:
    """Read a computed rule's name, else ``MM-DD`` as a fixed date."""
    computed = (
        lunar_new_year,
        holi,
        spring,
        easter,
        vaisakhi,
        eid_al_fitr,
        vesak,
        summer,
        eid_al_adha,
        fall,
        mid_autumn,
        diwali,
        thanksgiving,
        hanukkah,
        winter,
    )
    rule = {rule.__name__: rule for rule in computed}.get(dates)
    if rule is not None:
        return rule
    month, day = dates.split("-")
    return partial(_annual, month=int(month), day=int(day))


class _IconEntry(TypedDict):
    name: str
    palette: dict[str, str]
    pixels: str


class _HolidayEntry(TypedDict):
    name: str
    dates: str
    why: str  # Required, so every holiday states which inclusion rule it meets.
    workday: NotRequired[Literal["friday", "monday"]]
    icons: list[_IconEntry]


def _annual(year: int, *, month: int, day: int) -> tuple[date, ...]:
    return (date(year, month, day),)


def _vesak_estimate(year: int) -> tuple[date, ...]:
    return (_new_or_full_moon(date(year, 5, 1), phase="full", zone=UTC),)


def _listed(
    year: int,
    *,
    dates: tuple[str, ...],
    otherwise: Callable[[int], tuple[date, ...]],
) -> tuple[date, ...]:
    """Return ``dates`` in ``year`` when they span it, else ``otherwise(year)``."""
    if int(dates[0][:4]) <= year <= int(dates[-1][:4]):
        return tuple(date.fromisoformat(d) for d in dates if d.startswith(f"{year}-"))
    return otherwise(year)


# ``zone=None`` is the user's own time zone, as in ``datetime.astimezone``.
def _civil_date(jde: float, *, zone: tzinfo | None, shift_hours: float = 0) -> date:
    """Return the date in ``zone`` of Julian day ``jde``, ``shift_hours`` later."""
    j2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)
    moment = j2000 + timedelta(
        days=jde - 2_451_545.0,
        hours=shift_hours,
        seconds=-_delta_t_sec(jde),
    )
    return moment.astimezone(zone).date()


def _delta_t_sec(jde: float) -> float:
    """Return how far terrestrial time runs ahead of UTC (Espenak and Meeus, 2006)."""
    # Over a minute today and some three by 2090: enough to move a moon
    # phase near midnight onto the other day.
    year = 2000 + (jde - 2_451_545.0) / 365.25
    t = year - 2000
    if year < 2005:
        return 63.86 + t * (
            0.3345
            + t
            * (
                -0.060_374
                + t * (0.001_727_5 + t * (0.000_651_814 + t * 0.000_023_735_99))
            )
        )
    if year < 2050:
        return 62.92 + t * (0.322_17 + t * 0.005_589)
    u = (year - 1820) / 100
    return -20 + 32 * u * u - 0.5628 * (2150 - year)


def _season(year: int, index: int) -> float:
    """Return the Julian day of the March (0), June, September, or December point."""
    y = (year - 2000) / 1000
    a0, a1, a2, a3, a4 = (
        (2_451_623.809_84, 365_242.374_04, 0.051_69, -0.004_11, -0.000_57),
        (2_451_716.567_67, 365_241.626_03, 0.003_25, 0.008_88, -0.000_30),
        (2_451_810.217_15, 365_242.017_67, -0.115_75, 0.003_37, 0.000_78),
        (2_451_900.059_52, 365_242.740_49, -0.062_23, -0.008_23, 0.000_32),
    )[index]
    return a0 + y * (a1 + y * (a2 + y * (a3 + y * a4)))


def _new_or_full_moon(
    start: date,
    *,
    phase: Literal["new", "full"],
    zone: tzinfo | None,
    shift_hours: float = 0,
) -> date:
    """Return the first new or full moon's date in ``zone`` on or after ``start``."""
    return _first_phase(
        start,
        phase=phase,
        day=partial(_civil_date, zone=zone, shift_hours=shift_hours),
    )


def _first_phase(
    start: date,
    *,
    phase: Literal["new", "full"],
    day: Callable[[float], date],
) -> date:
    """Return the first ``day`` of a new or full moon's Julian day on or after ``start``."""
    # pragma: no mutate start -- an estimate a lunation early only adds steps to the loop below (Meeus ch. 49).
    lunation = 29.530_588_861
    jd = start.toordinal() + 1_721_424.5
    half = {"new": 0.0, "full": 0.5}[phase]
    k = math.floor((jd - 2_451_550.097_66) / lunation) - 1 + half
    # pragma: no mutate end
    while (found := day(_moon_phase(k))) < start:
        k += 1
    return found


# It begins as the moon comes within 12 degrees of the sun, found by Newton's method
# from the new moon.
def _tithi_start(new_moon: float) -> float:
    """Return the Julian day the lunar day ending at ``new_moon`` begins."""
    jde = new_moon
    target = 348
    tolerance = 1e-7
    # pragma: no mutate start -- Newton's slope (Meeus ch. 49), wrap and ``>`` only set the steps to the one root.
    rate = 360 / 29.530_588_861
    while abs(step := math.remainder(_elongation(jde) - target, 360)) > tolerance:
        jde -= step / rate
    # pragma: no mutate end
    return jde


# Meeus ch. 25 for the sun and the largest terms of ch. 47 for the moon: within a
# hundredth of a degree, some minutes of a lunar day. Nutation shifts both alike.
def _elongation(jde: float) -> float:
    """Return how far the moon's longitude runs ahead of the sun's, in degrees."""
    t = (jde - 2_451_545.0) / 36_525
    sun_mean = 280.466_46 + 36_000.769_83 * t + 0.000_303_2 * t * t
    m = math.radians(357.529_11 + 35_999.050_29 * t - 0.000_153_7 * t * t)
    sun = (
        sun_mean
        + (1.914_602 - 0.004_817 * t - 0.000_014 * t * t) * math.sin(m)
        + (0.019_993 - 0.000_101 * t) * math.sin(2 * m)
        + 0.000_289 * math.sin(3 * m)
        - 0.005_69
    )
    moon_mean = 218.316_447_7 + 481_267.881_234_21 * t - 0.001_578_6 * t * t
    d = math.radians(297.850_192_1 + 445_267.111_403_4 * t - 0.001_881_9 * t * t)
    mp = math.radians(134.963_396_4 + 477_198.867_505_5 * t + 0.008_741_4 * t * t)
    f = math.radians(93.272_095_0 + 483_202.017_523_3 * t - 0.003_653_9 * t * t)
    e = 1 - 0.002_516 * t - 0.000_007_4 * t * t
    moon = moon_mean + 1e-6 * sum(
        coefficient
        # These terms take M at most once, so E, never E squared.
        * (e if cm else 1)
        * math.sin(cd * d + cm * m + cmp * mp + cf * f)
        for coefficient, cd, cm, cmp, cf in _moon_longitude_terms()
    )
    return (moon - sun) % 360


def _moon_longitude_terms() -> tuple[tuple[int, int, int, int, int], ...]:
    """Return Meeus table 47.A's largest terms: millionths of a degree, D, M, M', F."""
    return (
        (6_288_774, 0, 0, 1, 0),
        (1_274_027, 2, 0, -1, 0),
        (658_314, 2, 0, 0, 0),
        (213_618, 0, 0, 2, 0),
        (-185_116, 0, 1, 0, 0),
        (-114_332, 0, 0, 0, 2),
        (58_793, 2, 0, -2, 0),
        (57_066, 2, -1, -1, 0),
        (53_322, 2, 0, 1, 0),
        (45_758, 2, -1, 0, 0),
        (-40_923, 0, 1, -1, 0),
        (-34_720, 1, 0, 0, 0),
        (-30_383, 0, 1, 1, 0),
        (15_327, 2, 0, 0, -2),
        (-12_528, 0, 0, 1, 2),
        (10_980, 0, 0, 1, -2),
        (10_675, 4, 0, -1, 0),
        (10_034, 0, 0, 3, 0),
        (8_548, 4, 0, -2, 0),
        (-7_888, 2, 1, -1, 0),
        (-6_766, 2, 1, 0, 0),
        (-5_163, 1, 0, -1, 0),
        (4_987, 1, 1, 0, 0),
        (4_036, 2, -1, 1, 0),
    )


# Meeus ch. 49 with every periodic term, good to seconds; seven terms alone stray by
# minutes, enough to move a phase across midnight.
def _moon_phase(k: float) -> float:
    """Return the Julian day of new moon ``k`` (integer) or full moon (half)."""
    t = k / 1236.85
    jde = (
        2_451_550.097_66
        + 29.530_588_861 * k
        + t * t * (0.000_154_37 + t * (-0.000_000_150 + t * 0.000_000_000_73))
    )
    e = 1 - t * (0.002_516 + t * 0.000_007_4)
    sun = math.radians(2.5534 + 29.105_356_70 * k - t * t * (0.000_001_4 + t * 1.1e-7))
    moon = math.radians(
        201.5643
        + 385.816_935_28 * k
        + t * t * (0.010_758_2 + t * (0.000_012_38 - t * 5.8e-8)),
    )
    lat = math.radians(
        160.7108
        + 390.670_502_84 * k
        - t * t * (0.001_611_8 + t * (0.000_002_27 - t * 1.1e-8)),
    )
    node = math.radians(124.7746 - 1.563_755_88 * k + t * t * 0.002_067_2)
    periodic = sum(
        coefficient
        * (1, e, e * e)[e_power]
        * math.sin(a * moon + b * sun + c * lat + d * node)
        for coefficient, e_power, a, b, c, d in _lunar_terms(full=bool(k % 1))
    )
    planetary = sum(
        coefficient * math.sin(math.radians(start + rate * k + squared * t * t))
        for coefficient, start, rate, squared in _planetary_terms()
    )
    return jde + periodic + planetary


def _lunar_terms(*, full: bool) -> tuple[tuple[float, int, int, int, int, int], ...]:
    """Return Meeus table 49.A: coefficient, power of E, and multiples of M', M, F, Ω."""
    first = (
        (-0.406_14, 0.173_02, 0.016_14, 0.010_43, 0.007_34, -0.005_15, 0.002_09)
        if full
        else (-0.407_20, 0.172_41, 0.016_08, 0.010_39, 0.007_39, -0.005_14, 0.002_08)
    )
    return (
        (first[0], 0, 1, 0, 0, 0),
        (first[1], 1, 0, 1, 0, 0),
        (first[2], 0, 2, 0, 0, 0),
        (first[3], 0, 0, 0, 2, 0),
        (first[4], 1, 1, -1, 0, 0),
        (first[5], 1, 1, 1, 0, 0),
        (first[6], 2, 0, 2, 0, 0),
        (-0.001_11, 0, 1, 0, -2, 0),
        (-0.000_57, 0, 1, 0, 2, 0),
        (0.000_56, 1, 2, 1, 0, 0),
        (-0.000_42, 0, 3, 0, 0, 0),
        (0.000_42, 1, 0, 1, 2, 0),
        (0.000_38, 1, 0, 1, -2, 0),
        (-0.000_24, 1, 2, -1, 0, 0),
        (-0.000_17, 0, 0, 0, 0, 1),
        (-0.000_07, 0, 1, 2, 0, 0),
        (0.000_04, 0, 2, 0, -2, 0),
        (0.000_04, 0, 0, 3, 0, 0),
        (0.000_03, 0, 1, 1, -2, 0),
        (0.000_03, 0, 2, 0, 2, 0),
        (-0.000_03, 0, 1, 1, 2, 0),
        (0.000_03, 0, 1, -1, 2, 0),
        (-0.000_02, 0, 1, -1, -2, 0),
        (-0.000_02, 0, 3, 1, 0, 0),
        (0.000_02, 0, 4, 0, 0, 0),
    )


def _planetary_terms() -> tuple[tuple[float, float, float, float], ...]:
    """Return Meeus's additional corrections A1-A14: coefficient, angle, rate, T²."""
    return (
        (0.000_325, 299.77, 0.107_408, -0.009_173),
        (0.000_165, 251.88, 0.016_321, 0),
        (0.000_164, 251.83, 26.651_886, 0),
        (0.000_126, 349.42, 36.412_478, 0),
        (0.000_110, 84.66, 18.206_239, 0),
        (0.000_062, 141.74, 53.303_771, 0),
        (0.000_060, 207.14, 2.453_732, 0),
        (0.000_056, 154.84, 7.306_860, 0),
        (0.000_047, 34.52, 27.261_239, 0),
        (0.000_042, 207.19, 0.121_824, 0),
        (0.000_040, 291.34, 1.844_379, 0),
        (0.000_037, 161.72, 24.198_154, 0),
        (0.000_035, 239.56, 25.513_099, 0),
        (0.000_023, 331.55, 3.592_518, 0),
    )


def _hebrew_new_year(hebrew_year: int) -> int:
    """Return the Gregorian ordinal of 1 Tishrei (Dershowitz and Reingold)."""
    elapsed = [_hebrew_elapsed_days(hebrew_year + i) for i in (-1, 0, 1)]
    if elapsed[2] - elapsed[1] == 356:
        delay = 2
    elif elapsed[1] - elapsed[0] == 382:
        delay = 1
    else:
        delay = 0
    return -1_373_427 + elapsed[1] + delay


def _hebrew_elapsed_days(hebrew_year: int) -> int:
    """Return days from the epoch to the molad of Tishrei, after postponement."""
    months = (235 * hebrew_year - 234) // 19
    # The epoch's molad, 12,084 parts, then 13,753 a month.
    # pragma: no mutate start -- Dershowitz-Reingold "hebrew-calendar-elapsed-days"; one part more moves no new year 3800-9999.
    parts = 12_084 + 13_753 * months
    # pragma: no mutate end
    days = 29 * months + parts // 25_920
    return days + 1 if (3 * (days + 1)) % 7 < 3 else days


def _islamic(year: int, *, month: int, day: int) -> tuple[date, ...]:
    """Return each ``month``/``day`` of the arithmetic Islamic calendar in ``year``."""
    # pragma: no mutate start -- a wider window of years adds only dates the filter drops.
    first = (year - 622) * 33 // 32
    hijri_years = range(first - 1, first + 3)
    # pragma: no mutate end
    days = (
        date.fromordinal(
            day
            + 29 * (month - 1)
            + month // 2
            + (hijri - 1) * 354
            + (3 + 11 * hijri) // 30
            + 227_014,
        )
        for hijri in hijri_years
    )
    return tuple(d for d in days if d.year == year)


def _india() -> timezone:
    """Return India Standard Time, UTC+05:30, which both Indian festivals keep."""
    return timezone(timedelta(hours=5, minutes=30))
