"""Holidays, their dates, and the icons that replace the welcome logo then.

Hebrew dates are exact arithmetic. Seasons and lunisolar dates follow Meeus,
"Astronomical Algorithms" (1998), ch. 27 and 49, accurate to minutes;
``holiday_test`` checks them against published dates. Eid and Vesak follow
moon sightings and national committees, so they come from announced dates,
with an arithmetic estimate outside them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial
from typing import TYPE_CHECKING

import math


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
    """A recurring holiday: its start dates in a year, its length, its icons."""

    name: str
    starts: Callable[[int], tuple[date, ...]]
    icons: tuple[Icon, Icon]
    days: int = 1


def holiday_icon(today: date, *, rng: Random) -> Icon | None:
    """Pick an icon of a random holiday from the weekday before it to its end."""
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


def holidays() -> tuple[Holiday, ...]:
    """List every holiday, in calendar order.

    Returns:
      holidays: Each holiday with its dates and its two icons.

    """
    return (
        Holiday(
            name="New Year's",
            starts=partial(_annual, month=1, day=1),
            icons=(
                Icon(
                    name="fireworks",
                    palette={
                        "Y": "#ffd23f",
                        "R": "#ff4d4d",
                        "B": "#4da6ff",
                        "W": "#ffffff",
                        "g": "#888888",
                    },
                    pixels=(
                        "...R.B.R...",
                        ".B.......B.",
                        "R...Y.Y...R",
                        "...Y...Y...",
                        "B.Y..W..Y.B",
                        "....WWW....",
                        "B.Y..W..Y.B",
                        "...Y...Y...",
                        "R...Y.Y...R",
                        ".B.......B.",
                        "...R.B.R...",
                        "...........",
                    ),
                ),
                Icon(
                    name="champagne toast",
                    palette={
                        "G": "#f4c542",
                        "W": "#e8f4ff",
                        "B": "#fffbe6",
                        "S": "#cfd8dc",
                        "Y": "#ffd23f",
                    },
                    pixels=(
                        ".....Y.....",
                        "...Y.Y.Y...",
                        ".W..W.W..W.",
                        ".WGGW.WGGW.",
                        ".WGBW.WBGW.",
                        ".WBGW.WGBW.",
                        ".WGGW.WGGW.",
                        "..WW...WW..",
                        "...S...S...",
                        "...S...S...",
                        "..SSS.SSS..",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Lunar New Year",
            starts=lunar_new_year,
            icons=(
                Icon(
                    name="lantern",
                    palette={
                        "R": "#e0262f",
                        "D": "#a01c2c",
                        "Y": "#ffd23f",
                        "O": "#ffb300",
                    },
                    pixels=(
                        ".....Y.....",
                        "...YYYYY...",
                        "..RRRRRRR..",
                        ".RRDRRRDRR.",
                        "RRRDRRRDRRR",
                        "RRRDRYRDRRR",
                        "RRRDRRRDRRR",
                        ".RRDRRRDRR.",
                        "..RRRRRRR..",
                        "...YYYYY...",
                        "....OOO....",
                        "....O.O....",
                    ),
                ),
                Icon(
                    name="firecrackers",
                    palette={
                        "R": "#e0262f",
                        "Y": "#ffd23f",
                        "f": "#8b5a2b",
                        "O": "#ff8c1a",
                    },
                    pixels=(
                        ".O...O...O.",
                        ".f...f...f.",
                        "RRR.RRR.RRR",
                        "YYY.YYY.YYY",
                        "RRR.RRR.RRR",
                        "RRR.RRR.RRR",
                        "RRR.RRR.RRR",
                        "YYY.YYY.YYY",
                        "RRR.RRR.RRR",
                        "...........",
                        "...........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Valentine's Day",
            starts=partial(_annual, month=2, day=14),
            icons=(
                Icon(
                    name="heart",
                    palette={"R": "#e53950", "P": "#ff8fa3"},
                    pixels=(
                        "...........",
                        ".RRR...RRR.",
                        "RPPRR.RRRRR",
                        "RPRRRRRRRRR",
                        "RRRRRRRRRRR",
                        "RRRRRRRRRRR",
                        ".RRRRRRRRR.",
                        "..RRRRRRR..",
                        "...RRRRR...",
                        "....RRR....",
                        ".....R.....",
                        "...........",
                    ),
                ),
                Icon(
                    name="rose",
                    palette={
                        "R": "#d32f2f",
                        "r": "#ff6b6b",
                        "D": "#8e1b1b",
                        "G": "#3fa34d",
                    },
                    pixels=(
                        "...RRRRR...",
                        "..RrrrRDR..",
                        ".RrRRRrDRR.",
                        ".RrRDRrDRR.",
                        ".RRrRRrRRR.",
                        "..RRrrrRR..",
                        "...RRRRR...",
                        ".....G.....",
                        ".GG..G.....",
                        "..GGGG..GG.",
                        ".....GGGG..",
                        ".....G.....",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Pi Day",
            starts=partial(_annual, month=3, day=14),
            icons=(
                Icon(
                    name="pi",
                    palette={"B": "#4a90c4"},
                    pixels=(
                        "...........",
                        "BBBBBBBBBBB",
                        "BBBBBBBBBBB",
                        "..BB...BB..",
                        "..BB...BB..",
                        "..BB...BB..",
                        "..BB...BB..",
                        ".BB....BB..",
                        ".BB....BBB.",
                        "BB......BB.",
                        "...........",
                        "...........",
                    ),
                ),
                Icon(
                    name="pie slice",
                    palette={
                        "C": "#d9a35b",
                        "D": "#a8702e",
                        "R": "#c62839",
                        "P": "#f06070",
                    },
                    pixels=(
                        "...........",
                        "..........C",
                        "........CCC",
                        "......CCCCC",
                        "....CCCCCCC",
                        "..CCCCCCCCC",
                        "CRRPRRPRRRC",
                        "CRRRRRRRRRC",
                        "CPRRRPRRRPC",
                        "DDDDDDDDDDD",
                        "...........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="St. Patrick's Day",
            starts=partial(_annual, month=3, day=17),
            icons=(
                Icon(
                    name="shamrock",
                    palette={"G": "#3fa34d", "L": "#7fd67f", "D": "#2a7a36"},
                    pixels=(
                        "....GGG....",
                        "...GLGGG...",
                        "...GGGGG...",
                        ".GG.GGG.GG.",
                        "GLGG.G.GLGG",
                        "GGGGGGGGGGG",
                        ".GGGG.GGGG.",
                        "..GG.D.GG..",
                        ".....D.....",
                        "......D....",
                        "......D....",
                        "...........",
                    ),
                ),
                Icon(
                    name="pot of gold",
                    palette={
                        "K": "#4a4a4a",
                        "k": "#7a7a7a",
                        "Y": "#ffd23f",
                        "O": "#e0a800",
                    },
                    pixels=(
                        "...........",
                        "...Y.Y.Y...",
                        "..YOYYOYY..",
                        ".YYYOYYOYY.",
                        "kkkkkkkkkkk",
                        ".KKKKKKKKK.",
                        "KKKKKKKKKKK",
                        "KKKKKKKKKKK",
                        "KKKKKKKKKKK",
                        ".KKKKKKKKK.",
                        "..K.....K..",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Holi",
            starts=holi,
            icons=(
                Icon(
                    name="powder burst",
                    palette={
                        "M": "#e91e8c",
                        "Y": "#ffd600",
                        "G": "#2ecc40",
                        "C": "#00c8e8",
                        "O": "#ff7a00",
                    },
                    pixels=(
                        "..M....G...",
                        ".MMM.M.GG.G",
                        "MMMMMMGGGG.",
                        ".MMMMMGGGGG",
                        "..MMMYGGGG.",
                        ".OOOYYYCCC.",
                        "OOOOOYCCCCC",
                        ".OOOOOCCCC.",
                        "OOOOO.CCCC.",
                        ".O.OO..CC.C",
                        "..O.....C..",
                        "...........",
                    ),
                ),
                Icon(
                    name="handprint",
                    palette={
                        "P": "#ff4fa3",
                        "O": "#ff8c1a",
                        "Y": "#ffd600",
                        "C": "#00c8e8",
                    },
                    pixels=(
                        "..P.P.P....",
                        "..P.P.P.P..",
                        "..P.P.P.P..",
                        "..PPPPPPP..",
                        "P.PPPPPPP..",
                        "PPPPPPPPP..",
                        ".PPPPPPPP..",
                        "..PPPPPPP..",
                        "...PPPPP...",
                        "...PPPPP...",
                        "O.......Y..",
                        "...C..O...C",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Nowruz",
            starts=nowruz,
            icons=(
                Icon(
                    name="hyacinth",
                    palette={
                        "P": "#8e5fd6",
                        "p": "#c3a6f0",
                        "G": "#4caf50",
                        "T": "#c1663f",
                    },
                    pixels=(
                        "....P.P....",
                        "...PpPpP...",
                        "...PPpPP...",
                        "...PpPpP...",
                        "...PPpPP...",
                        "...PpPpP...",
                        "....PPP....",
                        ".G..G.G..G.",
                        "..GG.G.GG..",
                        "...TTTTT...",
                        "...TTTTT...",
                        "....TTT....",
                    ),
                ),
                Icon(
                    name="sabzeh",
                    palette={
                        "G": "#4cbb5a",
                        "L": "#8be38b",
                        "R": "#d32f2f",
                        "T": "#c1663f",
                    },
                    pixels=(
                        ".L.G.L.G.L.",
                        ".G.L.G.L.G.",
                        "LG.GLG.GL.G",
                        ".GLGLGLGLG.",
                        ".LGLGLGLGL.",
                        ".GGGGGGGGG.",
                        "RRRRRRRRRRR",
                        ".TTTTTTTTT.",
                        ".TTTTTTTTT.",
                        "..TTTTTTT..",
                        "...........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="First day of spring",
            starts=spring,
            icons=(
                Icon(
                    name="tulip",
                    palette={"P": "#f06292", "p": "#f8bbd0", "G": "#4caf50"},
                    pixels=(
                        "...P.P.P...",
                        "...PpPpP...",
                        "...PPpPP...",
                        "...PPPPP...",
                        "....PPP....",
                        ".....G.....",
                        ".GG..G..GG.",
                        "..GG.G.GG..",
                        "...GGGGG...",
                        ".....G.....",
                        ".....G.....",
                        "...........",
                    ),
                ),
                Icon(
                    name="cherry blossom",
                    palette={"P": "#f8a5c2", "W": "#fff0f5", "B": "#8d6e63"},
                    pixels=(
                        "...........",
                        ".P.......P.",
                        "PWP.....PWP",
                        ".P.....B.P.",
                        "......B....",
                        "..P..B.....",
                        ".PWPB......",
                        "..PB..P....",
                        "..B..PWP...",
                        ".B....P....",
                        "B..........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Easter",
            starts=easter,
            icons=(
                Icon(
                    name="egg",
                    palette={
                        "P": "#f7a8c4",
                        "B": "#8fd3ff",
                        "Y": "#ffe680",
                        "W": "#ffffff",
                    },
                    pixels=(
                        "....PPP....",
                        "...PPPPP...",
                        "..BBBBBBB..",
                        "..YBYBYBY..",
                        ".YYYYYYYYY.",
                        ".PPPPPPPPP.",
                        ".PWPWPWPWP.",
                        ".BBBBBBBBB.",
                        "..BBBBBBB..",
                        "..YYYYYYY..",
                        "...PPPPP...",
                        "...........",
                    ),
                ),
                Icon(
                    name="bunny",
                    palette={
                        "W": "#f5f5f5",
                        "P": "#f7a8c4",
                        "K": "#333333",
                        "S": "#c8c8c8",
                    },
                    pixels=(
                        "..WW...WW..",
                        "..WP...PW..",
                        "..WP...PW..",
                        "..WP...PW..",
                        "..WWWWWWW..",
                        ".WWWWWWWWW.",
                        ".WKWWWWWKW.",
                        ".WWWWPWWWW.",
                        "..WWWWWWW..",
                        ".WWWWWWWWW.",
                        "WWWWWWWWWWW",
                        ".SS.....SS.",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Passover",
            starts=passover,
            days=8,
            icons=(
                Icon(
                    name="matzah",
                    palette={"T": "#e8c890", "D": "#a8763e"},
                    pixels=(
                        "...........",
                        "TTTTTTTTTTT",
                        "TDTTDTTDTTT",
                        "TTTTTTTTTDT",
                        "TTDTTDTTTTT",
                        "TTTTTTTTTTT",
                        "TDTTTDTTDTT",
                        "TTTTDTTTTTT",
                        "TTDTTTTDTDT",
                        "TTTTTTTTTTT",
                        "...........",
                        "...........",
                    ),
                ),
                Icon(
                    name="wine cup",
                    palette={
                        "P": "#7b1fa2",
                        "p": "#b05ad8",
                        "S": "#c0c7cf",
                        "s": "#8e98a3",
                    },
                    pixels=(
                        "...........",
                        ".SSSSSSSSS.",
                        ".SPPPPPPPS.",
                        ".SPpPPPPPS.",
                        "..SPPPPPS..",
                        "...SSSSS...",
                        ".....S.....",
                        ".....s.....",
                        "....SSS....",
                        "..SSSSSSS..",
                        "...........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Vaisakhi",
            starts=vaisakhi,
            icons=(
                Icon(
                    name="wheat sheaf",
                    palette={"Y": "#e8c05a", "G": "#b8902a", "O": "#ff9933"},
                    pixels=(
                        ".Y.Y.Y.Y.Y.",
                        "YGYGYGYGYGY",
                        ".Y.YYYYY.Y.",
                        "..YYGYGYY..",
                        "...YYYYY...",
                        "....GYG....",
                        "...OOOOO...",
                        "....GYG....",
                        "...YGYGY...",
                        "..Y.GYG.Y..",
                        ".Y..G.G..Y.",
                        "...........",
                    ),
                ),
                Icon(
                    name="dhol drum",
                    palette={
                        "O": "#ff9933",
                        "Y": "#ffd23f",
                        "W": "#f2f2f2",
                        "T": "#8b5a2b",
                    },
                    pixels=(
                        "T.........T",
                        ".T.......T.",
                        "..YYYYYYY..",
                        ".YOWOOOWOY.",
                        "YOOOWOWOOOY",
                        "YOOOOWOOOOY",
                        "YOOOWOWOOOY",
                        ".YOWOOOWOY.",
                        "..YYYYYYY..",
                        "...........",
                        "...........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Eid al-Fitr",
            starts=eid_al_fitr,
            icons=(
                Icon(
                    name="crescent + star",
                    palette={"Y": "#ffd23f"},
                    pixels=(
                        "....YYYY...",
                        "..YYYY.....",
                        ".YYY.......",
                        "YYY.....Y..",
                        "YYY....YYY.",
                        "YYY.....Y..",
                        "YYY........",
                        ".YYY.......",
                        "..YYYY.....",
                        "....YYYY...",
                        "...........",
                        "...........",
                    ),
                ),
                Icon(
                    name="fanous",
                    palette={
                        "Y": "#e6b422",
                        "T": "#26a69a",
                        "t": "#80cbc4",
                        "W": "#fff3b0",
                    },
                    pixels=(
                        ".....Y.....",
                        "....YYY....",
                        "...YYYYY...",
                        "..YTTTTTY..",
                        "..YTtWtTY..",
                        "..YTWWWTY..",
                        "..YTtWtTY..",
                        "..YTTTTTY..",
                        "...YYYYY...",
                        "....YYY....",
                        ".....Y.....",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Vesak",
            starts=vesak,
            icons=(
                Icon(
                    name="lotus",
                    palette={
                        "P": "#f48fb1",
                        "p": "#fce4ec",
                        "G": "#4caf50",
                        "Y": "#ffd23f",
                    },
                    pixels=(
                        "...........",
                        ".....P.....",
                        "....PpP....",
                        ".P..PpP..P.",
                        ".PP.PpP.PP.",
                        ".PpPPpPPpP.",
                        "..PpPYPpP..",
                        "...PPPPP...",
                        ".GGGGGGGGG.",
                        "GGG.....GGG",
                        "...........",
                        "...........",
                    ),
                ),
                Icon(
                    name="star lantern",
                    palette={
                        "Y": "#ffd23f",
                        "R": "#e53935",
                        "W": "#fff6c8",
                        "O": "#ff8c1a",
                    },
                    pixels=(
                        ".....Y.....",
                        ".....R.....",
                        "....RRR....",
                        "....RWR....",
                        "RRRRWWWRRRR",
                        ".RRWWWWWRR.",
                        "..RRWWWRR..",
                        "..RRRRRRR..",
                        ".RRR...RRR.",
                        ".RR.....RR.",
                        ".O.......O.",
                        ".O.......O.",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Juneteenth",
            starts=partial(_annual, month=6, day=19),
            icons=(
                Icon(
                    name="Juneteenth flag",
                    palette={
                        "B": "#2b4c9b",
                        "R": "#d7263d",
                        "W": "#ffffff",
                        "w": "#b8cdf5",
                    },
                    pixels=(
                        "...........",
                        "BBwBwBwBBBB",
                        "BBBBWBBBBBB",
                        "BwBWWWBwBBB",
                        "BBBWBWBBBBB",
                        "BBwBwBwBBBB",
                        "BBBRRRRRBBB",
                        "BRRRRRRRRRB",
                        "RRRRRRRRRRR",
                        "RRRRRRRRRRR",
                        "RRRRRRRRRRR",
                        "...........",
                    ),
                ),
                Icon(
                    name="red velvet cake",
                    palette={"R": "#b22234", "W": "#fff5f5"},
                    pixels=(
                        "...........",
                        "..........W",
                        "........WWW",
                        "......WWWWW",
                        "....WWWWWWW",
                        "..WWWWWWWWW",
                        "RRRRRRRRRRR",
                        "RRRRRRRRRRR",
                        "WWWWWWWWWWW",
                        "RRRRRRRRRRR",
                        "RRRRRRRRRRR",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="First day of summer",
            starts=summer,
            icons=(
                Icon(
                    name="sun",
                    palette={"Y": "#ffd23f", "O": "#ff8c1a"},
                    pixels=(
                        ".....O.....",
                        ".O...O...O.",
                        "..O.....O..",
                        "....YYY....",
                        "...YYYYY...",
                        "OO.YYYYY.OO",
                        "...YYYYY...",
                        "....YYY....",
                        "..O.....O..",
                        ".O...O...O.",
                        ".....O.....",
                        "...........",
                    ),
                ),
                Icon(
                    name="ice pop",
                    palette={
                        "R": "#ff5252",
                        "O": "#ff8c1a",
                        "Y": "#ffd23f",
                        "T": "#d9a35b",
                    },
                    pixels=(
                        "....RRR....",
                        "...RRRRR...",
                        "...RRRRR...",
                        "...OOOOO...",
                        "...OOOOO...",
                        "...YYYYY...",
                        "...YYYYY...",
                        "....YYY....",
                        ".....T.....",
                        ".....T.....",
                        ".....T.....",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Independence Day",
            starts=partial(_annual, month=7, day=4),
            icons=(
                Icon(
                    name="flag",
                    palette={"R": "#d7263d", "W": "#f2f2f2", "B": "#2b4c9b"},
                    pixels=(
                        "...........",
                        "BWBWBRRRRRR",
                        "BBWBBWWWWWW",
                        "BWBWBRRRRRR",
                        "BBWBBWWWWWW",
                        "BWBWBRRRRRR",
                        "WWWWWWWWWWW",
                        "RRRRRRRRRRR",
                        "WWWWWWWWWWW",
                        "RRRRRRRRRRR",
                        "...........",
                        "...........",
                    ),
                ),
                Icon(
                    name="rocket",
                    palette={
                        "R": "#d7263d",
                        "W": "#f2f2f2",
                        "B": "#2b4c9b",
                        "Y": "#ffd23f",
                        "O": "#ff8c1a",
                    },
                    pixels=(
                        ".....R.....",
                        "....RRR....",
                        "....WWW....",
                        "....BWB....",
                        "....WWW....",
                        "....RRR....",
                        "....RRR....",
                        "...RRRRR...",
                        "....YOY....",
                        "...Y.O.Y...",
                        "..Y..Y..Y..",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Eid al-Adha",
            starts=eid_al_adha,
            icons=(
                Icon(
                    name="mosque",
                    palette={"T": "#26a69a", "Y": "#ffd23f", "W": "#f0e6d2"},
                    pixels=(
                        ".....Y.....",
                        ".Y...Y...Y.",
                        ".T...Y...T.",
                        ".W..TTT..W.",
                        ".W.TTTTT.W.",
                        ".W.TTTTT.W.",
                        ".WWWWWWWWW.",
                        ".WWWWWWWWW.",
                        ".WW.WWW.WW.",
                        ".WW.W.W.WW.",
                        ".WWWWWWWWW.",
                        "...........",
                    ),
                ),
                Icon(
                    name="crescent over dunes",
                    palette={"Y": "#ffd23f", "S": "#e0b872", "s": "#b8873e"},
                    pixels=(
                        "......YYY..",
                        ".Y...YY....",
                        ".....Y.....",
                        ".....YY....",
                        "......YYY..",
                        "...........",
                        "...SSS.....",
                        ".SSSSSSS.ss",
                        "SSSSSSSssss",
                        "SSSSSsssssS",
                        "sssssssSSSS",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="First day of fall",
            starts=fall,
            icons=(
                Icon(
                    name="maple leaf",
                    palette={"R": "#d84315", "O": "#ff8c1a", "B": "#6d4c41"},
                    pixels=(
                        ".....R.....",
                        "....RRR....",
                        ".R.RRORR.R.",
                        ".RRRRORRRR.",
                        "RRRRRORRRRR",
                        ".RRROOORRR.",
                        "..RRRORRR..",
                        ".RRRRORRRR.",
                        "....ROR....",
                        ".....B.....",
                        ".....B.....",
                        "...........",
                    ),
                ),
                Icon(
                    name="acorn",
                    palette={
                        "C": "#8b5a2b",
                        "c": "#5d4037",
                        "T": "#d9a35b",
                        "t": "#b07a3a",
                    },
                    pixels=(
                        ".....c.....",
                        ".....c.....",
                        "...CCCCC...",
                        "..CcCcCcC..",
                        ".CCCCCCCCC.",
                        "..TTTTTTT..",
                        "..TTTTTTt..",
                        "..TTTTTTt..",
                        "...TTTTt...",
                        "....TTt....",
                        ".....t.....",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Mid-Autumn Festival",
            starts=mid_autumn,
            icons=(
                Icon(
                    name="mooncake",
                    palette={"G": "#d9a441", "D": "#9c6a2a"},
                    pixels=(
                        "...GGGGG...",
                        ".GGGDGDGGG.",
                        ".GDGGGGGDG.",
                        "GGGDDDDDGGG",
                        "GDGDGGGDGDG",
                        "GGGDGDGDGGG",
                        "GDGDGGGDGDG",
                        "GGGDDDDDGGG",
                        ".GDGGGGGDG.",
                        ".GGGDGDGGG.",
                        "...GGGGG...",
                        "...........",
                    ),
                ),
                Icon(
                    name="moon rabbit",
                    palette={"Y": "#fff2a8", "S": "#cbb866", "P": "#ff8fa3"},
                    pixels=(
                        "...YYYYY...",
                        ".YYYYYYYYY.",
                        "YYYYSYSYYYY",
                        "YYYYSYSYYYY",
                        "YYYYSSSYYYY",
                        "YYYSSPSSYYY",
                        "YYYSSSSSYYY",
                        "YYSSSSSSSYY",
                        ".YYSSSSSYY.",
                        "..YYYYYYY..",
                        "...........",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Rosh Hashanah",
            starts=rosh_hashanah,
            icons=(
                Icon(
                    name="apple + honey",
                    palette={
                        "R": "#d32f2f",
                        "r": "#ff6b6b",
                        "G": "#4cbb5a",
                        "T": "#8b5a2b",
                        "Y": "#f5b301",
                    },
                    pixels=(
                        ".....T.G...",
                        ".....TGG...",
                        "..RRRTRRR..",
                        ".RrRRRRRRR.",
                        "YYYYYYYYYYY",
                        "RYRRYRRRYRR",
                        "RYRRRRRRYRR",
                        "RRRRRRRRRRR",
                        ".RRRRRRRRR.",
                        "..RRRRRRR..",
                        "...RR.RR...",
                        "...........",
                    ),
                ),
                Icon(
                    name="shofar",
                    palette={"I": "#f0e2c0", "B": "#a0703e"},
                    pixels=(
                        ".........I.",
                        "........II.",
                        "........I..",
                        ".......II..",
                        "......III..",
                        ".....IIII..",
                        "....IIIII..",
                        "..IIIIIII..",
                        "IIIIIIII...",
                        "BIIIIII....",
                        "BBII.......",
                        "B..........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Halloween",
            starts=partial(_annual, month=10, day=31),
            icons=(
                Icon(
                    name="jack-o-lantern",
                    palette={"O": "#e8761c", "G": "#4caf50", "Y": "#ffd23f"},
                    pixels=(
                        ".....GG....",
                        "....GG.....",
                        "..OOOOOOO..",
                        ".OOOOOOOOO.",
                        "OOOYOOOYOOO",
                        "OOYYOOOYYOO",
                        "OOOOOYOOOOO",
                        "OOOOOOOOOOO",
                        "OYYOYYYOYYO",
                        "OOYYYYYYYOO",
                        ".OOOYOYOOO.",
                        "..OOOOOOO..",
                    ),
                ),
                Icon(
                    name="boo ghost",
                    palette={
                        "W": "#f5f5f5",
                        "S": "#c8d0dc",
                        "K": "#222222",
                        "R": "#e0405a",
                    },
                    pixels=(
                        "...WWWWW...",
                        ".WWWWWWWWW.",
                        "WWWWKWWKWWW",
                        "WWWWKWWKWWW",
                        "WWWWWWWWWWW",
                        "WWKKKKKKKWW",
                        "WWKRRRRRKWW",
                        "WWWKRRRKWWW",
                        "WWWWRRRWWWS",
                        "SWWWWWWWWS.",
                        ".SSWWWWWSS.",
                        "...SSSSS...",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Diwali",
            starts=diwali,
            icons=(
                Icon(
                    name="diya",
                    palette={
                        "T": "#c1663f",
                        "D": "#8a3f22",
                        "Y": "#ffd23f",
                        "O": "#ff8c1a",
                        "W": "#fff6c8",
                    },
                    pixels=(
                        ".....Y.....",
                        "....YOY....",
                        "....OWO....",
                        "....YWY....",
                        ".....Y.....",
                        "...........",
                        "TTT.....TTT",
                        "TTTTTTTTTTT",
                        ".TTDTDTDTT.",
                        "..TTTTTTT..",
                        "....TTT....",
                        "...........",
                    ),
                ),
                Icon(
                    name="rangoli",
                    palette={
                        "M": "#e91e8c",
                        "O": "#ff8c1a",
                        "T": "#00a6a6",
                        "Y": "#ffd600",
                    },
                    pixels=(
                        ".....M.....",
                        "....MYM....",
                        "...TMOMT...",
                        "..T.OYO.T..",
                        ".MMOYTYOMM.",
                        "MYOYTMTYOYM",
                        ".MMOYTYOMM.",
                        "..T.OYO.T..",
                        "...TMOMT...",
                        "....MYM....",
                        ".....M.....",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Thanksgiving",
            starts=thanksgiving,
            icons=(
                Icon(
                    name="turkey",
                    palette={
                        "R": "#d7263d",
                        "O": "#ff8c1a",
                        "Y": "#ffd23f",
                        "b": "#8b5a2b",
                        "h": "#c08850",
                        "K": "#222222",
                        "y": "#ffb300",
                        "r": "#e0262f",
                    },
                    pixels=(
                        "..RROOORR..",
                        ".ROOYYYOOR.",
                        "ROOYbbbYOOR",
                        "ROYbhhhbYOR",
                        "ROYbKhKbYOR",
                        ".RYbhyhbYR.",
                        "..bbhrhbb..",
                        ".bbbbbbbbb.",
                        ".bbbbbbbbb.",
                        "..bbbbbbb..",
                        "...y...y...",
                        "..yy...yy..",
                    ),
                ),
                Icon(
                    name="pumpkin pie",
                    palette={
                        "C": "#d9a35b",
                        "D": "#a8702e",
                        "O": "#f08a24",
                        "o": "#c8641a",
                        "W": "#fff8ee",
                    },
                    pixels=(
                        "...........",
                        "...........",
                        "...........",
                        "......WW..C",
                        ".....WWWWCC",
                        "....OOOOOCC",
                        "..OOOOOOOCC",
                        "OOOOOOOOOCC",
                        "oooooooooCC",
                        "oooooooooCC",
                        "DDDDDDDDDDD",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Hanukkah",
            starts=hanukkah,
            days=8,
            icons=(
                Icon(
                    name="menorah",
                    palette={"G": "#e6b422", "Y": "#ffd23f", "B": "#4a90e2"},
                    pixels=(
                        ".....Y.....",
                        ".YYYYBYYYY.",
                        ".BBBBBBBBB.",
                        ".BBBBBBBBB.",
                        ".GGGGGGGGG.",
                        "..GG.G.GG..",
                        "...GGGGG...",
                        ".....G.....",
                        ".....G.....",
                        "...GGGGG...",
                        "...........",
                        "...........",
                    ),
                ),
                Icon(
                    name="dreidel",
                    palette={
                        "B": "#2b6cb0",
                        "b": "#5a9bd8",
                        "S": "#c0c7cf",
                        "W": "#ffffff",
                    },
                    pixels=(
                        ".....S.....",
                        ".....S.....",
                        "..BBBBBBB..",
                        "..BbbbbbB..",
                        "..BbWWWbB..",
                        "..BbbbWbB..",
                        "..BbWbWbB..",
                        "..BbWbWbB..",
                        "...BbbbB...",
                        "....BbB....",
                        ".....B.....",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="First day of winter",
            starts=winter,
            icons=(
                Icon(
                    name="snowflake",
                    palette={"W": "#ffffff", "I": "#8fd3ff"},
                    pixels=(
                        ".....W.....",
                        "...I.W.I...",
                        "....IWI....",
                        ".I...W...I.",
                        "..I..W..I..",
                        "WWWWWIWWWWW",
                        "..I..W..I..",
                        ".I...W...I.",
                        "....IWI....",
                        "...I.W.I...",
                        ".....W.....",
                        "...........",
                    ),
                ),
                Icon(
                    name="snowman",
                    palette={
                        "W": "#f5f5f5",
                        "K": "#4a4a4a",
                        "O": "#ff8c1a",
                        "R": "#d32f2f",
                    },
                    pixels=(
                        "...KKKKK...",
                        "...KKKKK...",
                        "..KKKKKKK..",
                        "...WWWWW...",
                        "...WKWKW...",
                        "...WWOOW...",
                        "..RRRRRRR..",
                        "..WWWRWWW..",
                        ".WWWWKWWWW.",
                        ".WWWWWWWWW.",
                        ".WWWWKWWWW.",
                        "..WWWWWWW..",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Christmas",
            starts=partial(_annual, month=12, day=25),
            icons=(
                Icon(
                    name="tree",
                    palette={
                        "G": "#2e8b3e",
                        "Y": "#ffd23f",
                        "R": "#e53935",
                        "B": "#42a5f5",
                        "T": "#8b5a2b",
                    },
                    pixels=(
                        ".....Y.....",
                        "....YYY....",
                        ".....G.....",
                        "....GRG....",
                        "...GGGGG...",
                        "..GBGGGRG..",
                        "...GGGGG...",
                        "..GGYGGGG..",
                        ".GRGGGBGGG.",
                        "GGGGGGGGGRG",
                        "....TTT....",
                        "....TTT....",
                    ),
                ),
                Icon(
                    name="gift",
                    palette={"R": "#d32f2f", "D": "#a82020", "Y": "#ffd23f"},
                    pixels=(
                        "...YY.YY...",
                        "..Y..Y..Y..",
                        "...YYYYY...",
                        "RRRRRYRRRRR",
                        "RRRRRYRRRRR",
                        "YYYYYYYYYYY",
                        ".RRRRYRRRR.",
                        ".RRRRYRRRR.",
                        ".RRRRYRRRR.",
                        ".RRRRYRRRR.",
                        ".DDDDYDDDD.",
                        "...........",
                    ),
                ),
            ),
        ),
        Holiday(
            name="Kwanzaa",
            starts=partial(_annual, month=12, day=26),
            days=7,
            icons=(
                Icon(
                    name="kinara",
                    palette={
                        "Y": "#ffd23f",
                        "R": "#e53935",
                        "r": "#9e1b1b",
                        "K": "#5a5a5a",
                        "G": "#43a047",
                        "g": "#1b5e20",
                        "T": "#8b5a2b",
                    },
                    pixels=(
                        ".....Y.....",
                        "....YKY....",
                        "...YRKGY...",
                        "..YrRKGgY..",
                        "..RrRKGgG..",
                        "..RrRKGgG..",
                        ".TTTTTTTTT.",
                        ".TTTTTTTTT.",
                        "..T.....T..",
                        "..T.....T..",
                        ".TTT...TTT.",
                        "...........",
                    ),
                ),
                Icon(
                    name="corn",
                    palette={"Y": "#ffd23f", "y": "#e0a800", "G": "#6aa84f"},
                    pixels=(
                        "....YyY....",
                        "...YyYyY...",
                        "...yYyYy...",
                        "...YyYyY...",
                        "...yYyYy...",
                        ".G.YyYyY.G.",
                        ".GGyYyYyGG.",
                        "..GGYyYGG..",
                        "...GGGGG...",
                        "....GGG....",
                        ".....G.....",
                        "...........",
                    ),
                ),
            ),
        ),
    )


def spring(year: int) -> tuple[date, ...]:
    """Return the March equinox's UTC date."""
    return (_utc_date(_season(year, 0), utc_hours=0),)


def summer(year: int) -> tuple[date, ...]:
    """Return the June solstice's UTC date."""
    return (_utc_date(_season(year, 1), utc_hours=0),)


def fall(year: int) -> tuple[date, ...]:
    """Return the September equinox's UTC date."""
    return (_utc_date(_season(year, 2), utc_hours=0),)


def winter(year: int) -> tuple[date, ...]:
    """Return the December solstice's UTC date."""
    return (_utc_date(_season(year, 3), utc_hours=0),)


def nowruz(year: int) -> tuple[date, ...]:
    """Return the March equinox's day in Tehran, or the next if after noon there."""
    return (_utc_date(_season(year, 0), utc_hours=3.5 + 12),)


def lunar_new_year(year: int) -> tuple[date, ...]:
    """Return the first new moon on or after January 21, in China time."""
    return (_new_or_full_moon(date(year, 1, 21), full=False, utc_hours=8),)


def mid_autumn(year: int) -> tuple[date, ...]:
    """Return day 15 of the month begun by the last new moon before the equinox.

    Args:
      year: The Gregorian year.

    Returns:
      days: The Mid-Autumn Festival's date.

    """
    # China time. Searching from 29 days before skips a new moon on the
    # equinox's own day, which starts the next month: in 2033 that keeps the
    # festival on September 8, not October 8. In 1990-2099 no 30-day gap puts
    # the first new moon of the search on the equinox itself.
    equinox = _utc_date(_season(year, 2), utc_hours=8)
    new_moon = _new_or_full_moon(equinox - timedelta(days=29), full=False, utc_hours=8)
    return (new_moon + timedelta(days=14),)


def holi(year: int) -> tuple[date, ...]:
    """Return the morning after the bonfire night of the Phalguna full moon."""
    # The full moon ends its lunar day. Ending after 17:00 India time, that day
    # holds the evening bonfire, so Holi is the next day: +7 h rolls over then.
    return (_new_or_full_moon(date(year, 2, 28), full=True, utc_hours=5.5 + 7),)


def diwali(year: int) -> tuple[date, ...]:
    """Return the evening of prayers in the Kartika new moon's lunar day."""
    # The new moon ends its lunar day. Ending after 18:00 India time, that day
    # holds the evening prayers; otherwise the evening before does: -18 h.
    return (_new_or_full_moon(date(year, 10, 17), full=False, utc_hours=5.5 - 18),)


def vaisakhi(year: int) -> tuple[date, ...]:
    """Return the sun's entry into sidereal Aries (Lahiri), in India time."""
    ayanamsa_deg = 24.22 + 0.013_97 * (year - 2026)
    sun_deg_per_day = 0.985
    jde = _season(year, 0) + ayanamsa_deg / sun_deg_per_day
    return (_utc_date(jde, utc_hours=5.5),)


def rosh_hashanah(year: int) -> tuple[date, ...]:
    """Return 1 Tishrei of the Hebrew year that begins in ``year``."""
    return (date.fromordinal(_hebrew_new_year(year + 3761)),)


def passover(year: int) -> tuple[date, ...]:
    """Return 15 Nisan, always 163 days before the next Rosh Hashanah."""
    return (date.fromordinal(_hebrew_new_year(year + 3761) - 163),)


def hanukkah(year: int) -> tuple[date, ...]:
    """Return 25 Kislev, after a Cheshvan of 29 days, or 30 in a complete year."""
    hebrew_year = year + 3761
    new_year = _hebrew_new_year(hebrew_year)
    complete = (_hebrew_new_year(hebrew_year + 1) - new_year) % 10 == 5
    return (date.fromordinal(new_year + 30 + (30 if complete else 29) + 24),)


def eid_al_fitr(year: int) -> tuple[date, ...]:
    """Return 1 Shawwal: Singapore's announced dates, else the arithmetic calendar.

    Args:
      year: The Gregorian year.

    Returns:
      days: Eid al-Fitr's dates; some years hold two.

    """
    return _listed(
        year,
        dates=(
            "2026-03-21",
            "2027-03-10",
            "2028-02-27",
            "2029-02-15",
            "2030-02-04",
            "2031-01-25",
            "2032-01-14",
            "2033-01-03",
            "2033-12-23",
            "2034-12-12",
            "2035-12-01",
        ),
        otherwise=partial(_islamic, month=10, day=1),
    )


def eid_al_adha(year: int) -> tuple[date, ...]:
    """Return 10 Dhu al-Hijjah: Singapore's announced dates, else arithmetic.

    Args:
      year: The Gregorian year.

    Returns:
      days: Eid al-Adha's dates; some years hold two.

    """
    return _listed(
        year,
        dates=(
            "2026-05-27",
            "2027-05-17",
            "2028-05-05",
            "2029-04-25",
            "2030-04-14",
            "2031-04-03",
            "2032-03-22",
            "2033-03-12",
            "2034-03-01",
            "2035-02-19",
        ),
        otherwise=partial(_islamic, month=12, day=10),
    )


def vesak(year: int) -> tuple[date, ...]:
    """Return Singapore's Vesak Day, else the first full moon from May 8.

    Args:
      year: The Gregorian year.

    Returns:
      days: Vesak's date.

    """
    return _listed(
        year,
        dates=(
            "2026-05-31",
            "2027-05-20",
            "2028-05-09",
            "2029-05-27",
            "2030-05-16",
            "2031-06-04",
            "2032-05-23",
            "2033-05-13",
            "2034-06-01",
            "2035-05-22",
        ),
        otherwise=_vesak_estimate,
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
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
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
    """Whether ``today`` falls from the weekday before a start to its last day."""
    for year in (today.year - 1, today.year, today.year + 1):
        for start in holiday.starts(year):
            opens = start - timedelta(days=1)
            while opens.weekday() >= 5:  # Saturday or Sunday.
                opens -= timedelta(days=1)
            if opens <= today < start + timedelta(days=holiday.days):
                return True
    return False


def _annual(year: int, *, month: int, day: int) -> tuple[date, ...]:
    return (date(year, month, day),)


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


def _vesak_estimate(year: int) -> tuple[date, ...]:
    return (_new_or_full_moon(date(year, 5, 8), full=True, utc_hours=8),)


def _utc_date(jde: float, *, utc_hours: float) -> date:
    """Return the civil date of Julian day ``jde`` at ``utc_hours`` from UTC."""
    return date.fromordinal(math.floor(jde + 0.5 + utc_hours / 24) - 1_721_425)


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


def _new_or_full_moon(start: date, *, full: bool, utc_hours: float) -> date:
    """Return the first new (or full) moon's local date on or after ``start``."""
    lunation = 29.530_588_861
    jd = start.toordinal() + 1_721_424.5 - utc_hours / 24
    k = math.floor((jd - 2_451_550.097_66) / lunation) - 1 + (0.5 if full else 0)
    while (day := _utc_date(_moon_phase(k), utc_hours=utc_hours)) < start:
        k += 1
    return day


def _moon_phase(k: float) -> float:
    """Return the Julian day of new moon ``k`` (integer) or full moon (half)."""
    t = k / 1236.85
    jde = 2_451_550.097_66 + 29.530_588_861 * k + 0.000_154_37 * t * t
    e = 1 - 0.002_516 * t
    sun = math.radians(2.5534 + 29.105_356_70 * k)
    moon = math.radians(201.5643 + 385.816_935_28 * k)
    node = math.radians(160.7108 + 390.670_502_84 * k)
    c = (
        (-0.406_14, 0.173_02, 0.016_14, 0.010_43, 0.007_34, -0.005_15, 0.002_09)
        if k % 1
        else (-0.407_20, 0.172_41, 0.016_08, 0.010_39, 0.007_39, -0.005_14, 0.002_08)
    )
    return jde + (
        c[0] * math.sin(moon)
        + c[1] * e * math.sin(sun)
        + c[2] * math.sin(2 * moon)
        + c[3] * math.sin(2 * node)
        + c[4] * e * math.sin(moon - sun)
        + c[5] * e * math.sin(moon + sun)
        + c[6] * e * e * math.sin(2 * sun)
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
    parts = 12_084 + 13_753 * months
    days = 29 * months + parts // 25_920
    return days + 1 if (3 * (days + 1)) % 7 < 3 else days


def _islamic(year: int, *, month: int, day: int) -> tuple[date, ...]:
    """Return each ``month``/``day`` of the arithmetic Islamic calendar in ``year``."""
    first = (year - 622) * 33 // 32
    days = (
        date.fromordinal(
            day
            + 29 * (month - 1)
            + (6 * month - 1) // 11
            + (hijri - 1) * 354
            + (3 + 11 * hijri) // 30
            + 227_014,
        )
        for hijri in range(first - 1, first + 3)
    )
    return tuple(d for d in days if d.year == year)
