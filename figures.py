"""Static figures for the README. GitHub does not render interactive charts."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import survival as sv

SURFACE, INK, INK_SOFT, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Categorical slots 1-3 of the reference palette, checked with the palette
# validator. Colour follows the entity: blue is always football, orange is
# always the NBA, aqua is football in recent seasons.
FOOTBALL, NBA, RECENT = "#2a78d6", "#eb6834", "#1baf7a"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def _strip(ax, xgrid=False, ygrid=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    # Passing style arguments switches a grid on, so style only the ones wanted.
    for axis, on in ((ax.xaxis, xgrid), (ax.yaxis, ygrid)):
        if on:
            axis.grid(True, color=GRID, linewidth=0.8)
        else:
            axis.grid(False)
    ax.set_axisbelow(True)


def _header(fig, title, subtitle, source, top):
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9.5,
             color=INK_SOFT, linespacing=1.45)
    fig.text(0.012, 0.008, source, ha="left", va="bottom", fontsize=8.5, color=INK_SOFT)
    fig.subplots_adjust(top=top)


def _returned(km: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Share returned on each day of a grid, read off a Kaplan-Meier table."""
    t = np.concatenate([[0.0], km.time.values])
    s = np.concatenate([[1.0], km.surv.values])
    idx = np.searchsorted(t, grid, side="right") - 1
    return 1 - s[idx]


def return_curves(panels: list[dict], out, horizon=730, title="", subtitle="", source=""):
    """
    One panel per injury. Each sport is a line (the data as recorded) and a
    wash reaching to its other calendar version, so the width of the wash is
    the uncertainty the calendar adds.
    """
    grid = np.arange(0, horizon + 1)
    fig, axes = plt.subplots(1, len(panels), figsize=(13.5, 4.9), dpi=200, sharey=True)
    for ax, p in zip(axes, panels):
        for key, color in (("football", FOOTBALL), ("nba", NBA)):
            main = _returned(sv.km_curve(*p[key]), grid)
            alt = _returned(sv.km_curve(*p[key + "_alt"]), grid)
            ax.fill_between(grid, np.minimum(main, alt), np.maximum(main, alt),
                            color=color, alpha=0.14, linewidth=0, step="post", zorder=1)
            ax.step(grid, main, where="post", color=color, linewidth=2.0, zorder=3)
            ax.step(grid, alt, where="post", color=color, linewidth=0.9, zorder=2)
        ax.axvline(365, color=AXIS, linewidth=0.8, zorder=0)
        ax.text(369, 0.02, "1 year", fontsize=8.5, color=MUTED)
        _strip(ax)
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 180, 365, 545, 730])
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.25)])
        ax.set_title(p["title"], loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
        ax.set_title(p["counts"], loc="right", fontsize=8.5, color=INK_SOFT, pad=9)
        ax.set_xlabel("Days since injury", fontsize=9.5, labelpad=6)
    axes[0].set_ylabel("Share back", fontsize=9.5, labelpad=8)

    handles = [
        Line2D([], [], color=FOOTBALL, linewidth=2.0, label="Football: Transfermarkt end date"),
        Patch(facecolor=FOOTBALL, alpha=0.3, label="…if summer returns wait until 31 August"),
        Line2D([], [], color=NBA, linewidth=2.0, label="NBA: first game played"),
        Patch(facecolor=NBA, alpha=0.3, label="…if summer returns were ready when the season ended"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.865), ncol=4,
               frameon=False, fontsize=8.8, handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.03, 1, 0.80))
    _header(fig, title, subtitle, source, top=0.74)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def medians(table: pd.DataFrame, out, title="", subtitle="", source=""):
    """
    Median days with 95% bootstrap intervals. table columns:
    injury, sport, version ('recorded' or 'bound'), median, low, high.
    """
    injuries = list(dict.fromkeys(table.injury))
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    offsets = {("Football", "recorded"): 0.21, ("Football", "bound"): 0.07,
               ("NBA", "recorded"): -0.07, ("NBA", "bound"): -0.21}
    colors = {"Football": FOOTBALL, "NBA": NBA}
    for i, inj in enumerate(injuries):
        y0 = len(injuries) - 1 - i
        for r in table[table.injury == inj].itertuples(index=False):
            y = y0 + offsets[(r.sport, r.version)]
            c = colors[r.sport]
            ax.plot([r.low, r.high], [y, y], color=c, linewidth=2.0, alpha=0.9, zorder=2)
            face = c if r.version == "recorded" else SURFACE
            ax.plot(r.median, y, "o", markersize=8, markerfacecolor=face, markeredgecolor=c,
                    markeredgewidth=1.8, zorder=3)
            ax.text(r.high + 8, y, f"{r.median:.0f}", va="center", fontsize=8.5, color=INK_SOFT)
    ax.set_yticks(range(len(injuries)))
    ax.set_yticklabels(list(reversed(injuries)), fontsize=10.5, color=INK)
    _strip(ax, xgrid=True, ygrid=False)
    ax.set_xlim(0, max(table.high) * 1.12)
    ax.set_ylim(-0.6, len(injuries) - 0.4)
    ax.set_xlabel("Median days to return, with 95% interval", fontsize=9.5, labelpad=6)
    handles = [
        Line2D([], [], marker="o", color=FOOTBALL, markerfacecolor=FOOTBALL, markersize=8,
               linewidth=2.0, label="Football, as recorded"),
        Line2D([], [], marker="o", color=FOOTBALL, markerfacecolor=SURFACE, markersize=8,
               markeredgewidth=1.8, linewidth=2.0, label="Football, summer returns on 31 Aug"),
        Line2D([], [], marker="o", color=NBA, markerfacecolor=NBA, markersize=8,
               linewidth=2.0, label="NBA, first game played"),
        Line2D([], [], marker="o", color=NBA, markerfacecolor=SURFACE, markersize=8,
               markeredgewidth=1.8, linewidth=2.0, label="NBA, summer returns at season's end"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.83), ncol=2,
               frameon=False, fontsize=8.8, handlelength=2.2, columnspacing=1.8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.70))
    _header(fig, title, subtitle, source, top=0.66)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def return_months(football: pd.Series, nba: pd.Series, out, title="", subtitle="", source=""):
    """In which calendar month players come back, one panel per sport."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=200, sharey=True)
    for ax, s, color, name in ((axes[0], football, FOOTBALL, "Football"),
                               (axes[1], nba, NBA, "NBA")):
        share = s.dt.month.value_counts(normalize=True).reindex(range(1, 13), fill_value=0)
        x = np.arange(12)
        ax.bar(x, share.values, width=0.56, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        for xi, v in zip(x, share.values):
            if v >= 0.12:
                ax.text(xi, v + 0.008, f"{v * 100:.0f}%", ha="center", fontsize=8.5, color=INK_SOFT)
        _strip(ax)
        ax.set_xlim(-0.6, 11.6)
        ax.set_xticks(x)
        ax.set_xticklabels(months, fontsize=9)
        ax.set_title(f"{name}  ·  {len(s)} returns", loc="left", fontsize=11.5,
                     fontweight="bold", color=INK, pad=8)
    ymax = 0.42
    axes[0].set_ylim(0, ymax)
    axes[0].set_yticks(np.arange(0, ymax + 0.001, 0.1))
    axes[0].set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, ymax + 0.001, 0.1)])
    axes[0].set_ylabel("Share of returns", fontsize=9.5, labelpad=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.80))
    _header(fig, title, subtitle, source, top=0.76)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def audit_bars(counts: pd.Series, out, title="", subtitle="", source=""):
    """Verdicts on the old episodes. Correct ones in blue, problems in orange."""
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=200)
    y = np.arange(len(counts))
    colors = [FOOTBALL if k == "correct" else NBA for k in counts.index]
    ax.barh(y, counts.values, height=0.56, color=colors, zorder=3, edgecolor=SURFACE, linewidth=1.0)
    for yi, v in zip(y, counts.values):
        ax.text(v + 0.3, yi, str(v), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index, fontsize=10, color=INK)
    _strip(ax, xgrid=True, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, counts.max() * 1.15)
    ax.set_xlabel("Episodes", fontsize=9.5, labelpad=6)
    handles = [Patch(facecolor=FOOTBALL, label="matches the game logs"),
               Patch(facecolor=NBA, label="does not")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.80), ncol=2,
               frameon=False, fontsize=8.8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.74))
    _header(fig, title, subtitle, source, top=0.70)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out


def then_and_now(panels: list[dict], out, horizon=730, title="", subtitle="", source=""):
    """Football curves for 2010-2019 against 2021-2024, one panel per injury."""
    grid = np.arange(0, horizon + 1)
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.4), dpi=200, sharey=True)
    for ax, p in zip(axes, panels):
        for key, color in (("then", FOOTBALL), ("now", RECENT)):
            km = sv.km_curve(*p[key])
            ax.step(grid, _returned(km, grid), where="post", color=color, linewidth=2.0)
        ax.axvline(365, color=AXIS, linewidth=0.8, zorder=0)
        _strip(ax)
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 180, 365, 545, 730])
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.set_yticklabels([f"{int(v * 100)}%" for v in np.arange(0, 1.01, 0.25)])
        ax.set_title(p["title"], loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
        ax.set_title(p["counts"], loc="right", fontsize=8.5, color=INK_SOFT, pad=9)
        ax.set_xlabel("Days since injury", fontsize=9.5, labelpad=6)
    axes[0].set_ylabel("Share back", fontsize=9.5, labelpad=8)
    handles = [Line2D([], [], color=FOOTBALL, linewidth=2.0, label="2010/11 - 2019/20"),
               Line2D([], [], color=RECENT, linewidth=2.0, label="2021/22 - 2024/25")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.85), ncol=2,
               frameon=False, fontsize=8.8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.78))
    _header(fig, title, subtitle, source, top=0.72)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return out
