"""Static report figures (SVG): depth curve, reliability diagram, kappa sweep.

Two series at most (internal vs external), so the first two slots of the
validated categorical palette are used. Ink stays in text colours; the grid is
recessive. SVG rather than PNG so figures are small, diffable and outside the
repo's blanket raster-image ignore.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcai.pipeline import EvaluationResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"internal": "#2a78d6", "external": "#eb6834"}


def _axes(title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    return fig, ax


def depth_curve(r: EvaluationResult, path: Path) -> None:
    """Headline figure: tooth-level sensitivity by lesion depth, internal vs external."""
    fig, ax = _axes("Sensitivity by lesion depth (tooth level, 95% CI)", "sensitivity")
    names = [s.name for s in r.internal.stratified.strata]
    sets = [("internal", r.internal.stratified)]
    if r.external is not None:
        sets.append(("external", r.external.stratified))
    offsets = (-0.08, 0.08) if len(sets) == 2 else (0.0,)
    for offset, (label, rep) in zip(offsets, sets, strict=True):
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(rep.strata):
            if s.sensitivity is None:
                continue
            xs.append(i + offset)
            ys.append(s.sensitivity.value)
            lo.append(s.sensitivity.value - s.sensitivity.lo)
            hi.append(s.sensitivity.hi - s.sensitivity.value)
        color = SERIES[label]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=color, ecolor=color, elinewidth=2,
                    linewidth=2, markersize=7, capsize=0, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=9)
    ax.set_xticks(range(len(names)), names, color=INK)
    ax.set_xlim(-0.4, len(names) - 0.4)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=SURFACE)
    plt.close(fig)


def reliability(r: EvaluationResult, path: Path) -> None:
    fig, ax = _axes("Reliability, per-tooth P(lesion)", "observed lesion rate")
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=1, label="perfect calibration")
    for label, s in (("internal", r.internal), ("external", r.external)):
        if s is None:
            continue
        bins = s.calibration.curve.bins
        ax.plot([b.mean_predicted for b in bins], [b.observed for b in bins], "o-",
                color=SERIES[label], linewidth=2, markersize=6, label=label)
    ax.set_xlabel("mean predicted P(lesion)", color=INK_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=SURFACE)
    plt.close(fig)


def kappa_sweep(r: EvaluationResult, path: Path) -> bool:
    """Model kappa across thresholds against the human ceiling band. False if not computable."""
    sw = r.agreement.sweep
    if sw is None:
        return False
    fig, ax = _axes("Agreement with readers across thresholds (tooth level)", "mean Cohen's κ")
    ax.axhspan(sw.ceiling.lo, sw.ceiling.hi, color=GRID, zorder=0)
    ax.axhline(sw.ceiling.value, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate("human ceiling (LOO consensus, 95% CI)", (0.99, sw.ceiling.hi),
                xytext=(0, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=9)
    ax.plot(sw.thresholds, sw.model_kappa, color=SERIES["internal"], linewidth=2)
    for x, label in ((sw.operating_threshold, "operating point"),
                     (sw.best_threshold, "κ-optimal")):
        y = float(np.interp(x, sw.thresholds, sw.model_kappa))
        ax.plot([x], [y], "o", color=SERIES["internal"], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{label}\n{x:.2f}", (x, y), xytext=(6, -24), textcoords="offset points",
                    color=INK_2, fontsize=9)
    ax.set_xlabel("threshold on P(lesion)", color=INK_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(min(0.0, float(np.nanmin(sw.model_kappa))), 1)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=SURFACE)
    plt.close(fig)
    return True


def write_figures(r: EvaluationResult, out_dir: Path) -> dict[str, str]:
    """Write all figures; return paths relative to the report's directory."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    depth_curve(r, fig_dir / "depth_sensitivity.svg")
    reliability(r, fig_dir / "reliability.svg")
    out = {"depth": "figures/depth_sensitivity.svg", "reliability": "figures/reliability.svg"}
    if kappa_sweep(r, fig_dir / "kappa_sweep.svg"):
        out["sweep"] = "figures/kappa_sweep.svg"
    return out
