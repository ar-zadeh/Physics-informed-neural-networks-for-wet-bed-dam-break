"""Create the focused chapter analysis directly from runs.tar.gz.

The script intentionally ignores every result directory outside the supplied
archive.  It writes publication figures and a machine-readable audit record for
the six-condition, three-seed ablation campaign.
"""
from __future__ import annotations

import csv
import io
import json
import math
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
_ARCHIVE_CANDIDATES = [HERE.parent / "runs.tar.gz", HERE.parent.parent / "runs.tar.gz"]
ARCHIVE = next((path for path in _ARCHIVE_CANDIDATES if path.exists()), _ARCHIVE_CANDIDATES[0])
OUT = HERE / "focused_figures"
AUDIT = HERE / "focused_analysis.json"

G = 9.81
HL = 1.0
HR = 0.1
XL = -10.0
XR = 10.0
TEND = 1.25
EVAL_TIME = 1.2

ORDER = [
    "jacobian_uniform",
    "jacobian_rar",
    "roe_uniform",
    "roe_jacobian_rar",
    "roe_roe_rar",
    "roe_hybrid",
]
LABEL = {
    "jacobian_uniform": "Differential + uniform",
    "jacobian_rar": "Differential + RAR",
    "roe_uniform": "Roe + uniform",
    "roe_jacobian_rar": "Roe + differential RAR",
    "roe_roe_rar": "Roe + Roe RAR",
    "roe_hybrid": "Roe + hybrid",
}
SHORT = {
    "jacobian_uniform": "Diff.-U",
    "jacobian_rar": "Diff.-RAR",
    "roe_uniform": "Roe-U",
    "roe_jacobian_rar": "Roe-J-RAR",
    "roe_roe_rar": "Roe-R-RAR",
    "roe_hybrid": "Roe-Hybrid",
}
COLORS = {
    "jacobian_uniform": "#6f6f6f",
    "jacobian_rar": "#b35c44",
    "roe_uniform": "#3b78a8",
    "roe_jacobian_rar": "#4d9b72",
    "roe_roe_rar": "#214f83",
    "roe_hybrid": "#9a6ab0",
}
MARKERS = {0: "o", 1: "s", 2: "^"}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def member_bytes(tf: tarfile.TarFile, name: str) -> bytes:
    member = tf.extractfile(name)
    if member is None:
        raise FileNotFoundError(name)
    return member.read()


def load_archive():
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        rows = list(
            csv.DictReader(
                io.StringIO(member_bytes(tf, "runs/ablation/runs.csv").decode("utf-8"))
            )
        )
        manifest = json.loads(
            member_bytes(tf, "runs/ablation/manifest.json").decode("utf-8")
        )
        predictions = {}
        anchors = {}
        for row in rows:
            condition = row["condition"]
            seed = int(row["seed"])
            root = f"runs/ablation/{condition}__seed{seed}"
            with np.load(io.BytesIO(member_bytes(tf, f"{root}/prediction_t1.2.npz"))) as npz:
                predictions[(condition, seed)] = {k: np.asarray(npz[k]) for k in npz.files}
            anchor_name = f"{root}/rar_anchors.npz"
            try:
                payload = member_bytes(tf, anchor_name)
            except (FileNotFoundError, KeyError):
                continue
            with np.load(io.BytesIO(payload)) as npz:
                anchors[(condition, seed)] = [np.asarray(npz[k]) for k in sorted(npz.files)]
    typed = []
    for row in rows:
        result = dict(row)
        for key in [
            "combined_error",
            "relative_l2_h_t1.2",
            "relative_l2_u_t1.2",
            "shock_location_error_m_t1.2",
            "mass_error_pct_t1.2",
            "elapsed_seconds",
            "lbfgs_final_loss",
        ]:
            result[key] = float(row[key])
        for key in ["seed", "shock_crossing_count", "lbfgs_function_evals"]:
            result[key] = int(row[key])
        result["metrics_all_times"] = json.loads(row["metrics_all_times"].replace("'", '"'))
        typed.append(result)
    return typed, manifest, predictions, anchors


def stoker_parameters():
    c_l = math.sqrt(G * HL)

    def rarefaction_u(h):
        return 2.0 * (c_l - math.sqrt(G * h))

    def shock_u(h):
        return (h - HR) * math.sqrt(G * (h + HR) / (2.0 * h * HR))

    h_star = brentq(lambda h: rarefaction_u(h) - shock_u(h), HR * (1 + 1e-8), HL)
    u_star = rarefaction_u(h_star)
    c_star = math.sqrt(G * h_star)
    return {
        "h_star": h_star,
        "u_star": u_star,
        "fan_head_speed": -c_l,
        "fan_tail_speed": u_star - c_star,
        "shock_speed": h_star * u_star / (h_star - HR),
    }


def exact_solution(x, t, params):
    x = np.asarray(x)
    if t <= 0:
        return np.where(x < 0, HL, HR), np.zeros_like(x)
    xi = x / t
    h = np.full_like(xi, HR, dtype=float)
    u = np.zeros_like(xi, dtype=float)
    left = xi <= params["fan_head_speed"]
    fan = (xi > params["fan_head_speed"]) & (xi <= params["fan_tail_speed"])
    middle = (xi > params["fan_tail_speed"]) & (xi <= params["shock_speed"])
    h[left] = HL
    c = (2 * math.sqrt(G * HL) - xi[fan]) / 3
    h[fan] = c**2 / G
    u[fan] = 2 * (math.sqrt(G * HL) + xi[fan]) / 3
    h[middle] = params["h_star"]
    u[middle] = params["u_star"]
    return h, u


def group(rows):
    return {condition: sorted([r for r in rows if r["condition"] == condition], key=lambda r: r["seed"])
            for condition in ORDER}


def representative_seed(condition_rows):
    values = np.asarray([r["combined_error"] for r in condition_rows])
    med = np.median(values)
    return condition_rows[int(np.argmin(np.abs(values - med)))]["seed"]


def summarize(rows):
    grouped = group(rows)
    summary = {}
    for condition, rr in grouped.items():
        def stats(key):
            a = np.asarray([r[key] for r in rr], dtype=float)
            return {"median": float(np.nanmedian(a)), "min": float(np.nanmin(a)), "max": float(np.nanmax(a))}

        valid = [r for r in rr if r["shock_crossing_count"] == 1]
        summary[condition] = {
            "label": LABEL[condition],
            "n": len(rr),
            "combined_error": stats("combined_error"),
            "depth_error_t1.2": stats("relative_l2_h_t1.2"),
            "velocity_error_t1.2": stats("relative_l2_u_t1.2"),
            "mass_error_pct_t1.2": stats("mass_error_pct_t1.2"),
            "elapsed_seconds": stats("elapsed_seconds"),
            "valid_shocks": len(valid),
            "shock_error_m": ({
                "median": float(np.median([r["shock_location_error_m_t1.2"] for r in valid])),
                "min": float(np.min([r["shock_location_error_m_t1.2"] for r in valid])),
                "max": float(np.max([r["shock_location_error_m_t1.2"] for r in valid])),
            } if valid else None),
            "representative_seed": representative_seed(rr),
        }
    comparisons = [
        ("roe_vs_differential_uniform", "jacobian_uniform", "roe_uniform"),
        ("roe_vs_differential_same_indicator_rar", "jacobian_rar", "roe_jacobian_rar"),
        ("rar_effect_differential", "jacobian_uniform", "jacobian_rar"),
        ("rar_effect_roe", "roe_uniform", "roe_roe_rar"),
        ("roe_indicator_vs_differential_indicator", "roe_jacobian_rar", "roe_roe_rar"),
        ("full_rar_vs_hybrid", "roe_hybrid", "roe_roe_rar"),
    ]
    effects = {}
    for name, baseline, candidate in comparisons:
        a = {r["seed"]: r["combined_error"] for r in grouped[baseline]}
        b = {r["seed"]: r["combined_error"] for r in grouped[candidate]}
        reduction = {str(seed): 100 * (a[seed] - b[seed]) / a[seed] for seed in sorted(a)}
        effects[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "percent_reduction_by_seed": reduction,
            "median_percent_reduction": float(np.median(list(reduction.values()))),
            "candidate_better_seeds": int(sum(value > 0 for value in reduction.values())),
        }
    return summary, effects


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def figure_wave_system(params):
    x = np.linspace(XL, XR, 4001)
    h, u = exact_solution(x, EVAL_TIME, params)
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    axes[0].plot(x, h, color="#214f83", lw=2)
    axes[1].plot(x, u, color="#b35c44", lw=2)
    for ax in axes:
        for key, ls in [("fan_head_speed", ":"), ("fan_tail_speed", "--"), ("shock_speed", "-.")]:
            ax.axvline(params[key] * EVAL_TIME, color="0.35", ls=ls, lw=0.9)
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Depth, $h$ (m)")
    axes[1].set_ylabel("Velocity, $u$ (m s$^{-1}$)")
    axes[1].set_xlabel("Position, $x$ (m)")
    axes[0].set_title("Exact Stoker wave system at $t=1.2$ s", loc="left")
    save(fig, "wave_system.png")


def figure_design():
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.axis("off")
    boxes = [
        (0.04, 0.69, 0.25, 0.18, "Common network\n5 × 32 tanh; $h>0$"),
        (0.375, 0.69, 0.25, 0.18, "Training residual\nDifferential or Roe flux"),
        (0.71, 0.69, 0.25, 0.18, "Point addition\nUniform, RAR, or hybrid"),
        (0.04, 0.20, 0.25, 0.18, "Matched conditions\nIC, BC, optimizer, seeds"),
        (0.375, 0.20, 0.25, 0.18, "Exact Stoker reference\nSix times + final profile"),
        (0.71, 0.20, 0.25, 0.18, "Hydraulic checks\nErrors, shock, mass, time"),
    ]
    for x, y, w, h, text in boxes:
        patch = plt.Rectangle((x, y), w, h, facecolor="#f4f6f8", edgecolor="#365d7d", lw=1.1)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    arrows = [((0.29, 0.78), (0.375, 0.78)), ((0.625, 0.78), (0.71, 0.78)),
              ((0.835, 0.69), (0.835, 0.38)), ((0.71, 0.29), (0.625, 0.29)),
              ((0.375, 0.29), (0.29, 0.29)), ((0.165, 0.38), (0.165, 0.69))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1, color="0.25"))
    ax.text(0.5, 0.52, "Six controlled arms × three seeds", ha="center", va="center",
            fontsize=11, weight="bold", color="#214f83")
    save(fig, "experimental_design.png")


def figure_anchor_patterns(anchors):
    conditions = ["roe_uniform", "roe_jacobian_rar", "roe_roe_rar", "roe_hybrid"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), sharex=True, sharey=True)
    seed = 1
    for ax, condition in zip(axes, conditions):
        rounds = anchors[(condition, seed)]
        for idx, points in enumerate(rounds):
            ax.scatter(points[:, 0], points[:, 1], s=2.2, alpha=0.55, label=str(idx + 1))
        ax.set_title(SHORT[condition])
        ax.set_xlim(XL, XR)
        ax.set_ylim(0, TEND)
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("Time, $t$ (s)")
    for ax in axes:
        ax.set_xlabel("$x$ (m)")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, color=f"C{i}", label=f"Round {i+1}") for i in range(5)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.76, wspace=0.12)
    save(fig, "anchor_patterns.png")


def plot_profile_family(grouped, predictions, params, conditions, name):
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.7), sharex=True)
    x = predictions[(conditions[0], 0)]["x"]
    h_exact, u_exact = exact_solution(x, EVAL_TIME, params)
    axes[0].plot(x, h_exact, color="black", lw=2.1, label="Exact")
    axes[1].plot(x, u_exact, color="black", lw=2.1, label="Exact")
    for condition in conditions:
        seed = representative_seed(grouped[condition])
        pred = predictions[(condition, seed)]
        label = f"{LABEL[condition]} (seed {seed})"
        axes[0].plot(pred["x"], pred["h"], color=COLORS[condition], lw=1.35, label=label)
        axes[1].plot(pred["x"], pred["u"], color=COLORS[condition], lw=1.35, label=label)
    axes[0].set_ylabel("Depth, $h$ (m)")
    axes[1].set_ylabel("Velocity, $u$ (m s$^{-1}$)")
    axes[1].set_xlabel("Position, $x$ (m)")
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, ncol=2, loc="best")
    save(fig, name)


def figure_error_history(grouped):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharex=True)
    for condition in ORDER:
        rr = grouped[condition]
        times = np.array([m["t"] for m in rr[0]["metrics_all_times"]])
        for ax, key in zip(axes, ["relative_l2_h", "relative_l2_u"]):
            values = np.asarray([[m[key] for m in r["metrics_all_times"]] for r in rr])
            med = np.median(values, axis=0)
            ax.plot(times, 100 * med, color=COLORS[condition], lw=1.5, label=SHORT[condition])
            ax.fill_between(times, 100 * values.min(axis=0), 100 * values.max(axis=0),
                            color=COLORS[condition], alpha=0.09)
    axes[0].set_ylabel("Relative depth error (%)")
    axes[1].set_ylabel("Relative velocity error (%)")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_yscale("log")
        ax.grid(alpha=0.18, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(top=0.78, wspace=0.28)
    save(fig, "error_evolution.png")


def figure_seed_results(grouped):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for i, condition in enumerate(ORDER):
        rr = grouped[condition]
        for r in rr:
            ax.scatter(i, r["combined_error"], s=34, marker=MARKERS[r["seed"]],
                       facecolor=COLORS[condition], edgecolor="white", linewidth=0.45, zorder=3)
        med = np.median([r["combined_error"] for r in rr])
        ax.plot([i - 0.23, i + 0.23], [med, med], color="black", lw=1.4, zorder=4)
    ax.set_xticks(range(len(ORDER)), [SHORT[c] for c in ORDER], rotation=18, ha="right")
    ax.set_ylabel("Time-balanced error, $E$")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.2, which="both")
    handles = [Line2D([0], [0], marker=MARKERS[s], color="none", markerfacecolor="0.4",
                      markeredgecolor="white", markersize=6, label=f"Seed {s}") for s in range(3)]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    save(fig, "seed_results.png")


def figure_diagnostics(grouped):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15))
    for condition in ORDER:
        for r in grouped[condition]:
            valid = r["shock_crossing_count"] == 1
            y = 100 * r["shock_location_error_m_t1.2"] if valid else 110
            axes[0].scatter(r["combined_error"], y, color=COLORS[condition], marker=MARKERS[r["seed"]],
                            s=32, edgecolor="white", linewidth=0.4)
            axes[1].scatter(r["combined_error"], r["mass_error_pct_t1.2"], color=COLORS[condition],
                            marker=MARKERS[r["seed"]], s=32, edgecolor="white", linewidth=0.4)
    axes[0].axhline(100, color="0.45", ls=":", lw=0.8)
    axes[0].text(0.22, 112, "invalid or multiple crossing", fontsize=7.5, color="0.35")
    axes[0].set_ylabel("Shock-location error (cm)")
    axes[1].set_ylabel("Mass error at 1.2 s (%)")
    for ax in axes:
        ax.set_xlabel("Time-balanced error, $E$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.18, which="both")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[c],
                      markersize=5, label=SHORT[c]) for c in ORDER]
    fig.legend(handles=handles, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.subplots_adjust(top=0.78, wspace=0.30)
    save(fig, "hydraulic_diagnostics.png")


def main():
    configure_plotting()
    OUT.mkdir(exist_ok=True)
    rows, manifest, predictions, anchors = load_archive()
    if len(rows) != 18:
        raise RuntimeError(f"Expected 18 completed runs, found {len(rows)}")
    if sorted(set(r["condition"] for r in rows)) != sorted(ORDER):
        raise RuntimeError("Archive conditions differ from the focused design")
    if any(int(r["anchor_exact_unique"]) != 3000 for r in rows):
        raise RuntimeError("Archive contains missing or duplicated added points")
    params = stoker_parameters()
    grouped = group(rows)
    summary, effects = summarize(rows)
    figure_wave_system(params)
    figure_design()
    figure_anchor_patterns(anchors)
    plot_profile_family(grouped, predictions, params,
                        ["jacobian_uniform", "jacobian_rar", "roe_uniform", "roe_roe_rar"],
                        "core_profiles.png")
    plot_profile_family(grouped, predictions, params,
                        ["roe_uniform", "roe_jacobian_rar", "roe_roe_rar", "roe_hybrid"],
                        "roe_profiles.png")
    figure_error_history(grouped)
    figure_seed_results(grouped)
    figure_diagnostics(grouped)
    record = {
        "source_archive": str(ARCHIVE),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "manifest": manifest,
        "stoker_parameters": params,
        "evaluation_time": EVAL_TIME,
        "summary": summary,
        "paired_effects": effects,
        "figures": sorted(p.name for p in OUT.glob("*.png")),
        "scope": "Only runs.tar.gz was used for quantitative results and plots.",
    }
    AUDIT.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "figures": len(record["figures"]), "audit": str(AUDIT)}, indent=2))


if __name__ == "__main__":
    main()
