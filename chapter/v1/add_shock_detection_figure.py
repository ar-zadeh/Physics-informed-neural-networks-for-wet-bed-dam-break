"""Clarify the shock-detection paragraph and add an explanatory figure."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches
from PIL import Image


HERE = Path(__file__).resolve().parent
DOCUMENT = HERE / "Chapter_Roe_RAR_V3.docx"
BACKUP = HERE / "Chapter_Roe_RAR_V3_before_shock_graphs.docx"
LOCKED_OUTPUT = HERE / "Chapter_Roe_RAR_V3_visualized.docx"
FIGURE = HERE / "focused_figures" / "shock_detection_rule.png"


def make_figure() -> None:
    """Visualize every step of the shock-location calculation."""
    run_dir = HERE.parent.parent / "runs_extracted" / "runs" / "ablation" / "roe_roe_rar__seed0"
    with np.load(run_dir / "prediction_t1.2.npz") as data:
        x_actual = np.asarray(data["x"])
        h_actual = np.asarray(data["h"])
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    h_star = 0.3961748167990253
    h_right = 0.1
    threshold = (h_star + h_right) / 2
    exact_shock = float(result["shock_exact_m_t1.2"])
    predicted_shock = float(result["shock_prediction_m_t1.2"])
    position_error = float(result["shock_location_error_m_t1.2"])

    descending = np.where(
        (x_actual[:-1] > 0.5)
        & (h_actual[:-1] >= threshold)
        & (h_actual[1:] < threshold)
    )[0]
    if len(descending) != 1:
        raise RuntimeError(f"Expected one actual crossing; found {len(descending)}")
    crossing_index = int(descending[0])
    x_left, x_right = x_actual[crossing_index : crossing_index + 2]
    h_left, h_right_grid = h_actual[crossing_index : crossing_index + 2]

    x = np.linspace(0.5, 5.0, 700)
    no_crossing = 0.285 + (h_star - 0.285) / (1 + np.exp((x - 3.65) / 0.42))

    # A broad front with localized ringing, representative of an ambiguous profile.
    multiple = h_right + (h_star - h_right) / (1 + np.exp((x - 3.78) / 0.20))
    multiple += 0.075 * np.exp(-((x - 3.82) / 0.55) ** 2) * np.sin(18 * (x - 3.82))

    colors = {
        "profile": "#0072B2",
        "threshold": "#D55E00",
        "exact": "#222222",
        "estimate": "#7B3294",
        "invalid": "#B2182B",
        "grid": "#D7DCE2",
        "accept": "#18864B",
    }

    fig = plt.figure(figsize=(7.2, 8.4), facecolor="white")
    grid = fig.add_gridspec(4, 3, height_ratios=[1.2, 1.55, 1.05, 1.55],
                           hspace=0.78, wspace=0.34)

    # Step 1: show that the threshold is exactly halfway between the two depths.
    ax = fig.add_subplot(grid[0, :])
    ax.set_title("1  Calculate the midpoint depth", fontsize=12, fontweight="bold", loc="left", pad=7)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.07, 0.43)
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)", fontsize=9.5)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    ax.vlines(1.2, h_right, h_star, color="#7A8693", linewidth=2.0)
    ax.scatter([1.2, 1.2], [h_star, h_right], s=42, color=colors["profile"], zorder=4)
    ax.scatter([1.2], [threshold], s=65, color=colors["threshold"], zorder=5)
    ax.annotate("", xy=(0.63, h_star), xytext=(0.63, threshold),
                arrowprops=dict(arrowstyle="<->", color="#52606D", linewidth=1.2))
    ax.annotate("", xy=(0.63, threshold), xytext=(0.63, h_right),
                arrowprops=dict(arrowstyle="<->", color="#52606D", linewidth=1.2))
    ax.text(1.65, h_star, r"upper depth: $h_*=0.396175$ m", va="center", fontsize=9.3)
    ax.text(1.65, threshold, r"midpoint: $h_{thr}=0.248088$ m", va="center",
            fontsize=9.3, color=colors["threshold"], fontweight="bold")
    ax.text(1.65, h_right, r"downstream depth: $h_R=0.100000$ m", va="center", fontsize=9.3)
    ax.text(7.35, 0.25,
            r"$h_{thr}=\frac{h_*+h_R}{2}$" "\n"
            r"$=\frac{0.396175+0.100000}{2}$" "\n"
            r"$=0.248088$ m",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF4E8", edgecolor="#E4A46A"))

    # Step 2: display the actual two grid values and their straight-line intersection.
    ax = fig.add_subplot(grid[1, :2])
    ax.set_title("2  Locate the crossing between two grid points",
                 fontsize=11.5, fontweight="bold", loc="left", pad=7)
    neighborhood = slice(crossing_index - 2, crossing_index + 4)
    ax.plot(x_actual[neighborhood], h_actual[neighborhood], color=colors["profile"],
            linewidth=1.6, alpha=0.65)
    ax.plot([x_left, x_right], [h_left, h_right_grid], color=colors["estimate"], linewidth=2.5)
    ax.scatter([x_left, x_right], [h_left, h_right_grid], s=55,
               color=colors["profile"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.axhline(threshold, color=colors["threshold"], linestyle="--", linewidth=1.5)
    ax.axvline(predicted_shock, color=colors["estimate"], linestyle=":", linewidth=1.5)
    ax.scatter([predicted_shock], [threshold], marker="*", s=125,
               color=colors["threshold"], edgecolor="white", linewidth=0.6, zorder=6)
    ax.annotate(f"({x_left:.3f}, {h_left:.6f})", xy=(x_left, h_left),
                xytext=(-4, 14), textcoords="offset points", ha="right", fontsize=8.0)
    ax.annotate(f"({x_right:.3f}, {h_right_grid:.6f})", xy=(x_right, h_right_grid),
                xytext=(5, -17), textcoords="offset points", ha="left", fontsize=8.0)
    ax.annotate(f"crossing = {predicted_shock:.6f} m", xy=(predicted_shock, threshold),
                xytext=(3.7510, 0.258), textcoords="data", fontsize=8.5,
                color=colors["estimate"], arrowprops=dict(arrowstyle="->", color=colors["estimate"]))
    ax.set_xlim(x_left - 0.003, x_right + 0.003)
    ax.set_ylim(h_right_grid - 0.015, h_left + 0.018)
    ax.set_xlabel("Position, x (m)", fontsize=9.5)
    ax.set_ylabel("Predicted depth (m)", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.grid(True, color=colors["grid"], linewidth=0.6)

    formula_ax = fig.add_subplot(grid[1, 2])
    formula_ax.axis("off")
    formula_ax.text(0.5, 0.56,
                    "Substitute the values\n\n"
                    r"$x_{pred}=x_i+$" "\n"
                    r"$\frac{h_i-h_{thr}}{h_i-h_{i+1}}(x_{i+1}-x_i)$" "\n\n"
                    r"$=3.750+$" "\n"
                    r"$\frac{.276715-.248088}{.276715-.247109}(0.005)$" "\n\n"
                    r"$=3.754835$ m",
                    ha="center", va="center", fontsize=9.2, fontweight="normal",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4EFF8", edgecolor="#9C78B2"))

    # Step 3: position error is simply the gap between predicted and exact locations.
    ax = fig.add_subplot(grid[2, :])
    ax.set_title("3  Position error = the horizontal distance between the predicted and exact fronts",
                 fontsize=11.5, fontweight="bold", loc="left", pad=7)
    ax.set_xlim(3.715, 3.765)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Position, x (m)", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.axvline(exact_shock, ymin=0.18, ymax=0.88, color=colors["exact"], linestyle=":", linewidth=2.0)
    ax.axvline(predicted_shock, ymin=0.18, ymax=0.88, color=colors["estimate"], linestyle="-.", linewidth=2.0)
    ax.text(exact_shock, 0.91, f"exact = {exact_shock:.6f} m", ha="center", fontsize=8.7)
    ax.text(predicted_shock, 0.91, f"predicted = {predicted_shock:.6f} m",
            ha="center", fontsize=8.7, color=colors["estimate"])
    ax.annotate("", xy=(exact_shock, 0.54), xytext=(predicted_shock, 0.54),
                arrowprops=dict(arrowstyle="<->", color=colors["invalid"], linewidth=2.0))
    ax.text((exact_shock + predicted_shock) / 2, 0.61, f"error = {100*position_error:.2f} cm",
            ha="center", fontsize=9.2, color=colors["invalid"], fontweight="bold")
    ax.text(0.5, 0.03,
            r"exact: $3.105134\ \mathrm{m/s}\times1.2\ \mathrm{s}=3.726160$ m"
            rf"     |     error: $|3.754835-3.726160|={position_error:.6f}$ m $={100*position_error:.2f}$ cm",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEC", edgecolor="#D88989"))

    # Bottom row: retain the visual acceptance rule for one, zero, or many crossings.
    bottom_axes = [fig.add_subplot(grid[3, column]) for column in range(3)]
    profiles = [
        np.interp(x, x_actual, h_actual),
        no_crossing,
        multiple,
    ]
    titles = [
        "4a  Accept: 1 crossing",
        "4b  Reject: 0 crossings",
        "4c  Reject: >1 crossing",
    ]
    for index, (ax, profile, title) in enumerate(zip(bottom_axes, profiles, titles)):
        profile_color = colors["profile"] if index == 0 else colors["invalid"]
        ax.plot(x, profile, color=profile_color, linewidth=2.0)
        ax.axhline(threshold, color=colors["threshold"], linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6,
                     color=colors["accept"] if index == 0 else colors["invalid"])
        ax.set_xlim(0.5, 5.0)
        ax.set_ylim(0.06, 0.43)
        ax.set_xticks([0.5, 2.0, 3.5, 5.0])
        ax.grid(True, color=colors["grid"], linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Position, x (m)", fontsize=9)
    bottom_axes[0].set_ylabel("Predicted depth (m)", fontsize=9.5)
    bottom_axes[0].scatter([predicted_shock], [threshold], marker="o", s=50,
                       color=colors["accept"], edgecolor="white", linewidth=0.7, zorder=6)
    bottom_axes[0].annotate("one crossing", xy=(predicted_shock, threshold),
                        xytext=(2.45, 0.19), fontsize=8.0, color=colors["accept"],
                        arrowprops=dict(arrowstyle="->", color=colors["accept"]))

    # Mark all threshold crossings in the oscillatory profile.
    signs = np.sign(multiple - threshold)
    crossing_indices = np.where(signs[:-1] * signs[1:] < 0)[0]
    crossing_x = []
    for i in crossing_indices:
        fraction = (threshold - multiple[i]) / (multiple[i + 1] - multiple[i])
        crossing_x.append(x[i] + fraction * (x[i + 1] - x[i]))
    bottom_axes[2].scatter(
        crossing_x,
        [threshold] * len(crossing_x),
        marker="x",
        s=45,
        linewidths=1.8,
        color=colors["invalid"],
        zorder=6,
    )
    bottom_axes[2].annotate("three crossings", xy=(crossing_x[1], threshold),
                        xytext=(2.25, 0.18), fontsize=8.0, color=colors["invalid"],
                        arrowprops=dict(arrowstyle="->", color=colors["invalid"]))

    fig.suptitle("How the midpoint threshold and shock-position error are calculated",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.265, "Finally, decide whether the crossing is usable",
             ha="center", va="center", fontsize=11, fontweight="bold", color="#38434F")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.94, bottom=0.055)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def add_text(paragraph, text: str) -> None:
    paragraph.add_run(text)


def add_math(paragraph, math_element) -> None:
    paragraph._p.append(deepcopy(math_element))


def replace_paragraph(paragraph) -> None:
    math_elements = paragraph._p.xpath("./m:oMath")
    if len(math_elements) != 4:
        raise RuntimeError(f"Expected four equations in target paragraph; found {len(math_elements)}")

    paragraph.clear()
    add_text(paragraph, "Final depth and velocity errors are reported separately because ")
    add_math(paragraph, math_elements[0])
    add_text(
        paragraph,
        " can hide which field controls a difference. Figure 3 shows the shock-location "
        "calculation step by step for the representative Roe-R-RAR seed 0 prediction at "
        "t = 1.2 s. First, we search only downstream of ",
    )
    add_math(paragraph, math_elements[2])
    add_text(paragraph, " m and calculate the midpoint-depth threshold ")
    add_math(paragraph, math_elements[1])
    add_text(
        paragraph,
        ". This is an ordinary average: (0.396175 m + 0.100000 m)/2 = 0.248088 m. "
        "Next, we find the two neighboring grid points that lie on opposite sides of "
        "this threshold. Their predicted depths are 0.276715 m at x = 3.750 m and "
        "0.247109 m at x = 3.755 m. Drawing a straight line between these points places "
        "the threshold crossing at x = 3.754835 m. The exact shock is at x = 3.726160 m "
        "because its speed, 3.105134 m/s, is multiplied by 1.2 s. The position error is ",
    )
    add_math(paragraph, math_elements[3])
    add_text(
        paragraph,
        ", meaning the horizontal distance between the predicted and exact shock fronts. "
        "For this run, |3.754835 - 3.726160| = 0.028674 m, or 2.87 cm; the absolute-value "
        "sign makes the distance positive. A result is valid only when the depth falls "
        "through the threshold exactly once. A profile with no crossing or several "
        "crossings is marked invalid rather than being assigned a misleading grid point.",
    )


def replace_plain_text(paragraph, old: str, new: str) -> None:
    for run in paragraph.runs:
        if old in run.text:
            run.text = run.text.replace(old, new)


def update_existing_revision(document, target) -> None:
    replace_paragraph(target)

    figure_paragraph = next(
        (
            p
            for p in document.paragraphs
            if p._p.getprevious() is target._p and p._p.xpath(".//w:drawing")
        ),
        None,
    )
    if figure_paragraph is None:
        raise RuntimeError("Could not find the existing shock-detection figure")
    blip = figure_paragraph._p.xpath(".//a:blip")[0]
    image_part = document.part.related_parts[blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")]
    image_part._blob = FIGURE.read_bytes()
    inline_shape = next(
        shape
        for shape in document.inline_shapes
        if shape._inline.getparent().getparent().getparent() is figure_paragraph._p
    )
    with Image.open(FIGURE) as image:
        aspect_ratio = image.height / image.width
    inline_shape.width = Inches(6.3)
    inline_shape.height = Inches(6.3 * aspect_ratio)
    figure_paragraph.paragraph_format.keep_with_next = True
    caption_element = figure_paragraph._p.getnext()
    caption = next((p for p in document.paragraphs if p._p is caption_element), None)
    if caption is not None and caption.style.name == "Image Caption":
        replace_plain_text(caption, "Figure 2.", "Figure 3.")
        caption.text = (
            "Figure 3. Step-by-step shock-wave detection for the representative Roe-R-RAR "
            "seed 0 prediction at t = 1.2 s. The first three diagrams calculate the "
            "midpoint-depth threshold, locate the predicted crossing by linear interpolation, "
            "and measure its distance from the exact shock. The bottom row shows why one "
            "descending crossing is accepted while zero or several crossings are rejected. "
            "Values in the first three panels come from the stored run; the two invalid "
            "profiles are schematic."
        )


def revise_document() -> Path:
    if not BACKUP.exists():
        BACKUP.write_bytes(DOCUMENT.read_bytes())

    document = Document(DOCUMENT)
    original_target = next(
        (p for p in document.paragraphs if "A shock wave is considered detectable" in p.text),
        None,
    )
    existing_target = next(
        (p for p in document.paragraphs if "At t = 1.2 s, we locate" in p.text),
        None,
    )
    target = original_target or existing_target
    if target is None:
        raise RuntimeError("Could not find the shock-detection paragraph")
    if original_target is not None:
        replace_paragraph(target)
    else:
        update_existing_revision(document, target)

    if original_target is not None:
        # The earlier removal of the experimental-design graphic left a numbering gap.
        # Closing it lets the new explanatory graphic become Figure 3 without changing
        # the established Figure 4--8 numbering later in the chapter.
        for paragraph in document.paragraphs:
            if paragraph.style.name == "Image Caption" and paragraph.text.startswith("Figure 3."):
                replace_plain_text(paragraph, "Figure 3.", "Figure 2.")
            elif paragraph.text.startswith("Figure 3 shows"):
                replace_plain_text(paragraph, "Figure 3 shows", "Figure 2 shows")

    if original_target is not None:
        following = target._p.getnext()
        if following is None:
            raise RuntimeError("Target paragraph has no following paragraph")
        following_paragraph = next(p for p in document.paragraphs if p._p is following)

        figure_paragraph = following_paragraph.insert_paragraph_before(style="Captioned Figure")
        figure_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        figure_paragraph.add_run().add_picture(str(FIGURE), width=Inches(6.3))

        caption = following_paragraph.insert_paragraph_before(style="Image Caption")
        caption.add_run(
            "Figure 3. Step-by-step shock-wave detection for the representative Roe-R-RAR "
            "seed 0 prediction at t = 1.2 s. The first three diagrams calculate the "
            "midpoint-depth threshold, locate the predicted crossing by linear interpolation, "
            "and measure its distance from the exact shock. The bottom row shows why one "
            "descending crossing is accepted while zero or several crossings are rejected. "
            "Values in the first three panels come from the stored run; the two invalid "
            "profiles are schematic."
        )

    try:
        document.save(DOCUMENT)
        return DOCUMENT
    except PermissionError:
        document.save(LOCKED_OUTPUT)
        return LOCKED_OUTPUT


def main() -> None:
    make_figure()
    output = revise_document()
    print(f"Updated {output}")
    print(f"Figure {FIGURE}")
    print(f"Backup {BACKUP}")


if __name__ == "__main__":
    main()
