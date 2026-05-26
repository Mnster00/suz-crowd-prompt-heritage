import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ─────────────────────────────────────────────
# Heritage ontology class metadata (n = 8)
# C = {c1, …, c8}
# ─────────────────────────────────────────────
HERITAGE_CLASSES = [
    {"id": 0, "name_en": "Water Bodies",               "rgb": ( 64, 164, 223)},
    {"id": 1, "name_en": "Traditional Facades",        "rgb": (188, 143,  85)},
    {"id": 2, "name_en": "Modern Commercials",         "rgb": (180, 180, 180)},
    {"id": 3, "name_en": "Industrial Structures",      "rgb": (100,  80,  60)},
    {"id": 4, "name_en": "Greenery",                   "rgb": ( 80, 160,  60)},
    {"id": 5, "name_en": "Human Figures",              "rgb": (220,  20,  60)},
    {"id": 6, "name_en": "Sky",                        "rgb": ( 70, 130, 180)},
    {"id": 7, "name_en": "Ground Surfaces",            "rgb": (128, 120,  96)},
]

N_CLASSES = len(HERITAGE_CLASSES)

# ─────────────────────────────────────────────
# ADE20K label index → heritage class mapping
#
# ADE20K has 150 semantic categories (0-indexed).
# Each ADE20K category is mapped to one of the 8
# heritage classes; unmapped labels fall back to
# the most contextually plausible class or are
# absorbed into "Ground Surfaces" (id=7) as a
# catch-all.
#
# Key ADE20K indices used below (0-based):
#   0  wall                  → Traditional Facades (1)
#   1  building / skyscraper → Modern Commercial Facades (2)
#   2  sky                   → Sky (6)
#   3  floor / road          → Ground Surfaces (7)
#   4  tree                  → Greenery (4)
#   5  ceiling               → Ground Surfaces (7)
#   6  road / sidewalk       → Ground Surfaces (7)
#   7  grass                 → Greenery (4)
#   8  plant / bush          → Greenery (4)
#   9  mountain / hill       → Ground Surfaces (7)
#  10  water / sea / lake    → Water Bodies (0)
#  11  house                 → Traditional Facades (1)
#  12  sea / ocean           → Water Bodies (0)
#  13  bridge                → Industrial Structures (3)
#  14  windowpane            → Modern Commercial Facades (2)
#  15  fence / railing       → Industrial Structures (3)
#  16  person / man / woman  → Human Figures (5)
#  ... (full mapping table below covers all 150 labels)
# ─────────────────────────────────────────────

# Default = Ground Surfaces (7)
_ADE20K_TO_HERITAGE = np.full(150, 7, dtype=np.int64)

# ── Water Bodies (0) ──────────────────────────
for idx in [9, 21, 26, 37, 60, 109, 128]:
    # water, sea, river, waterfall, swimming pool, fountain, moat
    _ADE20K_TO_HERITAGE[idx] = 0

# ── Traditional Facades (1) ──────────────────
for idx in [0, 11, 30, 43, 55, 79, 84, 86, 95]:
    # wall, house, hovel/cabin, booth, tower, column, stairway, fireplace, arch
    _ADE20K_TO_HERITAGE[idx] = 1

# ── Modern Commercial Facades (2) ────────────
for idx in [1, 14, 25, 31, 42, 46, 52, 53, 67, 76, 85, 93, 94, 96, 99]:
    # building/skyscraper, windowpane, skyscraper, bus stop, shop/store,
    # signboard, awning, bulletin board, glass, escalator, office, screen,
    # flag, canopy, traffic light
    _ADE20K_TO_HERITAGE[idx] = 2

# ── Industrial Structures (3) ─────────────────
for idx in [13, 15, 27, 38, 41, 45, 56, 64, 77, 82, 83, 100, 103, 108, 116, 144]:
    # bridge, railing/fence, crane, pier, conveyor belt, ship,
    # pole, antenna, tank, stage, truck, car, bus, van, minibike, pipeline
    _ADE20K_TO_HERITAGE[idx] = 3

# ── Greenery (4) ──────────────────────────────
for idx in [4, 7, 8, 17, 66, 72, 96, 124, 125]:
    # tree, grass, plant/bush, palm tree, flower, pot/plant,
    # (canopy already mapped; hedge, cultivated field)
    _ADE20K_TO_HERITAGE[idx] = 4
# overwrite with greenery after industrial was set for 96 (canopy → greenery)
_ADE20K_TO_HERITAGE[66] = 4   # flower
_ADE20K_TO_HERITAGE[17] = 4   # palm tree
_ADE20K_TO_HERITAGE[72] = 4   # pot/plant

# ── Human Figures (5) ─────────────────────────
for idx in [12, 80, 127, 133, 137]:
    # person, man, woman, girl, boy
    _ADE20K_TO_HERITAGE[idx] = 5

# ── Sky (6) ───────────────────────────────────
_ADE20K_TO_HERITAGE[2] = 6   # sky

# ── Ground Surfaces (7) — explicit additions ──
for idx in [3, 5, 6, 9, 10, 16, 22, 28, 29, 53, 59, 92, 97, 110, 118]:
    # floor, ceiling, road, mountain, field, sidewalk, dirt track,
    # carpet, stairs, pavement, earth/ground, path, runway, terrain, gravel
    _ADE20K_TO_HERITAGE[idx] = 7


def get_heritage_palette() -> np.ndarray:
    """Return RGB palette array for the 8 heritage classes."""
    return np.array([c["rgb"] for c in HERITAGE_CLASSES], dtype=np.uint8)


# ─────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────

def segment_image(image_path: str):
    if not os.path.exists(image_path):
        print(f"Error: image not found – {image_path}")
        return

    print("Loading Mask2Former (Swin-L, ADE20K) …")
    model_name = "facebook/mask2former-swin-large-ade-semantic"
    processor  = AutoImageProcessor.from_pretrained(model_name)
    model      = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model.eval()

    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    print("Running semantic segmentation …")
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: returns a (H, W) label map in ADE20K space (150 classes)
    result = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]  # single image
    ade_labels = result.cpu().numpy().astype(np.int64)
    ade_labels = np.clip(ade_labels, 0, 149)

    # Map ADE20K labels → 8 heritage classes
    pred_classes = _ADE20K_TO_HERITAGE[ade_labels]

    # ── coloured segmentation map ──────────────────────────────────────────
    palette      = get_heritage_palette()
    seg_map      = palette[pred_classes]
    output_image = Image.fromarray(seg_map.astype(np.uint8))

    base_name, _ = os.path.splitext(image_path)
    output_seg   = f"{base_name}_ss.png"
    output_image.save(output_seg)
    print(f"Segmentation mask saved → {output_seg}")

    # ── class pixel statistics ─────────────────────────────────────────────
    total_pixels = pred_classes.size
    pixel_counts = np.bincount(pred_classes.ravel(), minlength=N_CLASSES)
    percentages  = pixel_counts / total_pixels * 100.0

    output_fig_png = f"{base_name}_class_distribution.png"
    plot_class_distribution(percentages, output_fig_png, source_image=image_path)
    print(f"Figure (PNG) saved → {output_fig_png}")

    return pred_classes, percentages


# ─────────────────────────────────────────────
# Publication-quality figure
# ─────────────────────────────────────────────

def _normalise_rgb(rgb_tuple):
    """Convert 0-255 tuple to 0-1 tuple for matplotlib."""
    return tuple(v / 255.0 for v in rgb_tuple)


def plot_class_distribution(
    percentages: np.ndarray,
    output_path: str,
    source_image: str = "",
):
    """
    Generate a publication-quality horizontal bar chart of heritage-ontology
    class pixel-area percentages, conforming to Nature / top SCI/SSCI journal
    style.

    Design principles
    -----------------
    • Single-column width (88 mm) scaled to full-page equivalent (174 mm).
    • Helvetica Neue / Arial typeface hierarchy (journal standard).
    • Sparse, Tufte-inspired axis spines (only left + bottom retained).
    • Each bar coloured with the heritage class colour.
    • Absent classes are shown with a translucent bar + dashed outline so that
      the full taxonomy remains legible – an important requirement for methods
      reproducibility.
    • Value labels positioned outside bars to avoid occlusion.
    • Minor grid on x-axis for precise reading without visual clutter.
    """

    # ── typography & global style ──────────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial",
                               "DejaVu Sans"],
        "font.size":          7,
        "axes.labelsize":     8,
        "axes.titlesize":     9,
        "xtick.labelsize":    6.5,
        "ytick.labelsize":    7,
        "legend.fontsize":    6.5,
        "figure.dpi":         300,
        "savefig.dpi":        600,
        "pdf.fonttype":       42,   # embed TrueType fonts in PDF
        "ps.fonttype":        42,
        "axes.linewidth":     0.6,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.width":  0.4,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.minor.size":   1.5,
        "lines.linewidth":    0.8,
    })

    # ── layout ─────────────────────────────────────────────────────────────
    fig_width_mm  = 90          # full-page Nature figure width
    fig_height_mm = 60          # tuned for 8 classes
    mm2in         = 1 / 25.4
    fig, ax = plt.subplots(
        figsize=(fig_width_mm * mm2in, fig_height_mm * mm2in),
        constrained_layout=False,
    )

    # ── data preparation ───────────────────────────────────────────────────
    labels     = [c["name_en"] for c in HERITAGE_CLASSES]
    bar_colors = [_normalise_rgb(c["rgb"]) for c in HERITAGE_CLASSES]
    sorted_pct = percentages                       # canonical order preserved
    present    = [p > 0 for p in sorted_pct]

    y_pos = np.arange(len(labels))

    # ── draw bars ──────────────────────────────────────────────────────────
    bar_height = 0.62

    for i, (pct, color, is_present) in enumerate(
            zip(sorted_pct, bar_colors, present)):

        if is_present:
            ax.barh(
                y_pos[i], pct,
                height=bar_height,
                color=color,
                alpha=0.88,
                zorder=3,
                linewidth=0,
            )
        else:
            # absent class: hatched, translucent
            ax.barh(
                y_pos[i], 0.05,          # tiny stub so hatch is visible
                height=bar_height,
                color=color,
                alpha=0.25,
                zorder=3,
                linewidth=0.5,
                edgecolor=color,
                linestyle="--",
                hatch="////",
            )

        # value label
        x_label   = pct + 0.3 if is_present else 0.35
        label_txt = f"{pct:.2f}%" if is_present else "0.00%"
        ax.text(
            x_label, y_pos[i],
            label_txt,
            va="center", ha="left",
            fontsize=5.8,
            color="#2b2b2b",
            zorder=4,
        )

    # ── axes cosmetics ─────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()

    max_pct = max(sorted_pct.max(), 1.0)
    ax.set_xlim(0, max_pct * 1.18)
    ax.set_xlabel("Pixel Area Proportion (%)", labelpad=4)

    # remove top & right spines (Tufte principle)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    # minor x-grid
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis="x", which="major", color="#cccccc",
            linewidth=0.4, linestyle="--", zorder=0)
    ax.grid(axis="x", which="minor", color="#e8e8e8",
            linewidth=0.25, linestyle=":", zorder=0)
    ax.set_axisbelow(True)

    # y-axis: alternate row shading for readability
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5,
                       color="#f7f7f7", zorder=0, linewidth=0)

    # ── save ───────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    input_image_path = "1x1.png"
    segment_image(input_image_path)
