
# Heritage Semantic Segmentation — Suzhou Urban Heritage Study

This repository provides the semantic segmentation inference code
accompanying the paper:

Reconstructing Contested Urban Heritage Spaces via Crowdsourced Prompts in Suzhou

## Overview

`cityscapes_solve.py` runs pixel-level semantic segmentation
on street-view imagery using **Mask2Former** (Swin-L backbone, ADE20K),
then remaps all 150 ADE20K categories onto a custom **8-class heritage
ontology** (Water Bodies, Traditional Facades, Modern Commercials,
Industrial Structures, Greenery, Human Figures, Sky, Ground Surfaces).

## Outputs

- `*_ss.png` — colour-coded segmentation mask
- `*_class_distribution.png` — publication-quality horizontal bar chart
  of pixel-area proportions per heritage class

## Requirements

```bash
pip install torch transformers Pillow matplotlib numpy
```

## Usage

```bash
python cityscapes_solve_v4_adv2kto8.py   # edit input_image_path at bottom
```

## Notes

The ADE20K → heritage mapping table is fully documented in-code.
Absent classes are rendered with hatched stubs to preserve full
taxonomy legibility in figures.
```


