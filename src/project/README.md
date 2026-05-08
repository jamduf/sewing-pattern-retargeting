# Pattern Lasso Tool

Interactive tool for extracting sewing pattern panels from images and exporting them into a structured format for downstream processing (e.g., `.ply` meshes for GarmentCode-style datasets).

# Selecting a Pattern

## Ideal Patterns

The current prototype works best with:

- Shirt / Pants patterns are currently the only supported pattern types

- Non-stretch woven garments
- Simpler garments with fewer panels -- currently the model is trained on a pretty limited dataset, so simpler patterns work best 
- Clear pattern outlines
- Flat scanned sewing patterns
- Minimal overlapping markings

---

## Patterns to Avoid

The current implementation does **not** handle highly detailed construction features well.

- stretch / athletic wear
- knit garments
- overlapping pattern pieces
- heavily annotated scans

---

# How to Use

Run the lasso tool:

```bash
python pattern_lasso.py

```

---

# Application Overview

<img src="../../docs/images/lassotoolex_1.png" width="350">

# Controls

| Key | Action |
|---|---|
| Left Click | Add anchor point |
| Right Click | Undo last anchor |
| Enter | Close current panel |
| N | Save completed panel |
| D | Duplicate selected panel |
| H | Mirror panel horizontally |
| V | Flip panel vertically |
| E | Export current panel |
| X | Export all saved panels |
| R | Reset current selection |
| [ / ] | Rotate selected panel |
| S | Toggle edge matching mode |

# Workflow Guide

---

## 1. Selecting Edges

Use left click to place anchor points along the garment outline.

The magnetic lasso will attempt to snap to nearby edges automatically.

Right click removes the most recent anchor.

<img src="../../docs/images/lassotoolex_2.png" width="350">

---

## 2. Closing and Saving a Panel

Once the full outline is traced:

1. Press `Enter` to close the polygon
2. Select which panel label is most appropriate -- for example, 'left_ftorso' is the front left torso panel, and 'pant_b_r' is the back right pants panel.
3. Press `N` to save the panel

<img src="../../docs/images/lassotoolex_3.png" width="350">
<img src="../../docs/images/lassotoolex_4.png" width="350">

---

## 3. Duplicating and Transforming Panels

Panels can be duplicated and transformed to accelerate annotation of symmetric garments.

### Duplicate Panel (`D`)

Useful for mirrored garment pieces such as sleeves or pant legs. Look for notations on the pattern like 'fold' or lines indicating folding to see which panels need to be duplicated on shirts / jackets.

<img src="../../docs/images/lassotoolex_5.png" width="350">

---

### Mirror / Flip Panels (`H` / `V`)

Mirror or vertically flip duplicated pieces to align with the original garment layout. This will be necessary for edge matching later on. 

<img src="../../docs/images/lassotoolex_6.png" width="350">

---

### Completed Panel Layout

A completed annotation should include all major garment panels. Because of limitations with the model training set and what details are supported, details like pockets should not be included in the completed panel layout. Additionally, there should not be more than one of any panel type, meaning that there can only be up to 4 torso panels etc. 

---

# Edge Matching Mode

Press `S` to enter edge matching mode.

In this mode:

1. Select one edge
2. Select the matching sewn edge
3. The pair will be stored as a seam relationship

These seam pairings are later used to reconstruct garment topology.

---

## Which Edges Should Be Matched?

Edges that are physically sewn together in the final garment should be paired.

Examples:
- sleeve ↔ armhole
- front torso ↔ back torso side seams
- inseam ↔ inseam
- duplicated pieces, if intended to be one solid piece should be matched on the folded edge.

<img src="../../docs/images/lassotoolex_7.png" width="350">

---

## Fully Matched Garment

Once all seam relationships are defined, the garment topology is complete.

<img src="../../docs/images/lassotoolex_8.png" width="350">

*Figure 8: Completed seam pairing layout.*

---

# Exporting the Pattern

Press `J` to export all saved panels and seam relationships into the .json filetype.

The exported format is compatible with the downstream geometry and ML pipeline.

---


