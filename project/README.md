# Pattern Lasso Tool

Interactive tool for extracting sewing pattern panels from images and exporting them into a structured format for downstream processing (e.g., `.ply` meshes for GarmentCode-style datasets).

---

## Features

- Magnetic lasso / edge-snapping selection
- Interactive panel extraction from raster sewing patterns
- Panel duplication and symmetry tools
- Edge matching / seam pairing workflow
- Export to `.ply` mesh format
- Compatible with downstream pattern → mesh → ML retargeting pipeline

---

## Requirements

- Python 3.9+
- PyQt6
- numpy
- opencv-python
- mapbox_earcut

Install dependencies:

```bash
pip install PyQt6 numpy opencv-python mapbox_earcut
```

---

# Selecting a Pattern

## Ideal Patterns

The current prototype works best with:

- Non-stretch woven garments
- Simpler garments with fewer panels
- Clear pattern outlines
- Flat scanned sewing patterns
- Minimal overlapping markings

Examples:
- basic shirts
- simple trousers
- skirts
- simple jackets

---

## Patterns to Avoid

The current implementation does **not** handle highly detailed construction features well.

Avoid:

### Pants
- button flies
- zipper flies
- pockets
- cargo details

### Shirts
- chest pockets
- decorative pleats
- complex collars

### Jackets
- cuffs
- welt pockets
- lining panels
- shoulder pads
- highly segmented tailoring

### General
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

![Application Overview](lassotoolex_1.png)

*Figure 1: Main interface overview.*

---

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

---

# Workflow Guide

---

## 1. Selecting Edges

Use left click to place anchor points along the garment outline.

The magnetic lasso will attempt to snap to nearby edges automatically.

Right click removes the most recent anchor.

![Selecting Edges](lassotoolex_2.png)

*Figure 2: Edge selection with magnetic snapping.*

---

## 2. Closing and Saving a Panel

Once the full outline is traced:

1. Press `Enter` to close the polygon
2. Press `N` to save the panel

![Saving a Panel](lassotoolex_3.png)

*Figure 3: Closing and saving a panel.*

---

## 3. Duplicating and Transforming Panels

Panels can be duplicated and transformed to accelerate annotation of symmetric garments.

### Duplicate Panel (`D`)

Useful for mirrored garment pieces such as sleeves or pant legs.

![Duplicating Panels](lassotoolex_4.png)

*Figure 4: Duplicating a panel.*

---

### Mirror / Flip Panels (`H` / `V`)

Mirror or vertically flip duplicated pieces to align with the original garment layout.

![Mirroring Panels](lassotoolex_5.png)

*Figure 5: Mirroring and flipping panels.*

---

### Completed Panel Layout

A completed annotation should include all major garment panels.

**Note:**  
In this example, the shirt pocket is intentionally excluded because the current lasso tool and downstream ML pipeline do not yet handle small decorative subcomponents reliably.

![Completed Pattern](lassotoolex_6.png)

*Figure 6: Example completed panel extraction.*

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
- collar attachment edges

![Edge Matching](lassotoolex_7.png)

*Figure 7: Selecting matching sewn edges.*

---

## Fully Matched Garment

Once all seam relationships are defined, the garment topology is complete.

![Matched Garment](lassotoolex_8.png)

*Figure 8: Completed seam pairing layout.*

---

# Exporting the Pattern

Press `X` to export all saved panels and seam relationships.

The exported format is compatible with the downstream geometry and ML pipeline.

![Exporting Pattern](lassotoolex_9.png)

*Figure 9: Exported structured garment pattern.*

---

# Current Limitations

This project is currently a research prototype.

Known limitations include:

- limited support for highly detailed garments
- limited handling of curved seam topology
- no automatic pocket / cuff / lining detection
- limited support for stretch garments
- no fabric simulation
- partial support for nested or overlapping pattern pieces

---

# Pipeline Integration

The lasso tool is designed as the front-end extraction stage of a larger pipeline:

```text
Pattern Image / PDF
        ↓
Pattern Lasso Tool
        ↓
Structured Pattern Representation
        ↓
Mesh / Geometry Conversion
        ↓
ML-Based Garment Retargeting
        ↓
Resized Garment Pattern
```

---

# Project Status

Current supported workflows:

- Shirt panel extraction
- Pants panel extraction
- Basic jacket panel extraction
- Pattern export
- Seam pairing
- ML retargeting experiments

This project is actively under development.
