# Pattern Lasso Tool

Interactive tool for extracting sewing pattern panels from images and exporting them into a structured format for downstream processing (e.g., `.ply` meshes for GarmentCode-style datasets).

## Features

- Magnetic lasso / edge-snapping selection
- Panel extraction from raster sewing patterns
- Export to `.ply` mesh format
- Compatible with downstream pattern → mesh → ML pipeline

## Requirements

- Python 3.9+
- PyQt6
- numpy
- opencv-python
- mapbox_earcut

Install:

```bash
pip install PyQt6 numpy opencv-python mapbox_earcut
