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

## Selecting a Pattern
  Which patterns would be ideal?
    - nonstretch
    - currently less panels is better -- still a prototype and training set is limited
    - 
  Which patterns should you not choose ?
  - athletic wear made of stretch materials, 

## How to Use

To use the lasso tool, run the following command:  
```bash
python pattern_lasso.py

picture 1: overview of the application

Then, use these controls to select the pattern pieces:

| Key  | Action |
| ------------- | ------------- |
| Left Click  | Add anchor point  |
| Right Click  | Undo last anchor  |
| Enter | Close Panel  |
| N  | Save Panel  |
| D  | Duplicate Panel  |
| M  | Mirror Panel (Horizontal)  |
| F  | Flip Panel (Vertical)  |
| E  | Export Current Panel  |
| X  | Export All Saved Panels |
| R  | Reset Current Selection |
| / or \ | Rotate |
| S  | Change to Edge Matching Mode |


### GUIDE HERE:

picture 2: selecting edges -- left click (will follow cursor until left clicked to confirm / right click to undo)

picture 3: saving a face -- enter to convert into a panel, n to save

picture 4 / 5 / 6: duplicating a face, flipping the panel, and a general overview of what a pattern with all selected panels should look like
NOTE: in this example, do not include the shirt pocket as both the lasso tool / model can't handle details like that

picture 7: edge matching mode -- which edges should be matched ?

picture 8: what it should look like with all matched edges

picture 9: exporting the formatted garment pattern

limitations on patterns -- what parts should you not select for pants / shirts / jackets ?
pants - button fly / pockets
shirts - pockets
jackets - pockets, cuffs



