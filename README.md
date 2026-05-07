# Sewing Pattern Extraction and Retargeting

<!-- I shortened the title above.  You need to redo your intro here so that it is actually interesting.
Use this format: 1 sentence about what is the problem. 1 sentence about why existing solutions fail. 1 sentence about 1 is your solution. 1 sentence about why your solution is better than other solutions. -->

<p align="center">
  <img src="docs/images/repothumbnail.png" width="1200">
</p>

<!-- the text you had here didn't explain anything not already explained in the image. -->

## Usage

<!-- you need a step-by-step set of instructions for running your code.  I've provided a rough outline here of a possibility. -->

Install all dependencies by running
```
$ pip3 install -r requirements.txt
```

Convert an input pattern (in png image format) into the custom XXX format the resizing model expects.
```
$ python3 run_tool.py input.png
```
This command launches a window that allows you to manually annotate where the edges in the pattern are.
The animated gif below shows the tool in action.
<!-- this should be an animated gif showing the tool in action -->
<img src="docs/images/lassotoolex_6.png" width="400">

The output file is saved to a `output.XXX`.
The last step is to pass this to our model.
```
$ python3 resize.py output.XXX --newsize=YYY
```
The command above generates a new pattern shown below
<img src='fixme.png'>

## How it Works

You need to write more here about how it works under the hood.

The retargeting model:

- consumes structured garment panel geometry
- conditions on body measurements
- predicts resized panel geometry
- preserves garment structure through conservative blending, maintaining aesthetic choices 

Currently supported garment categories:

- shirts
- pants

---

# Current Status

This repository is an active research prototype.

Current strengths:

- successful panel extraction from real sewing patterns
- semantic panel labeling workflow
- end-to-end lasso → model → SVG pipeline
- pretrained shirt and pants retargeting checkpoints

Current limitations:

- limited training distribution
- training data limited in structure -- only allows 4 torso panels for a shirt, for example
- partial support for complex garment details
- no final sewing instruction generation yet

Patterns currently work best when they:

- use non-stretch woven fabrics
- contain relatively simple panel structures
- avoid highly decorative construction details such as pockets

---

# Example Pipeline

<!-- incorporate this into the Usage section I have above -->

## 1. Extract Pattern Panels

```bash
python src/pattern_lasso_v2.py
```

## 2. Label Panels + Match Seams

Export structured garment JSON.

## 3. Run Retargeting

```bash
./run_lasso_to_model.sh \
  pattern_project.json \
  source_measurements_gc.yaml \
  target_measurements_gc.yaml \
  checkpoint.pt \
  output_dir
```

## 4. Export SVG Output

Predicted garment panels are exported as SVG for visualization.

---
## Dataset / External Resources

<!-- this should be in the how it works section -->

This project uses garment geometry and body measurement data derived from the GarmentCode dataset and framework.

GarmentCode:
- https://github.com/maria-korosteleva/GarmentCode

If you use this repository for research purposes, please also cite the original GarmentCode work.

The machine learning models in this repository were trained on processed garment/body data generated from GarmentCode assets.

# Research Direction

This project investigates where learned geometric priors may be useful in sewing pattern adaptation.

Traditional grading methods work well for many standardized garments, but learned models may help preserve higher-level design characteristics in garments with:

- complex silhouettes
- stylistic ease variations
- unconventional proportions
- non-standardized construction logic

The project focuses on combining human-guided extraction with learned geometric retargeting.

---

# Requirements

<!--
This is all gross.  You should have a requirements.txt file that lists the dependencies.
-->
Core dependencies:

```bash
pip install \
    numpy \
    torch \
    PyQt6 \
    opencv-python \
    mapbox-earcut \
    matplotlib \
    pyyaml
```

---

# Author

<!-- do not include this -->
Senior thesis / research prototype exploring machine learning approaches for sewing pattern retargeting and garment geometry processing.
