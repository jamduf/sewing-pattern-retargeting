
# Panel Mapping & Retargeting Pipeline

Converts labeled sewing pattern geometry into model-ready panel tensors and predicts resized garment geometry conditioned on body measurements.

---

# Pipeline Overview

```text
lasso JSON
    ↓
semantic panel mapping
    ↓
body-conditioned model inference
    ↓
predicted garment geometry
    ↓
SVG export
```

---

# Requirements

```bash
pip install \
    numpy \
    torch \
    matplotlib \
    pyyaml
```

Additional utilities:

```bash
pip install cairosvg
```

---

# Required Inputs

## 1. Lasso Project JSON

Exported from:

```bash
pattern_lasso_v2.py
```

The JSON must contain:

- labeled panels
- polygon geometry
- seam relationships

Example labels:

```text
left_ftorso
right_ftorso
left_btorso
right_btorso
left_sleeve_f
```

---

## 2. Body Measurement YAML Files

Two measurement files are required:

- source body measurements
- target body measurements

Example:

```yaml
body:
  bust: 89.0
  waist: 71.0
  hips: 97.0
  shoulder_w: 38.0
  arm_length: 56.0
```

The required measurement fields must match the GarmentCode body measurement format. In the future, I hope to implement some sort of 3D scan system to easily get target measurements for the user. 

Templates are provided in:

```text
size_charts/
```

---

## 3. Trained Checkpoint

Current checkpoints:

```text
pattern_retarget_shirt_v2.pt
pattern_retarget_pants_only_v1.pt
```

---

# Running Inference

Example:

```bash
./run_lasso_to_model.sh \
  pattern_project.json \
  source_measurements.yaml \
  target_measurements.yaml \
  checkpoint.pt \
  output_dir
```

---

# Outputs

The pipeline exports:

```text
source_from_lasso_specification.json
predicted_specification.json
source_from_lasso.svg
predicted_pattern.svg
meta.json
```

---

# Example

<p align="center">
  <img src="../../../docs/images/outputex1.png" width="420">
  <img src="../../../docs/images/outputex2.png" width="420">
</p>

Left: extracted source pattern  
Right: predicted resized geometry

---

# Current Limitations

The current prototype works best when:

- garment panel structures resemble training data
- patterns contain relatively few decorative details
- garment patterns are intended for non-stretch woven fabrics

Known limitations:

- limited training distribution
- simplified polygon resampling
- partial support for complex sleeves/collars
- no seam allowance generation yet

---

# Notes

The current retargeting pipeline uses conservative blending between source geometry and predicted geometry to stabilize outputs for out-of-distribution sewing patterns.
