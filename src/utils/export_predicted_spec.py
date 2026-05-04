import argparse
import json
import tarfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from model_test import GarmentPairDataset, PatternRetargetNet, read_text_from_tar


def list_element_ids(tar_path: str):
    ids = set()
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name
            if name.count("/") < 2:
                continue
            parts = name.split("/")
            if len(parts) >= 3:
                el = parts[1]
                if el:
                    ids.add(el)
    return sorted(ids)


def parse_design_meta(design_yaml_text: str):
    import yaml
    d = yaml.safe_load(design_yaml_text)["design"]
    meta = d.get("meta", {})
    upper = meta.get("upper", {}).get("v", None)
    bottom = meta.get("bottom", {}).get("v", None)
    return upper, bottom


def is_shirt_example(design_yaml_text: str) -> bool:
    upper, bottom = parse_design_meta(design_yaml_text)
    return upper == "Shirt" and bottom is None


def get_filtered_element_ids(batch_dir: str, filter_mode="shirt"):
    batch_dir = Path(batch_dir)
    default_tar = str(batch_dir / "default_body" / "data.tar.gz")
    random_tar = str(batch_dir / "random_body" / "data.tar.gz")

    default_ids = set(list_element_ids(default_tar))
    random_ids = set(list_element_ids(random_tar))
    shared = sorted(default_ids & random_ids)

    filtered = []
    with tarfile.open(default_tar, "r:gz") as tf:
        for el in shared:
            try:
                design_txt = read_text_from_tar(tf, f"./{el}/{el}_design_params.yaml")
                keep = True
                if filter_mode == "shirt":
                    keep = is_shirt_example(design_txt)
                if keep:
                    filtered.append(el)
            except Exception:
                continue
    return filtered


def build_eval_split_indices(n, seed=42):
    n_train = int(0.9 * n)
    n_val = n - n_train
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    return train_idx, val_idx


def load_raw_specs(batch_dir: str, element_id: str):
    batch_dir = Path(batch_dir)
    default_tar = str(batch_dir / "default_body" / "data.tar.gz")
    random_tar = str(batch_dir / "random_body" / "data.tar.gz")

    with tarfile.open(default_tar, "r:gz") as tf_def, tarfile.open(random_tar, "r:gz") as tf_rand:
        src_spec_txt = read_text_from_tar(tf_def, f"./{element_id}/{element_id}_specification.json")
        tgt_spec_txt = read_text_from_tar(tf_rand, f"./{element_id}/{element_id}_specification.json")

    return json.loads(src_spec_txt), json.loads(tgt_spec_txt)


def denormalize_prediction(pred_y, src_y_norm, src_spec, panel_order, panel_mask):
    """
    Training normalized src_y/tgt_y by source garment bbox:
        y_norm = (y - center) / scale

    We reconstruct center/scale from the source spec vertices, then map predicted
    normalized points back to source spec coordinate space.
    """
    valid_pts = []

    for i, panel_name in enumerate(panel_order):
        if panel_mask[i] <= 0.5:
            continue
        if panel_name not in src_spec["pattern"]["panels"]:
            continue
        verts = np.asarray(src_spec["pattern"]["panels"][panel_name]["vertices"], dtype=np.float32)
        if len(verts) == 0:
            continue
        valid_pts.append(verts)

    if not valid_pts:
        raise ValueError("No valid source vertices found for denormalization")

    pts = np.concatenate(valid_pts, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = max(float((maxs - mins).max()), 1e-6)

    pred_abs = pred_y * scale + center
    return pred_abs


def replace_vertices_in_spec(src_spec, pred_abs, panel_order, panel_mask):
    """
    Replace each panel's vertices with predicted sampled boundary points.
    Keep all other fields intact.
    """
    out = deepcopy(src_spec)
    panels = out["pattern"]["panels"]

    for i, panel_name in enumerate(panel_order):
        if panel_mask[i] <= 0.5:
            continue
        if panel_name not in panels:
            continue

        pred_poly = pred_abs[i]
        pred_poly_list = [[float(x), float(y)] for x, y in pred_poly]

        panels[panel_name]["vertices"] = pred_poly_list

        # Rewrite edges as simple consecutive polygon edges.
        # This drops original curvature metadata in the exported prediction.
        n = len(pred_poly_list)
        new_edges = []
        for j in range(n):
            new_edges.append({
                "endpoints": [j, (j + 1) % n]
            })
        panels[panel_name]["edges"] = new_edges

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val_index", type=int, default=0, help="Index within held-out val split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default="exported_prediction")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    filter_mode = ckpt.get("filter_mode", "shirt")
    k_points = int(ckpt["k_points"])
    panel_order = ckpt["panel_order"]
    body_keys = ckpt["body_keys"]

    ds = GarmentPairDataset(
        batch_dir=args.batch_dir,
        k_points=k_points,
        limit=args.limit,
        filter_mode=filter_mode,
    )

    if ds.panel_order != panel_order:
        raise ValueError("Checkpoint panel_order does not match dataset panel_order")
    if ds.body_keys != body_keys:
        raise ValueError("Checkpoint body_keys do not match dataset body_keys")

    # Reconstruct the held-out validation indices exactly like evaluation/training split
    filtered_ids = get_filtered_element_ids(args.batch_dir, filter_mode=filter_mode)
    if args.limit is not None:
        filtered_ids = filtered_ids[:args.limit]

    if len(filtered_ids) != len(ds):
        raise ValueError("Filtered element count does not match dataset length")

    _, val_idx = build_eval_split_indices(len(ds), seed=42)

    if args.val_index < 0 or args.val_index >= len(val_idx):
        raise IndexError(f"val_index must be in [0, {len(val_idx)-1}]")

    ds_index = val_idx[args.val_index]
    element_id = filtered_ids[ds_index]

    sample = ds[ds_index]

    model = PatternRetargetNet(
        body_dim=len(ds.body_keys),
        num_panels=len(ds.panel_order),
        k_points=ds.k_points,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    src_body = sample["src_body"].unsqueeze(0).to(device)
    tgt_body = sample["tgt_body"].unsqueeze(0).to(device)
    src_y = sample["src_y"].unsqueeze(0).to(device)
    panel_mask = sample["panel_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_y = model(src_body, tgt_body, src_y, panel_mask)

    src_y_np = sample["src_y"].cpu().numpy()
    pred_y_np = pred_y[0].cpu().numpy()
    panel_mask_np = sample["panel_mask"].cpu().numpy()

    src_spec, tgt_spec = load_raw_specs(args.batch_dir, element_id)
    pred_abs = denormalize_prediction(pred_y_np, src_y_np, src_spec, panel_order, panel_mask_np)
    pred_spec = replace_vertices_in_spec(src_spec, pred_abs, panel_order, panel_mask_np)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "predicted_specification.json", "w", encoding="utf-8") as f:
        json.dump(pred_spec, f, indent=2)

    with open(out_dir / "source_specification.json", "w", encoding="utf-8") as f:
        json.dump(src_spec, f, indent=2)

    with open(out_dir / "target_specification.json", "w", encoding="utf-8") as f:
        json.dump(tgt_spec, f, indent=2)

    meta = {
        "element_id": element_id,
        "batch_dir": args.batch_dir,
        "checkpoint": args.checkpoint,
        "filter_mode": filter_mode,
        "dataset_index": int(ds_index),
        "val_index": int(args.val_index),
        "k_points": int(k_points),
        "num_valid_panels": int((panel_mask_np > 0.5).sum()),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote files to: {out_dir}")
    print(f"element_id: {element_id}")


if __name__ == "__main__":
    main()
