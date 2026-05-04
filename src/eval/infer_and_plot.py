#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

from model_test import GarmentPairDataset, PatternRetargetNet


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def closed(poly: np.ndarray) -> np.ndarray:
    if poly.shape[0] == 0:
        return poly
    if np.allclose(poly[0], poly[-1]):
        return poly
    return np.vstack([poly, poly[:1]])


def masked_panel_errors(src_y, pred_y, tgt_y, panel_mask):
    """
    Returns dict with:
      - per_panel_l2
      - per_panel_baseline_l2
      - overall_l2
      - overall_baseline_l2
    """
    src_y = to_numpy(src_y)
    pred_y = to_numpy(pred_y)
    tgt_y = to_numpy(tgt_y)
    panel_mask = to_numpy(panel_mask).astype(bool)

    per_panel = {}
    all_pred = []
    all_base = []

    for i in range(panel_mask.shape[0]):
        if not panel_mask[i]:
            continue
        pred_err = np.linalg.norm(pred_y[i] - tgt_y[i], axis=1).mean()
        base_err = np.linalg.norm(src_y[i] - tgt_y[i], axis=1).mean()
        per_panel[i] = {
            "pred_l2_mean": float(pred_err),
            "baseline_src_to_tgt_l2_mean": float(base_err),
            "improvement": float(base_err - pred_err),
        }
        all_pred.append(pred_err)
        all_base.append(base_err)

    overall = {
        "overall_pred_l2_mean": float(np.mean(all_pred)) if all_pred else None,
        "overall_baseline_src_to_tgt_l2_mean": float(np.mean(all_base)) if all_base else None,
        "overall_improvement": float(np.mean(all_base) - np.mean(all_pred)) if all_pred else None,
    }
    return per_panel, overall


def plot_overlay_all(src_y, pred_y, tgt_y, panel_mask, panel_order, out_path):
    src_y = to_numpy(src_y)
    pred_y = to_numpy(pred_y)
    tgt_y = to_numpy(tgt_y)
    panel_mask = to_numpy(panel_mask).astype(bool)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i, name in enumerate(panel_order):
        if not panel_mask[i]:
            continue
        s = closed(src_y[i])
        p = closed(pred_y[i])
        t = closed(tgt_y[i])

        ax.plot(s[:, 0], s[:, 1], linewidth=1.0, alpha=0.8, label="source" if i == 0 else None)
        ax.plot(t[:, 0], t[:, 1], linewidth=1.0, alpha=0.8, label="target" if i == 0 else None)
        ax.plot(p[:, 0], p[:, 1], linewidth=1.0, alpha=0.8, label="pred" if i == 0 else None)

    ax.set_title("All panels overlay: source vs target vs prediction")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_panel_grid(src_y, pred_y, tgt_y, panel_mask, panel_order, out_path, max_cols=4):
    src_y = to_numpy(src_y)
    pred_y = to_numpy(pred_y)
    tgt_y = to_numpy(tgt_y)
    panel_mask = to_numpy(panel_mask).astype(bool)

    valid_idxs = [i for i in range(len(panel_order)) if panel_mask[i]]
    if not valid_idxs:
        raise ValueError("No valid panels to plot")

    n = len(valid_idxs)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for ax in axes.ravel():
        ax.axis("off")

    for plot_idx, panel_idx in enumerate(valid_idxs):
        r = plot_idx // cols
        c = plot_idx % cols
        ax = axes[r, c]
        ax.axis("on")

        s = closed(src_y[panel_idx])
        p = closed(pred_y[panel_idx])
        t = closed(tgt_y[panel_idx])

        ax.plot(s[:, 0], s[:, 1], linewidth=1.5, label="source")
        ax.plot(t[:, 0], t[:, 1], linewidth=1.5, label="target")
        ax.plot(p[:, 0], p[:, 1], linewidth=1.5, label="pred")

        ax.set_title(panel_order[panel_idx], fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Per-panel comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_dir", required=True, help="Path like .../GarmentCodeData_v2/garments_5000_0")
    ap.add_argument("--checkpoint", default="pattern_retarget_baseline.pt")
    ap.add_argument("--k_points", type=int, default=None, help="Optional override; normally read from checkpoint")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val_index", type=int, default=0, help="Which validation example to visualize")
    ap.add_argument("--out_dir", default="infer_plots")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_k = int(ckpt["k_points"])
    k_points = args.k_points if args.k_points is not None else ckpt_k

    filter_mode = ckpt.get("filter_mode", "shirt")
    ds = GarmentPairDataset(
        args.batch_dir,
        k_points=k_points,
        limit=args.limit,
        filter_mode=filter_mode,
    )

    # Consistency checks
    if ds.body_keys != ckpt["body_keys"]:
        raise ValueError("Dataset body_keys do not match checkpoint body_keys")
    if ds.panel_order != ckpt["panel_order"]:
        raise ValueError("Dataset panel_order does not match checkpoint panel_order")
    if ds.k_points != ckpt_k:
        raise ValueError("Dataset k_points do not match checkpoint k_points")

    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    if args.val_index < 0 or args.val_index >= len(val_ds):
        raise IndexError(f"val_index must be in [0, {len(val_ds)-1}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatternRetargetNet(
        body_dim=len(ds.body_keys),
        num_panels=len(ds.panel_order),
        k_points=ds.k_points,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sample = val_ds[args.val_index]
    src_body = sample["src_body"].unsqueeze(0).to(device)
    tgt_body = sample["tgt_body"].unsqueeze(0).to(device)
    src_y = sample["src_y"].unsqueeze(0).to(device)
    tgt_y = sample["tgt_y"].unsqueeze(0).to(device)
    panel_mask = sample["panel_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_y = model(src_body, tgt_body, src_y, panel_mask)

    src_y_np = src_y[0].cpu().numpy()
    tgt_y_np = tgt_y[0].cpu().numpy()
    pred_y_np = pred_y[0].cpu().numpy()
    panel_mask_np = panel_mask[0].cpu().numpy()

    per_panel, overall = masked_panel_errors(src_y_np, pred_y_np, tgt_y_np, panel_mask_np)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_overlay_all(
        src_y_np, pred_y_np, tgt_y_np, panel_mask_np,
        ds.panel_order, out_dir / "panel_overlay_all.png"
    )
    plot_panel_grid(
        src_y_np, pred_y_np, tgt_y_np, panel_mask_np,
        ds.panel_order, out_dir / "panel_grid.png"
    )

    metrics = {
        "val_index": args.val_index,
        "num_valid_panels": int(np.sum(panel_mask_np > 0.5)),
        "overall": overall,
        "per_panel": {
            ds.panel_order[i]: per_panel[i]
            for i in per_panel
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved plots and metrics to: {out_dir}")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
