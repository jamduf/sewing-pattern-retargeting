import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_test import GarmentPairDataset, PatternRetargetNet


def closed(poly: np.ndarray) -> np.ndarray:
    if poly.shape[0] == 0:
        return poly
    if np.allclose(poly[0], poly[-1]):
        return poly
    return np.vstack([poly, poly[:1]])


def mean_panel_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for i in range(mask.shape[0]):
        if mask[i] <= 0.5:
            continue
        vals.append(np.linalg.norm(a[i] - b[i], axis=1).mean())
    return float(np.mean(vals)) if vals else float("nan")


def per_panel_metrics(src_y, pred_y, tgt_y, panel_mask, panel_order):
    out = {}
    for i, name in enumerate(panel_order):
        if panel_mask[i] <= 0.5:
            continue
        pred = float(np.linalg.norm(pred_y[i] - tgt_y[i], axis=1).mean())
        base = float(np.linalg.norm(src_y[i] - tgt_y[i], axis=1).mean())
        out[name] = {
            "pred_l2_mean": pred,
            "baseline_l2_mean": base,
            "improvement": base - pred,
        }
    return out


def save_overlay_plot(src_y, pred_y, tgt_y, panel_mask, panel_order, out_path, title=""):
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, name in enumerate(panel_order):
        if panel_mask[i] <= 0.5:
            continue
        s = closed(src_y[i])
        p = closed(pred_y[i])
        t = closed(tgt_y[i])

        ax.plot(s[:, 0], s[:, 1], linewidth=1.0, alpha=0.7, label="source" if i == 0 else None)
        ax.plot(t[:, 0], t[:, 1], linewidth=1.0, alpha=0.7, label="target" if i == 0 else None)
        ax.plot(p[:, 0], p[:, 1], linewidth=1.0, alpha=0.7, label="pred" if i == 0 else None)

    ax.set_title(title or "source vs target vs pred")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_dataset(batch_dirs, k_points, limit_per_batch, filter_mode):
    datasets = []
    for batch_dir in batch_dirs:
        ds = GarmentPairDataset(
            batch_dir=batch_dir,
            k_points=k_points,
            limit=limit_per_batch,
            filter_mode=filter_mode,
        )
        datasets.append((batch_dir, ds))
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batch_dirs",
        nargs="+",
        required=True,
        help="One or more held-out batch dirs like .../garments_5000_3",
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--limit_per_batch", type=int, default=None)
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--max_plots", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    filter_mode = ckpt.get("filter_mode", "shirt")
    k_points = int(ckpt["k_points"])
    ckpt_body_keys = ckpt["body_keys"]
    ckpt_panel_order = ckpt["panel_order"]

    datasets = make_dataset(
        batch_dirs=args.batch_dirs,
        k_points=k_points,
        limit_per_batch=args.limit_per_batch,
        filter_mode=filter_mode,
    )

    model = PatternRetargetNet(
        body_dim=len(ckpt_body_keys),
        num_panels=len(ckpt_panel_order),
        k_points=k_points,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    all_pred = []
    all_base = []
    plot_count = 0

    for batch_dir, ds in datasets:
        if ds.body_keys != ckpt_body_keys:
            raise ValueError(f"body_keys mismatch for {batch_dir}")
        if ds.panel_order != ckpt_panel_order:
            raise ValueError(f"panel_order mismatch for {batch_dir}")

        for idx in range(len(ds)):
            sample = ds[idx]
            src_body = sample["src_body"].unsqueeze(0).to(device)
            tgt_body = sample["tgt_body"].unsqueeze(0).to(device)
            src_y = sample["src_y"].unsqueeze(0).to(device)
            tgt_y = sample["tgt_y"].unsqueeze(0).to(device)
            panel_mask = sample["panel_mask"].unsqueeze(0).to(device)

            with torch.no_grad():
                pred_y = model(src_body, tgt_body, src_y, panel_mask)

            src_np = sample["src_y"].cpu().numpy()
            tgt_np = sample["tgt_y"].cpu().numpy()
            pred_np = pred_y[0].cpu().numpy()
            mask_np = sample["panel_mask"].cpu().numpy()

            pred_err = mean_panel_l2(pred_np, tgt_np, mask_np)
            base_err = mean_panel_l2(src_np, tgt_np, mask_np)
            improvement = base_err - pred_err
            improved = improvement > 0

            row = {
                "batch_dir": str(batch_dir),
                "index_in_dataset": idx,
                "pred_l2_mean": pred_err,
                "baseline_l2_mean": base_err,
                "improvement": improvement,
                "improved": int(improved),
                "num_valid_panels": int((mask_np > 0.5).sum()),
            }
            rows.append(row)

            if np.isfinite(pred_err):
                all_pred.append(pred_err)
            if np.isfinite(base_err):
                all_base.append(base_err)

            if plot_count < args.max_plots:
                plot_path = plots_dir / f"example_{plot_count:03d}.png"
                title = f"{Path(batch_dir).name} idx={idx} pred={pred_err:.5f} base={base_err:.5f}"
                save_overlay_plot(
                    src_np, pred_np, tgt_np, mask_np, ckpt_panel_order, plot_path, title=title
                )
                plot_count += 1

    if not rows:
        raise RuntimeError("No evaluation examples found.")

    pred_arr = np.array(all_pred, dtype=np.float64)
    base_arr = np.array(all_base, dtype=np.float64)
    imp_arr = base_arr - pred_arr

    summary = {
        "checkpoint": args.checkpoint,
        "filter_mode": filter_mode,
        "num_examples": len(rows),
        "num_batches": len(args.batch_dirs),
        "mean_pred_l2": float(np.mean(pred_arr)),
        "mean_baseline_l2": float(np.mean(base_arr)),
        "mean_improvement": float(np.mean(imp_arr)),
        "median_improvement": float(np.median(imp_arr)),
        "percent_improved": float(100.0 * np.mean(imp_arr > 0)),
        "max_plots_saved": int(plot_count),
        "batch_dirs": args.batch_dirs,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "per_example_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "batch_dir",
                "index_in_dataset",
                "pred_l2_mean",
                "baseline_l2_mean",
                "improvement",
                "improved",
                "num_valid_panels",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
