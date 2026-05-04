#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def load_spec(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_transformed_panels(spec):
    """
    Returns list of:
      {
        "name": panel_name,
        "points": [(x,y), ...]   # translated 2D panel vertices
      }
    """
    panels = spec["pattern"]["panels"]
    out = []

    for name, panel in panels.items():
        verts = panel.get("vertices", [])
        trans = panel.get("translation", [0.0, 0.0, 0.0])

        tx = float(trans[0]) if len(trans) > 0 else 0.0
        ty = float(trans[1]) if len(trans) > 1 else 0.0

        pts = []
        for v in verts:
            x = float(v[0]) + tx
            y = float(v[1]) + ty
            pts.append((x, y))

        if pts:
            out.append({"name": name, "points": pts})

    return out


def compute_viewbox(panels, pad=10.0):
    xs = []
    ys = []
    for p in panels:
        for x, y in p["points"]:
            xs.append(x)
            ys.append(y)

    if not xs:
        return (0.0, 0.0, 100.0, 100.0)

    xmin = min(xs) - pad
    xmax = max(xs) + pad
    ymin = min(ys) - pad
    ymax = max(ys) + pad
    return xmin, ymin, xmax - xmin, ymax - ymin


def panel_to_path(points):
    if not points:
        return ""
    cmds = [f"M {points[0][0]},{points[0][1]}"]
    for x, y in points[1:]:
        cmds.append(f"L {x},{y}")
    cmds.append("Z")
    return " ".join(cmds)


def build_svg(spec, stroke_width=1.0, show_labels=True):
    panels = collect_transformed_panels(spec)
    xmin, ymin, width, height = compute_viewbox(panels, pad=10.0)

    parts = []
    parts.append('<?xml version="1.0" encoding="utf-8" ?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'baseProfile="full" '
        f'width="{width}" height="{height}" '
        f'viewBox="{xmin} {ymin} {width} {height}">'
    )
    parts.append("  <defs/>")

    for p in panels:
        path_d = panel_to_path(p["points"])
        parts.append(
            f'  <path d="{path_d}" '
            f'fill="rgb(227,175,186)" '
            f'stroke="rgb(51,51,51)" '
            f'stroke-width="{stroke_width}"/>'
        )

        if show_labels:
            cx = sum(x for x, _ in p["points"]) / len(p["points"])
            cy = sum(y for _, y in p["points"]) / len(p["points"])
            parts.append(
                f'  <text x="{cx}" y="{cy}" '
                f'font-size="6" text-anchor="middle" '
                f'fill="rgb(20,20,20)">{escape(p["name"])}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to specification.json")
    ap.add_argument("--out_svg", required=True, help="Output SVG path")
    ap.add_argument("--stroke_width", type=float, default=1.0)
    ap.add_argument("--hide_labels", action="store_true")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    svg = build_svg(
        spec,
        stroke_width=args.stroke_width,
        show_labels=not args.hide_labels,
    )

    out_path = Path(args.out_svg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)

    print(f"Wrote SVG: {out_path}")


if __name__ == "__main__":
    main()
