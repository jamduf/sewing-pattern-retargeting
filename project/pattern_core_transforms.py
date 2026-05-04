from __future__ import annotations

import copy
import math
from typing import Tuple

from pattern_core_models import PanelAnnotation


def _transform_point(pt, fn):
    x, y = pt
    return fn(float(x), float(y))


def _panel_bbox(panel: PanelAnnotation):
    pts = panel.polygon if panel.polygon else panel.vertices
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def duplicate_panel(panel: PanelAnnotation, new_panel_id: str, new_name: str) -> PanelAnnotation:
    out = copy.deepcopy(panel)
    out.id = new_panel_id
    out.name = new_name
    out.mirrored_from_panel_id = None

    # Re-id edges to avoid collisions
    for i, e in enumerate(out.edges):
        e.id = f"{new_panel_id}_edge_{i:02d}"
        e.pair_edge_id = None
    return out


def translate_panel(panel: PanelAnnotation, dx: float, dy: float) -> PanelAnnotation:
    def fn(x, y):
        return (x + dx, y + dy)

    panel.vertices = [_transform_point(p, fn) for p in panel.vertices]
    panel.polygon = [_transform_point(p, fn) for p in panel.polygon]

    for e in panel.edges:
        e.polyline = [_transform_point(p, fn) for p in e.polyline]

    if panel.grainline is not None:
        a, b = panel.grainline
        panel.grainline = (_transform_point(a, fn), _transform_point(b, fn))

    return panel


def flip_panel_horizontal(panel: PanelAnnotation) -> PanelAnnotation:
    xmin, ymin, xmax, ymax = _panel_bbox(panel)
    cx = 0.5 * (xmin + xmax)

    def fn(x, y):
        return (2 * cx - x, y)

    panel.vertices = [_transform_point(p, fn) for p in panel.vertices]
    panel.polygon = [_transform_point(p, fn) for p in panel.polygon]

    for e in panel.edges:
        e.polyline = [_transform_point(p, fn) for p in e.polyline]

    if panel.grainline is not None:
        a, b = panel.grainline
        panel.grainline = (_transform_point(a, fn), _transform_point(b, fn))

    return panel


def flip_panel_vertical(panel: PanelAnnotation) -> PanelAnnotation:
    xmin, ymin, xmax, ymax = _panel_bbox(panel)
    cy = 0.5 * (ymin + ymax)

    def fn(x, y):
        return (x, 2 * cy - y)

    panel.vertices = [_transform_point(p, fn) for p in panel.vertices]
    panel.polygon = [_transform_point(p, fn) for p in panel.polygon]

    for e in panel.edges:
        e.polyline = [_transform_point(p, fn) for p in e.polyline]

    if panel.grainline is not None:
        a, b = panel.grainline
        panel.grainline = (_transform_point(a, fn), _transform_point(b, fn))

    return panel


def rotate_panel(panel: PanelAnnotation, angle_deg: float) -> PanelAnnotation:
    xmin, ymin, xmax, ymax = _panel_bbox(panel)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    th = math.radians(angle_deg)
    ct = math.cos(th)
    st = math.sin(th)

    def fn(x, y):
        xr = x - cx
        yr = y - cy
        return (cx + ct * xr - st * yr, cy + st * xr + ct * yr)

    panel.vertices = [_transform_point(p, fn) for p in panel.vertices]
    panel.polygon = [_transform_point(p, fn) for p in panel.polygon]

    for e in panel.edges:
        e.polyline = [_transform_point(p, fn) for p in e.polyline]

    if panel.grainline is not None:
        a, b = panel.grainline
        panel.grainline = (_transform_point(a, fn), _transform_point(b, fn))

    return panel
