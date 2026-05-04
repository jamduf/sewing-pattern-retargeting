from __future__ import annotations

import math
from typing import List

from pattern_core_models import PatternProject


def _edge_length(polyline):
    if len(polyline) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(polyline)):
        x0, y0 = polyline[i - 1]
        x1, y1 = polyline[i]
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def validate_project(project: PatternProject) -> List[str]:
    errors: List[str] = []

    panel_ids = {p.id for p in project.panels}
    edge_ids = {e.id for p in project.panels for e in p.edges}

    if not project.panels:
        errors.append("Project has no panels.")

    for panel in project.panels:
        if len(panel.vertices) < 3:
            errors.append(f"{panel.name}: fewer than 3 anchor vertices.")
        if len(panel.polygon) < 3:
            errors.append(f"{panel.name}: polygon has fewer than 3 points.")
        if len(panel.edges) != len(panel.vertices):
            errors.append(f"{panel.name}: edge count does not match vertex count.")

        for e in panel.edges:
            if len(e.polyline) < 2:
                errors.append(f"{panel.name}/{e.id}: edge polyline too short.")
            if _edge_length(e.polyline) <= 1e-6:
                errors.append(f"{panel.name}/{e.id}: zero-length edge.")

            if e.edge_type == "seam" and not e.pair_edge_id:
                errors.append(f"{panel.name}/{e.id}: seam edge has no pair.")

            if e.pair_edge_id and e.pair_edge_id not in edge_ids:
                errors.append(f"{panel.name}/{e.id}: pair_edge_id points to missing edge.")

            if e.edge_type == "fold" and e.pair_edge_id:
                errors.append(f"{panel.name}/{e.id}: fold edge should not also be seam-paired.")

        if panel.mirrored_from_panel_id and panel.mirrored_from_panel_id not in panel_ids:
            errors.append(f"{panel.name}: mirrored_from_panel_id points to missing panel.")

    for seam in project.seams:
        if seam.panel_a_id not in panel_ids:
            errors.append(f"{seam.id}: panel_a_id missing.")
        if seam.panel_b_id not in panel_ids:
            errors.append(f"{seam.id}: panel_b_id missing.")
        if seam.edge_a_id not in edge_ids:
            errors.append(f"{seam.id}: edge_a_id missing.")
        if seam.edge_b_id not in edge_ids:
            errors.append(f"{seam.id}: edge_b_id missing.")

    return errors

