from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from pattern_core_models import (
    PatternProject,
    PanelAnnotation,
    EdgeAnnotation,
    MarkerAnnotation,
    SeamConnection,
)


def save_project_json(project: PatternProject, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(project), f, indent=2)


def load_project_json(path: str) -> PatternProject:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    panels = []
    for p in data.get("panels", []):
        edges = [EdgeAnnotation(**e) for e in p.get("edges", [])]
        panels.append(
            PanelAnnotation(
                id=p["id"],
                name=p["name"],
                vertices=p["vertices"],
                edges=edges,
                polygon=p["polygon"],
                category=p.get("category", ""),
                cut_on_fold=p.get("cut_on_fold", False),
                mirrored_from_panel_id=p.get("mirrored_from_panel_id"),
                grainline=p.get("grainline"),
                transform=p.get("transform", {}),
                notes=p.get("notes", {}),
            )
        )

    seams = [SeamConnection(**s) for s in data.get("seams", [])]
    markers = [MarkerAnnotation(**m) for m in data.get("markers", [])]

    return PatternProject(
        version=data["version"],
        source_image_path=data.get("source_image_path", ""),
        image_size=tuple(data.get("image_size", [0, 0])),
        panels=panels,
        seams=seams,
        markers=markers,
        metadata=data.get("metadata", {}),
    )
