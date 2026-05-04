from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Literal, Any


Point2 = Tuple[float, float]

EdgeType = Literal[
    "cut",
    "seam",
    "fold",
    "hem",
    "dart",
    "internal",
    "unknown",
]

MarkerType = Literal[
    "notch",
    "drill_hole",
    "button",
    "buttonhole",
    "grainline_start",
    "grainline_end",
    "pleat_line",
    "stitch_point",
    "waist_marker",
    "collar_marker",
    "custom",
]


@dataclass
class MarkerAnnotation:
    id: str
    marker_type: MarkerType
    position: Point2
    panel_id: str
    edge_id: Optional[str] = None
    note: str = ""


@dataclass
class EdgeAnnotation:
    id: str
    start_vertex_idx: int
    end_vertex_idx: int
    polyline: List[Point2]
    edge_type: EdgeType = "unknown"
    label: str = ""
    pair_edge_id: Optional[str] = None
    mirrored_from_edge_id: Optional[str] = None
    notch_positions: List[float] = field(default_factory=list)
    stitch_label: str = ""
    is_curved: bool = False
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PanelAnnotation:
    id: str
    name: str
    vertices: List[Point2]
    edges: List[EdgeAnnotation]
    polygon: List[Point2]
    category: str = ""
    cut_on_fold: bool = False
    mirrored_from_panel_id: Optional[str] = None
    grainline: Optional[Tuple[Point2, Point2]] = None
    transform: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeamConnection:
    id: str
    panel_a_id: str
    edge_a_id: str
    panel_b_id: str
    edge_b_id: str
    seam_type: str = "stitch"
    allowance_mm: Optional[float] = None
    easing: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternProject:
    version: str
    source_image_path: str
    image_size: Tuple[int, int]
    panels: List[PanelAnnotation] = field(default_factory=list)
    seams: List[SeamConnection] = field(default_factory=list)
    markers: List[MarkerAnnotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def next_panel_id(self) -> str:
        return f"panel_{len(self.panels):03d}"

    def next_seam_id(self) -> str:
        return f"seam_{len(self.seams):03d}"

    def next_marker_id(self) -> str:
        return f"marker_{len(self.markers):03d}"

    def get_panel(self, panel_id: str) -> Optional[PanelAnnotation]:
        for p in self.panels:
            if p.id == panel_id:
                return p
        return None

    def get_edge(self, edge_id: str) -> Optional[EdgeAnnotation]:
        for p in self.panels:
            for e in p.edges:
                if e.id == edge_id:
                    return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
