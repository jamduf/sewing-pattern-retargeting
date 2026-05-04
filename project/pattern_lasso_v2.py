#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import math
import struct
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import mapbox_earcut as earcut

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QWidget, QMessageBox
)

from pattern_core_models import (
    PatternProject,
    PanelAnnotation,
    EdgeAnnotation,
    SeamConnection,
)
from pattern_core_transforms import (
    duplicate_panel,
    flip_panel_horizontal,
    flip_panel_vertical,
    translate_panel,
    rotate_panel,
)
from pattern_core_validation import validate_project
from pattern_core_io import save_project_json


# -----------------------------
# GarmentCode-style PLY writer
# -----------------------------

def write_ply_binary(path: str, Vxyz_f32: np.ndarray, UVst_f64: np.ndarray, F_i32: np.ndarray) -> None:
    Vxyz_f32 = np.asarray(Vxyz_f32, dtype=np.float32)
    UVst_f64 = np.asarray(UVst_f64, dtype=np.float64)
    F_i32 = np.asarray(F_i32, dtype=np.int32)

    nV = Vxyz_f32.shape[0]
    nF = F_i32.shape[0]

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "comment https://github.com/mikedh/trimesh",
        f"element vertex {nV}",
        "property float x",
        "property float y",
        "property float z",
        "property double s",
        "property double t",
        f"element face {nF}",
        "property list uchar int vertex_indices",
        "end_header\n"
    ]).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        for i in range(nV):
            x, y, z = map(float, Vxyz_f32[i])
            s, t = map(float, UVst_f64[i])
            f.write(struct.pack("<fffdd", x, y, z, s, t))
        for a, b, c in F_i32:
            f.write(struct.pack("<Biii", 3, int(a), int(b), int(c)))


def polygon_to_mesh(poly_xy: np.ndarray, z: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    poly_xy: (N,2) float64, not closed (no repeated last point)
    returns Vxyz float32 (N,3), UV float64 (N,2), F int32 (M,3)
    """
    coords = np.asarray(poly_xy, dtype=np.float64)
    if coords.shape[0] < 3:
        raise ValueError("Polygon needs at least 3 points")

    ring_ends = np.array([coords.shape[0]], dtype=np.uint32)
    tri = earcut.triangulate_float64(coords, ring_ends).astype(np.int64)
    F = tri.reshape(-1, 3).astype(np.int32)

    V = np.zeros((coords.shape[0], 3), dtype=np.float32)
    V[:, 0:2] = coords.astype(np.float32)
    V[:, 2] = np.float32(z)

    # Simple per-panel UV: normalize within bounding box to [0,1]
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    dx = max(float(xmax - xmin), 1e-9)
    dy = max(float(ymax - ymin), 1e-9)
    s = (coords[:, 0] - xmin) / dx
    t = (coords[:, 1] - ymin) / dy
    UV = np.stack([s, t], axis=1).astype(np.float64)

    return V, UV, F


def simplify_rdp(points: np.ndarray, eps: float) -> np.ndarray:
    """
    Ramer–Douglas–Peucker polyline simplification.
    points: (N,2)
    eps: distance threshold
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 2:
        return pts

    def dist_point_line(p, a, b):
        ap = p - a
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 <= 1e-12:
            return float(np.linalg.norm(ap))
        t = float(np.dot(ap, ab) / ab2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def rdp(seg):
        a = seg[0]
        b = seg[-1]
        max_d = -1.0
        idx = -1
        for i in range(1, len(seg) - 1):
            d = dist_point_line(seg[i], a, b)
            if d > max_d:
                max_d = d
                idx = i
        if max_d > eps:
            left = rdp(seg[:idx + 1])
            right = rdp(seg[idx:])
            return np.vstack([left[:-1], right])
        else:
            return np.vstack([a, b])

    return rdp(pts)


# -----------------------------
# Magnetic lasso / Live-wire
# -----------------------------

@dataclass
class EdgeField:
    cost: np.ndarray  # float32, low on edges
    w: int
    h: int


def compute_edge_cost(bgr: np.ndarray) -> EdgeField:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strong edges in sewing patterns: use Canny + gradient magnitude
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / (grad.max() + 1e-6)

    strength = np.clip(0.65 * grad + 0.35 * edges, 0.0, 1.0)

    # Convert to cost: edges => low cost
    cost = (1.0 - strength) + 0.05
    cost = cost.astype(np.float32)

    h, w = cost.shape
    return EdgeField(cost=cost, w=w, h=h)


def astar_path(cost: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    A* on 8-neighbor grid within the given cost array.
    cost: (H,W) float32
    start/goal: (x,y) pixel indices in this local grid
    returns list of (x,y) from start..goal
    """
    W = cost.shape[1]
    H = cost.shape[0]

    sx, sy = start
    gx, gy = goal
    sx = int(np.clip(sx, 0, W - 1))
    sy = int(np.clip(sy, 0, H - 1))
    gx = int(np.clip(gx, 0, W - 1))
    gy = int(np.clip(gy, 0, H - 1))

    def hfun(x, y):
        return math.hypot(gx - x, gy - y)

    nbrs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    INF = 1e30
    gscore = np.full((H, W), INF, dtype=np.float64)
    came = np.full((H, W, 2), -1, dtype=np.int32)
    closed = np.zeros((H, W), dtype=np.uint8)

    gscore[sy, sx] = 0.0
    heap = [(hfun(sx, sy), 0.0, sx, sy)]

    while heap:
        f, g, x, y = heapq.heappop(heap)
        if closed[y, x]:
            continue
        closed[y, x] = 1

        if x == gx and y == gy:
            break

        for dx, dy in nbrs:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= W or ny < 0 or ny >= H:
                continue
            if closed[ny, nx]:
                continue

            step = float(cost[ny, nx])
            if dx != 0 and dy != 0:
                step *= 1.4142

            ng = g + step
            if ng < gscore[ny, nx]:
                gscore[ny, nx] = ng
                came[ny, nx, 0] = x
                came[ny, nx, 1] = y
                heapq.heappush(heap, (ng + hfun(nx, ny), ng, nx, ny))

    path = []
    x, y = gx, gy
    path.append((x, y))
    for _ in range(W * H):
        px, py = came[y, x]
        if px < 0:
            break
        x, y = int(px), int(py)
        path.append((x, y))
        if x == sx and y == sy:
            break
    path.reverse()
    return path


# -----------------------------
# Qt UI
# -----------------------------

class PatternCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.img_bgr: Optional[np.ndarray] = None
        self.qimg: Optional[QImage] = None
        self.edge: Optional[EdgeField] = None

        # Current tracing state
        self.anchors: List[Tuple[int, int]] = []
        self.fixed_path: List[Tuple[int, int]] = []
        self.fixed_segments: List[List[Tuple[int, int]]] = []  # one snapped path per edge
        self.preview_path: List[Tuple[int, int]] = []
        self.closed = False

        # Structured pattern project
        self.project = PatternProject(
            version="2.0",
            source_image_path="",
            image_size=(0, 0),
        )
        self.selected_panel_id: Optional[str] = None
        self.selected_edge_ref: Optional[Tuple[str, str]] = None  # (panel_id, edge_id)

        self.seam_pair_mode = False
        self.pending_edge_ref: Optional[Tuple[str, str]] = None  # (panel_id, edge_id)
        # Settings
        self.window_radius = 180
        self.simplify_eps = 1.5
        self.max_preview_rate = 10
        self._move_ctr = 0

    # -------------------------
    # Image/project setup
    # -------------------------

    def load_image(self, path: str):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        self.img_bgr = bgr
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.edge = compute_edge_cost(bgr)

        self.project.source_image_path = path
        self.project.image_size = (w, h)

        self.setFixedSize(w, h)
        self.setFocus()
        self.reset_current()
        self.update()

    def reset_current(self):
        self.anchors = []
        self.fixed_path = []
        self.fixed_segments = []
        self.preview_path = []
        self.closed = False

    def _edge_color(self, edge):
        # pending first click in seam mode
        if self.pending_edge_ref is not None and edge.id == self.pending_edge_ref[1]:
            return QColor(255, 140, 0)   # orange

        # paired seam edges
        if edge.edge_type == "seam" and edge.pair_edge_id:
            return QColor(150, 0, 200)   # purple

        # fold edges
        if edge.edge_type == "fold":
            return QColor(0, 170, 255)   # cyan

        # default
        return QColor(0, 220, 90)

    def _dist_point_to_segment(self, px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab2 = abx * abx + aby * aby
        if ab2 <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = (apx * abx + apy * aby) / ab2
        t = max(0.0, min(1.0, t))
        qx = ax + t * abx
        qy = ay + t * aby
        return math.hypot(px - qx, py - qy)

    def _nearest_edge(self, x: float, y: float, max_dist: float = 8.0) -> Optional[Tuple[str, str]]:
        best = None
        best_d = max_dist

        for p in self.project.panels:
            for e in p.edges:
                pts = e.polyline
                for i in range(1, len(pts)):
                    ax, ay = pts[i - 1]
                    bx, by = pts[i]
                    d = self._dist_point_to_segment(x, y, ax, ay, bx, by)
                    if d < best_d:
                        best_d = d
                        best = (p.id, e.id)
        return best

    def _create_seam_connection(self, ref_a: Tuple[str, str], ref_b: Tuple[str, str]):
        panel_a_id, edge_a_id = ref_a
        panel_b_id, edge_b_id = ref_b

        if edge_a_id == edge_b_id:
            QMessageBox.warning(self, "Seam", "Cannot pair an edge with itself.")
            return

        edge_a = self.project.get_edge(edge_a_id)
        edge_b = self.project.get_edge(edge_b_id)
        if edge_a is None or edge_b is None:
            QMessageBox.warning(self, "Seam", "Could not find one of the selected edges.")
            return

        seam_id = self.project.next_seam_id()
        edge_a.edge_type = "seam"
        edge_b.edge_type = "seam"
        edge_a.pair_edge_id = edge_b.id
        edge_b.pair_edge_id = edge_a.id

        self.project.seams.append(
            SeamConnection(
                id=seam_id,
                panel_a_id=panel_a_id,
                edge_a_id=edge_a_id,
                panel_b_id=panel_b_id,
                edge_b_id=edge_b_id,
                seam_type="stitch",
            )
        )

    # -------------------------
    # Geometry helpers
    # -------------------------

    def _current_polygon_xy(self) -> Optional[np.ndarray]:
        if not self.closed or len(self.fixed_path) < 3:
            return None

        pts = np.array(self.fixed_path, dtype=np.float64)
        simp = simplify_rdp(pts, self.simplify_eps)

        if simp.shape[0] >= 2 and np.allclose(simp[0], simp[-1]):
            simp = simp[:-1]
        return simp

    def _build_current_panel_annotation(self) -> Optional[PanelAnnotation]:
        if not self.closed or len(self.anchors) < 3:
            return None
        if len(self.fixed_segments) != len(self.anchors):
            return None

        polygon = self._current_polygon_xy()
        if polygon is None or polygon.shape[0] < 3:
            return None

        panel_id = self.project.next_panel_id()
        panel_name = panel_id
        vertices = [(float(x), float(y)) for (x, y) in self.anchors]

        edges: List[EdgeAnnotation] = []
        for i, seg in enumerate(self.fixed_segments):
            polyline = [(float(x), float(y)) for (x, y) in seg]
            edges.append(
                EdgeAnnotation(
                    id=f"{panel_id}_edge_{i:02d}",
                    start_vertex_idx=i,
                    end_vertex_idx=(i + 1) % len(vertices),
                    polyline=polyline,
                    edge_type="unknown",
                    label="",
                )
            )

        panel = PanelAnnotation(
            id=panel_id,
            name=panel_name,
            vertices=vertices,
            edges=edges,
            polygon=[(float(x), float(y)) for (x, y) in polygon],
        )
        return panel

    def _panel_bbox(self, panel: PanelAnnotation):
        pts = panel.polygon if panel.polygon else panel.vertices
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def _point_in_panel_bbox(self, x: float, y: float, panel: PanelAnnotation, pad: float = 4.0) -> bool:
        xmin, ymin, xmax, ymax = self._panel_bbox(panel)
        return (xmin - pad) <= x <= (xmax + pad) and (ymin - pad) <= y <= (ymax + pad)

    def _select_panel_at(self, x: float, y: float) -> bool:
        # select top-most / latest panel whose bbox contains point
        for p in reversed(self.project.panels):
            if self._point_in_panel_bbox(x, y, p):
                self.selected_panel_id = p.id
                self.update()
                return True
        return False

    # -------------------------
    # Export helpers
    # -------------------------

    def _export_single(self, poly: np.ndarray):
        out_ply = QFileDialog.getSaveFileName(self, "Save panel PLY", "panel_boxmesh.ply", "PLY (*.ply)")[0]
        if not out_ply:
            return
        V, UV, F = polygon_to_mesh(poly, z=0.0)
        write_ply_binary(out_ply, V, UV, F)
        QMessageBox.information(self, "Export", f"Wrote:\n{out_ply}\nV={len(V)} F={len(F)}")

    def _export_project_json(self):
        if not self.project.panels:
            QMessageBox.warning(self, "Export", "No panels to export.")
            return

        errs = validate_project(self.project)
        if errs:
            msg = "Validation warnings/errors:\n\n" + "\n".join(errs[:20])
            QMessageBox.warning(self, "Validation", msg)

        out_json = QFileDialog.getSaveFileName(
            self, "Save pattern project JSON", "pattern_project.json", "JSON (*.json)"
        )[0]
        if not out_json:
            return

        save_project_json(self.project, out_json)
        QMessageBox.information(self, "Export", f"Wrote:\n{out_json}")

    def _export_all(self):
        polys = [np.array(p.polygon, dtype=np.float64) for p in self.project.panels]
        labels = [p.name for p in self.project.panels]

        if self.closed:
            panel = self._build_current_panel_annotation()
            if panel is not None:
                polys.append(np.array(panel.polygon, dtype=np.float64))
                labels.append(panel.name)

        if not polys:
            QMessageBox.warning(self, "Export", "No panels to export.")
            return

        out_ply = QFileDialog.getSaveFileName(self, "Save combined boxmesh PLY", "boxmesh.ply", "PLY (*.ply)")[0]
        if not out_ply:
            return
        out_seg = os.path.splitext(out_ply)[0] + "_segmentation.txt"

        allV = []
        allUV = []
        allF = []
        seg_lines = []
        v_off = 0

        for poly, lab in zip(polys, labels):
            V, UV, F = polygon_to_mesh(poly, z=0.0)
            allV.append(V)
            allUV.append(UV)
            allF.append(F + v_off)
            seg_lines.extend([lab] * V.shape[0])
            v_off += V.shape[0]

        Vc = np.vstack(allV).astype(np.float32)
        UVc = np.vstack(allUV).astype(np.float64)
        Fc = np.vstack(allF).astype(np.int32)

        write_ply_binary(out_ply, Vc, UVc, Fc)
        with open(out_seg, "w", encoding="utf-8") as f:
            for s in seg_lines:
                f.write(s + "\n")

        QMessageBox.information(self, "Export", f"Wrote:\n{out_ply}\n{out_seg}\nV={len(Vc)} F={len(Fc)}")

    # -------------------------
    # Live-wire
    # -------------------------

    def _snap_between(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Run A* in a local window around p0/p1 for speed.
        Returns list of (x,y) in full-image coordinates.
        """
        assert self.edge is not None
        cost_full = self.edge.cost
        W = self.edge.w
        H = self.edge.h

        x0, y0 = p0
        x1, y1 = p1

        xmin = max(0, min(x0, x1) - self.window_radius)
        xmax = min(W - 1, max(x0, x1) + self.window_radius)
        ymin = max(0, min(y0, y1) - self.window_radius)
        ymax = min(H - 1, max(y0, y1) + self.window_radius)

        local = cost_full[ymin:ymax + 1, xmin:xmax + 1]
        start = (x0 - xmin, y0 - ymin)
        goal = (x1 - xmin, y1 - ymin)

        path_local = astar_path(local, start, goal)
        path_full = [(x + xmin, y + ymin) for (x, y) in path_local]
        return path_full

    # -------------------------
    # Event handling
    # -------------------------

    def mousePressEvent(self, ev):
        self.setFocus()
        if self.qimg is None or self.edge is None:
            return

        x = int(ev.position().x())
        y = int(ev.position().y())

        if ev.button() == Qt.MouseButton.RightButton:
            # Undo last anchor in current trace
            if self.anchors:
                self.anchors.pop()
                if self.fixed_segments:
                    self.fixed_segments.pop()

                # Recompute flattened fixed path from remaining segments
                if self.anchors:
                    self.fixed_path = [self.anchors[0]]
                else:
                    self.fixed_path = []

                for seg in self.fixed_segments:
                    if seg:
                        self.fixed_path.extend(seg[1:])

                self.preview_path = []
                self.closed = False
                self.update()
            return

        if ev.button() == Qt.MouseButton.LeftButton:
            # If not actively tracing, use click for selection first
            if self.seam_pair_mode and not self.anchors:
                edge_ref = self._nearest_edge(x, y)
                if edge_ref is not None:
                    if self.pending_edge_ref is None:
                        self.pending_edge_ref = edge_ref
                    else:
                        self._create_seam_connection(self.pending_edge_ref, edge_ref)
                        self.pending_edge_ref = None
                    self.update()
                    return

            if not self.anchors and self._select_panel_at(x, y):
                return

            if self.closed:
                return

            self.anchors.append((x, y))
            if len(self.anchors) == 1:
                self.fixed_path = [(x, y)]
            else:
                seg = self._snap_between(self.anchors[-2], self.anchors[-1])
                if seg:
                    self.fixed_segments.append(seg[:])
                    self.fixed_path.extend(seg[1:])  # flatten for display only
            self.preview_path = []
            self.update()

    def mouseMoveEvent(self, ev):
        if self.qimg is None or self.edge is None:
            return
        if not self.anchors or self.closed:
            return

        self._move_ctr += 1
        if self._move_ctr % self.max_preview_rate != 0:
            return

        x = int(ev.position().x())
        y = int(ev.position().y())
        self.preview_path = self._snap_between(self.anchors[-1], (x, y))
        self.update()

    def keyPressEvent(self, ev):
        print("KEY:", ev.key())
        if self.qimg is None:
            return

        key = ev.key()

        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            # Close current polygon by snapping last anchor to first
            if len(self.anchors) >= 3 and not self.closed:
                seg = self._snap_between(self.anchors[-1], self.anchors[0])
                if seg:
                    self.fixed_segments.append(seg[:])
                    self.fixed_path.extend(seg[1:])
                self.closed = True
                self.preview_path = []
                self.update()

        elif key == Qt.Key.Key_N:
            # Commit current closed polygon as structured panel
            if self.closed:
                panel = self._build_current_panel_annotation()
                if panel is not None:
                    self.project.panels.append(panel)
                    self.selected_panel_id = panel.id
                self.reset_current()
                self.update()

        elif key == Qt.Key.Key_E:
            if self.closed:
                poly = self._current_polygon_xy()
                if poly is not None:
                    self._export_single(poly)

        elif key == Qt.Key.Key_X:
            self._export_all()

        elif key == Qt.Key.Key_J:
            self._export_project_json()

        elif key == Qt.Key.Key_R:
            self.reset_current()
            self.update()

        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.simplify_eps = max(0.1, self.simplify_eps - 0.3)
            self.update()

        elif key == Qt.Key.Key_Minus or key == Qt.Key.Key_Underscore:
            self.simplify_eps = self.simplify_eps + 0.3
            self.update()

        elif key == Qt.Key.Key_D:
            if self.selected_panel_id is not None:
                src = self.project.get_panel(self.selected_panel_id)
                if src is not None:
                    new_id = self.project.next_panel_id()
                    dup = duplicate_panel(src, new_id, new_id)
                    translate_panel(dup, 20.0, 20.0)
                    self.project.panels.append(dup)
                    self.selected_panel_id = dup.id
                    self.update()

        elif key == Qt.Key.Key_H:
            if self.selected_panel_id is not None:
                p = self.project.get_panel(self.selected_panel_id)
                if p is not None:
                    flip_panel_horizontal(p)
                    self.update()

        elif key == Qt.Key.Key_V:
            if self.selected_panel_id is not None:
                p = self.project.get_panel(self.selected_panel_id)
                if p is not None:
                    flip_panel_vertical(p)
                    self.update()

        elif key == Qt.Key.Key_BracketLeft:
            if self.selected_panel_id is not None:
                p = self.project.get_panel(self.selected_panel_id)
                if p is not None:
                    rotate_panel(p, -5.0)
                    self.update()

        elif key == Qt.Key.Key_S:
            self.seam_pair_mode = not self.seam_pair_mode
            self.pending_edge_ref = None
            self.update()

        elif key == Qt.Key.Key_BracketRight:
            if self.selected_panel_id is not None:
                p = self.project.get_panel(self.selected_panel_id)
                if p is not None:
                    rotate_panel(p, 5.0)
                    self.update()

        elif key == Qt.Key.Key_O:
            if self.selected_panel_id is not None:
                p = self.project.get_panel(self.selected_panel_id)
                if p is not None:
                    p.cut_on_fold = not p.cut_on_fold
                    self.update()

        elif key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            if self.selected_panel_id is not None:
                self.project.panels = [p for p in self.project.panels if p.id != self.selected_panel_id]
                self.selected_panel_id = None
                self.update()

    # -------------------------
    # Painting
    # -------------------------

    def paintEvent(self, ev):
        if self.qimg is None:
            return

        painter = QPainter(self)
        painter.drawImage(0, 0, self.qimg)

        # Draw committed structured panels edge-by-edge
        for p in self.project.panels:
            is_selected = (p.id == self.selected_panel_id)

            for e in p.edges:
                color = self._edge_color(e)
                width = 3 if is_selected else 2
                painter.setPen(QPen(color, width))

                pts = e.polyline
                for i in range(1, len(pts)):
                    painter.drawLine(
                        QPointF(pts[i - 1][0], pts[i - 1][1]),
                        QPointF(pts[i][0], pts[i][1])
                    )

            if e.edge_type == "seam" and e.pair_edge_id and len(pts) >= 2:
                    mx = sum(pt[0] for pt in pts) / len(pts)
                    my = sum(pt[1] for pt in pts) / len(pts)
                    painter.setPen(QPen(QColor(80, 0, 120), 1))
                    painter.drawText(int(mx), int(my), f"↔ {e.pair_edge_id}")

            # label panel at centroid-ish average
            poly = p.polygon
            if poly:
                cx = sum(pt[0] for pt in poly) / len(poly)
                cy = sum(pt[1] for pt in poly) / len(poly)
                painter.setPen(QPen(QColor(20, 20, 20), 1))
                suffix = " [fold]" if p.cut_on_fold else ""
                painter.drawText(int(cx), int(cy), p.name + suffix)

        # Draw current anchors
        pen_anchor = QPen(QColor(255, 80, 80), 6)
        painter.setPen(pen_anchor)
        for (x, y) in self.anchors:
            painter.drawPoint(QPointF(x, y))

        # Draw fixed snapped current path
        pen_path = QPen(QColor(0, 140, 255), 2)
        painter.setPen(pen_path)
        for i in range(1, len(self.fixed_path)):
            x0, y0 = self.fixed_path[i - 1]
            x1, y1 = self.fixed_path[i]
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # Draw preview path
        pen_prev = QPen(QColor(255, 200, 0), 2)
        painter.setPen(pen_prev)
        for i in range(1, len(self.preview_path)):
            x0, y0 = self.preview_path[i - 1]
            x1, y1 = self.preview_path[i]
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # Status text
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        msg = (
            f"Anchors: {len(self.anchors)} | Closed: {self.closed} | "
            f"Panels saved: {len(self.project.panels)} | "
            f"simplify_eps: {self.simplify_eps:.1f}px"
        )
        painter.drawText(10, 20, msg)
        painter.drawText(
            10, 40,
            "LMB add/select | RMB undo | Enter close | N save panel | "
            "D duplicate | H/V flip | [/] rotate | O fold | "
            "J save json | E export current | X export all | R reset | +/- simplify"
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Magnetic Lasso v2 → Structured Panels + PLY")
        self.canvas = PatternCanvas()
        self.setCentralWidget(self.canvas)
        self._open_on_start()

    def _open_on_start(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open pattern image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            QMessageBox.information(self, "Info", "No image selected. Close and re-run to pick one.")
            return
        try:
            self.canvas.load_image(path)
            self.resize(self.canvas.width(), self.canvas.height())
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
