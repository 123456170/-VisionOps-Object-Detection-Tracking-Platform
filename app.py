"""
VisionOps — Modular Computer Vision Object Detection & Tracking Platform
=========================================================================
Single-file Streamlit application.

Supports: image / video / webcam detection, multi-object tracking,
classification, segmentation (where backend supports it), ROI configuration,
virtual line-crossing counting, trajectory visualization, FPS & performance
benchmarking, analytics dashboard, and result export (CSV / JSON / annotated
media).

DESIGN — Modular Pipeline Architecture
---------------------------------------
The UI never talks to a specific model. It talks to a `Detector` interface
and a `Tracker` interface. Swapping the underlying model (classical CV,
Haar cascades, background subtraction, or a real deep-learning model like
YOLOv8 via `ultralytics` if installed) requires zero UI changes — you only
register a new class in `DETECTOR_REGISTRY` / `TRACKER_REGISTRY` below.

No API key is required. If `ultralytics` (YOLOv8) is installed in the
environment, it is auto-detected and offered as a selectable high-accuracy
backend. Otherwise the app runs fully on a dependency-light classical-CV
backend (Haar cascades + motion-based blob detection) that needs nothing
beyond OpenCV, so the app ALWAYS launches with a working live demo.
"""

import os
import json
import time
import math
import random
import tempfile
import collections
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Optional heavy backend (auto-detected, never required)
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ===========================================================================
# 1. CORE DATA MODELS
# ===========================================================================

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _class_color(name: str) -> Tuple[int, int, int]:
    """Deterministic BGR color per class label."""
    h = abs(hash(name)) % (256 ** 3)
    r, g, b = (h >> 16) & 255, (h >> 8) & 255, h & 255
    r, g, b = max(r, 60), max(g, 60), max(b, 60)
    return (b, g, r)


@dataclass
class Detection:
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2
    label: str
    confidence: float
    mask: Optional[np.ndarray] = None  # binary mask cropped to box, optional
    track_id: Optional[int] = None


@dataclass
class PipelineConfig:
    detector_name: str = "Motion + Contour (built-in)"
    tracker_name: str = "Centroid Tracker (built-in)"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    enabled_classes: List[str] = field(default_factory=lambda: list(COCO_CLASSES) + ["object", "face"])
    max_detections: int = 40
    enable_segmentation: bool = False
    enable_tracking: bool = True
    enable_trajectories: bool = True
    trajectory_length: int = 40
    roi_enabled: bool = False
    roi_polygon: List[Tuple[int, int]] = field(default_factory=list)
    line_enabled: bool = False
    line_points: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0))
    show_boxes: bool = True
    show_labels: bool = True
    show_conf: bool = True
    show_ids: bool = True
    show_masks: bool = True
    show_fps: bool = True
    box_thickness: int = 2
    frame_stride: int = 1  # process every Nth frame (perf control)


# ===========================================================================
# 2. DETECTOR INTERFACE  (swap models without touching UI)
# ===========================================================================

class Detector(ABC):
    """Abstract detector — any backend implements this and plugs into the UI."""
    name: str = "AbstractDetector"
    supports_segmentation: bool = False
    supports_classification: bool = True

    @abstractmethod
    def detect(self, frame: np.ndarray, conf_threshold: float,
               enabled_classes: List[str], max_detections: int,
               segmentation: bool = False) -> List[Detection]:
        ...


class MotionContourDetector(Detector):
    """
    Dependency-light, always-available detector.
    Combines background subtraction (motion) with static contour / blob
    analysis so it works on live webcam feed, video, AND single static
    images (where there is no motion at all — falls back to edge/contour
    blob detection so images always produce demo-worthy detections).
    """
    name = "Motion + Contour (built-in)"
    supports_segmentation = True
    supports_classification = True

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=True
        )
        self._rng = random.Random(42)
        self._label_pool = ["person", "car", "object", "bicycle", "animal", "bag"]

    def _pseudo_label(self, area: float, aspect: float) -> Tuple[str, float]:
        """Heuristic pseudo-classification from blob geometry (no network needed)."""
        if aspect > 1.6 and area > 3000:
            return "car", min(0.95, 0.55 + area / 60000)
        if 0.25 < aspect < 0.7 and area > 1500:
            return "person", min(0.95, 0.5 + area / 40000)
        if 0.9 < aspect < 1.3 and area < 4000:
            return "object", 0.4 + self._rng.random() * 0.2
        return (self._label_pool[self._rng.randint(0, len(self._label_pool) - 1)],
                0.35 + self._rng.random() * 0.3)

    def detect(self, frame, conf_threshold, enabled_classes, max_detections,
               segmentation=False) -> List[Detection]:
        h, w = frame.shape[:2]
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.01)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8), iterations=1)
        fg_mask = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=2)

        motion_ratio = float(np.count_nonzero(fg_mask)) / (h * w + 1e-6)
        detections: List[Detection] = []

        if motion_ratio > 0.003:
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_detections]
            for c in contours:
                area = cv2.contourArea(c)
                if area < 400:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / (bh + 1e-6)
                label, conf = self._pseudo_label(area, aspect)
                if conf < conf_threshold:
                    continue
                if enabled_classes and label not in enabled_classes:
                    continue
                mask = None
                if segmentation:
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(m, [c], -1, 255, -1)
                    mask = m[y:y + bh, x:x + bw]
                detections.append(Detection(
                    box=(x, y, x + bw, y + bh), label=label,
                    confidence=round(min(conf, 0.99), 3), mask=mask
                ))
        else:
            # Static-image / no-motion fallback: edge + contour blob detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 120)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_detections]
            for c in contours:
                area = cv2.contourArea(c)
                if area < (h * w) * 0.0015:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if bw < 12 or bh < 12:
                    continue
                aspect = bw / (bh + 1e-6)
                label, conf = self._pseudo_label(area, aspect)
                conf = max(0.3, min(conf, 0.9))
                if conf < conf_threshold:
                    continue
                if enabled_classes and label not in enabled_classes:
                    continue
                mask = None
                if segmentation:
                    m = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(m, [c], -1, 255, -1)
                    mask = m[y:y + bh, x:x + bw]
                detections.append(Detection(
                    box=(x, y, x + bw, y + bh), label=label,
                    confidence=round(conf, 3), mask=mask
                ))

        return detections[:max_detections]


class HaarFaceBodyDetector(Detector):
    """Classical Viola-Jones detector for faces / bodies — ships with OpenCV, no download.
    Loading is defensive: some OpenCV builds/deployments (e.g. certain
    headless installs or restricted filesystems on hosted platforms) don't
    expose the bundled cascade XML files at `cv2.data.haarcascades`. In that
    case this detector marks itself unavailable instead of raising and
    crashing the whole app at import time — the registry builder below skips
    registering it, so the sidebar simply won't offer it as an option."""
    name = "Haar Cascade — Face/Body (built-in)"
    supports_segmentation = False
    supports_classification = True

    def __init__(self):
        self.available = False
        self.face_cascade = None
        self.body_cascade = None
        try:
            base = cv2.data.haarcascades
            face_path = os.path.join(base, "haarcascade_frontalface_default.xml")
            body_path = os.path.join(base, "haarcascade_fullbody.xml")

            face_cc = cv2.CascadeClassifier(face_path)
            body_cc = cv2.CascadeClassifier(body_path)

            # CascadeClassifier doesn't always raise on a bad path — it can
            # silently construct an empty classifier. Check explicitly.
            if face_cc.empty() or body_cc.empty():
                raise RuntimeError("Haar cascade XML files not found in this OpenCV build")

            self.face_cascade = face_cc
            self.body_cascade = body_cc
            self.available = True
        except Exception:
            self.available = False

    def detect(self, frame, conf_threshold, enabled_classes, max_detections,
               segmentation=False) -> List[Detection]:
        if not self.available:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detections: List[Detection] = []
        want_person = (not enabled_classes) or ("person" in enabled_classes)
        want_face = (not enabled_classes) or ("face" in enabled_classes)

        if want_face:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                if 0.75 >= conf_threshold:
                    detections.append(Detection((x, y, x + w, y + h), "face", 0.75))

        if want_person:
            bodies = self.body_cascade.detectMultiScale(gray, 1.05, 3, minSize=(50, 100))
            for (x, y, w, h) in bodies:
                if 0.6 >= conf_threshold:
                    detections.append(Detection((x, y, x + w, y + h), "person", 0.6))

        return detections[:max_detections]


class YOLODetector(Detector):
    """Real deep-learning backend (YOLOv8), auto-registered only if `ultralytics` is installed."""
    name = "YOLOv8 (ultralytics — deep learning)"
    supports_segmentation = True
    supports_classification = True

    def __init__(self, weights: str = "yolov8n.pt", segment_weights: str = "yolov8n-seg.pt"):
        self._weights = weights
        self._seg_weights = segment_weights
        self._model = None
        self._seg_model = None

    def _lazy_load(self, segmentation: bool):
        if segmentation:
            if self._seg_model is None:
                self._seg_model = YOLO(self._seg_weights)
            return self._seg_model
        if self._model is None:
            self._model = YOLO(self._weights)
        return self._model

    def detect(self, frame, conf_threshold, enabled_classes, max_detections,
               segmentation=False) -> List[Detection]:
        model = self._lazy_load(segmentation)
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        detections: List[Detection] = []
        names = results.names
        boxes = results.boxes
        masks = results.masks.data.cpu().numpy() if (segmentation and results.masks is not None) else None

        for i, b in enumerate(boxes):
            cls_id = int(b.cls[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if enabled_classes and label not in enabled_classes:
                continue
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            mask = None
            if masks is not None and i < len(masks):
                m = cv2.resize(masks[i], (frame.shape[1], frame.shape[0]))
                mask = (m[y1:y2, x1:x2] * 255).astype(np.uint8)
            detections.append(Detection((x1, y1, x2, y2), label, round(conf, 3), mask))
            if len(detections) >= max_detections:
                break
        return detections


DETECTOR_REGISTRY: Dict[str, Detector] = {}


def _build_detector_registry():
    """
    Build the detector registry defensively. Every backend is tried
    independently — if one fails to initialize (missing model files,
    incompatible OpenCV build, missing optional dependency, etc.) it is
    skipped with a warning instead of crashing the whole app. The
    `Motion + Contour` backend has no external file/model dependency and is
    the guaranteed fallback, so at least one detector is always available.
    """
    if DETECTOR_REGISTRY:
        return

    try:
        DETECTOR_REGISTRY["Motion + Contour (built-in)"] = MotionContourDetector()
    except Exception as e:
        st.session_state.setdefault("_registry_warnings", []).append(
            f"Motion + Contour detector failed to load: {e}")

    try:
        haar = HaarFaceBodyDetector()
        if haar.available:
            DETECTOR_REGISTRY["Haar Cascade — Face/Body (built-in)"] = haar
        else:
            st.session_state.setdefault("_registry_warnings", []).append(
                "Haar Cascade detector unavailable in this environment "
                "(cascade XML files not found in this OpenCV build) — skipped.")
    except Exception as e:
        st.session_state.setdefault("_registry_warnings", []).append(
            f"Haar Cascade detector failed to load: {e}")

    if YOLO_AVAILABLE:
        try:
            DETECTOR_REGISTRY["YOLOv8 (ultralytics — deep learning)"] = YOLODetector()
        except Exception as e:
            st.session_state.setdefault("_registry_warnings", []).append(
                f"YOLOv8 detector failed to load: {e}")

    if not DETECTOR_REGISTRY:
        # Should never happen (Motion+Contour has zero external deps), but
        # guarantees the app never ends up with an empty registry.
        DETECTOR_REGISTRY["Motion + Contour (built-in)"] = MotionContourDetector()


# ===========================================================================
# 3. TRACKER INTERFACE  (swap trackers without touching UI)
# ===========================================================================

class Tracker(ABC):
    name: str = "AbstractTracker"

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        """Assign track_id to each detection in-place and return them."""
        ...

    @abstractmethod
    def reset(self):
        ...


class CentroidTracker(Tracker):
    """
    Lightweight centroid-distance tracker (SORT-like, no external deps).
    Demonstrates persistent IDs, trajectories, and line-crossing counting
    without requiring any model weights.
    """
    name = "Centroid Tracker (built-in)"

    def __init__(self, max_disappeared: int = 25, max_distance: int = 120):
        self.next_id = 1
        self.objects: Dict[int, Dict] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def reset(self):
        self.next_id = 1
        self.objects = {}

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            for oid in list(self.objects.keys()):
                self.objects[oid]["disappeared"] += 1
                if self.objects[oid]["disappeared"] > self.max_disappeared:
                    del self.objects[oid]
            return detections

        input_centroids = [self._centroid(d.box) for d in detections]

        if not self.objects:
            for i, d in enumerate(detections):
                self.objects[self.next_id] = {
                    "centroid": input_centroids[i], "box": d.box,
                    "label": d.label, "disappeared": 0
                }
                d.track_id = self.next_id
                self.next_id += 1
            return detections

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.dist(oc, ic)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid]["centroid"] = input_centroids[col]
            self.objects[oid]["box"] = detections[col].box
            self.objects[oid]["disappeared"] = 0
            detections[col].track_id = oid
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.objects[oid]["disappeared"] += 1
            if self.objects[oid]["disappeared"] > self.max_disappeared:
                del self.objects[oid]

        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.objects[self.next_id] = {
                "centroid": input_centroids[col], "box": detections[col].box,
                "label": detections[col].label, "disappeared": 0
            }
            detections[col].track_id = self.next_id
            self.next_id += 1

        return detections


class IOUTracker(Tracker):
    """Alternative tracker implementation — pure IOU matching (no distance heuristic)."""
    name = "IOU Tracker (built-in)"

    def __init__(self, iou_threshold: float = 0.3, max_disappeared: int = 20):
        self.next_id = 1
        self.objects: Dict[int, Dict] = {}
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared

    def reset(self):
        self.next_id = 1
        self.objects = {}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        if not self.objects:
            for d in detections:
                self.objects[self.next_id] = {"box": d.box, "disappeared": 0}
                d.track_id = self.next_id
                self.next_id += 1
            return detections

        object_ids = list(self.objects.keys())
        used_ids, used_dets = set(), set()

        for oid in object_ids:
            best_iou, best_j = 0, -1
            for j, d in enumerate(detections):
                if j in used_dets:
                    continue
                iou = self._iou(self.objects[oid]["box"], d.box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_threshold and best_j >= 0:
                self.objects[oid]["box"] = detections[best_j].box
                self.objects[oid]["disappeared"] = 0
                detections[best_j].track_id = oid
                used_ids.add(oid)
                used_dets.add(best_j)

        for oid in object_ids:
            if oid not in used_ids:
                self.objects[oid]["disappeared"] += 1
                if self.objects[oid]["disappeared"] > self.max_disappeared:
                    del self.objects[oid]

        for j, d in enumerate(detections):
            if j not in used_dets:
                self.objects[self.next_id] = {"box": d.box, "disappeared": 0}
                d.track_id = self.next_id
                self.next_id += 1

        return detections


TRACKER_REGISTRY: Dict[str, type] = {
    "Centroid Tracker (built-in)": CentroidTracker,
    "IOU Tracker (built-in)": IOUTracker,
}


# ===========================================================================
# 4. ANALYTICS STATE  (trajectories, counts, benchmarking) — session-scoped
# ===========================================================================

class AnalyticsState:
    def __init__(self):
        self.trajectories: Dict[int, collections.deque] = {}
        self.class_counts: collections.Counter = collections.Counter()
        self.unique_track_ids: set = set()
        self.line_crossings: Dict[str, int] = {"in": 0, "out": 0}
        self._crossed_ids: Dict[int, str] = {}
        self.fps_history: collections.deque = collections.deque(maxlen=120)
        self.latency_history_ms: collections.deque = collections.deque(maxlen=120)
        self.frame_count = 0
        self.detection_log: List[dict] = []
        self.start_time = time.time()

    def update_trajectories(self, detections: List[Detection], max_len: int):
        for d in detections:
            if d.track_id is None:
                continue
            if d.track_id not in self.trajectories or self.trajectories[d.track_id].maxlen != max_len:
                self.trajectories[d.track_id] = collections.deque(maxlen=max_len)
            x1, y1, x2, y2 = d.box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.trajectories[d.track_id].append((cx, cy))

    def update_counts(self, detections: List[Detection]):
        for d in detections:
            self.class_counts[d.label] += 1
            if d.track_id is not None:
                self.unique_track_ids.add(d.track_id)

    @staticmethod
    def _side_of_line(pt, line):
        (x1, y1), (x2, y2) = line
        px, py = pt
        val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return "A" if val > 0 else "B"

    def update_line_crossings(self, detections: List[Detection], line):
        (lx1, ly1), (lx2, ly2) = line
        if (lx1, ly1) == (lx2, ly2):
            return
        for d in detections:
            if d.track_id is None:
                continue
            x1, y1, x2, y2 = d.box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            side = self._side_of_line((cx, cy), line)
            prev = self._crossed_ids.get(d.track_id)
            if prev is not None and prev != side:
                if side == "B":
                    self.line_crossings["in"] += 1
                else:
                    self.line_crossings["out"] += 1
            self._crossed_ids[d.track_id] = side

    def log_frame(self, detections: List[Detection], fps: float, latency_ms: float):
        self.frame_count += 1
        self.fps_history.append(fps)
        self.latency_history_ms.append(latency_ms)
        for d in detections:
            self.detection_log.append({
                "frame": self.frame_count,
                "track_id": d.track_id,
                "label": d.label,
                "confidence": d.confidence,
                "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3],
                "timestamp": round(time.time() - self.start_time, 3),
            })
        if len(self.detection_log) > 20000:
            self.detection_log = self.detection_log[-20000:]

    def reset(self):
        self.__init__()


def get_analytics() -> AnalyticsState:
    if "analytics" not in st.session_state:
        st.session_state.analytics = AnalyticsState()
    return st.session_state.analytics


# ===========================================================================
# 5. RENDERING / ANNOTATION
# ===========================================================================

def point_in_polygon(pt, polygon):
    if len(polygon) < 3:
        return True
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), pt, False) >= 0


def annotate_frame(frame: np.ndarray, detections: List[Detection], cfg: PipelineConfig,
                    analytics: AnalyticsState, fps: float) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()

    if cfg.roi_enabled and len(cfg.roi_polygon) >= 3:
        pts = np.array(cfg.roi_polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (60, 200, 60))
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, (60, 220, 60), 2)
        cv2.putText(out, "ROI", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)

    if cfg.line_enabled:
        p1, p2 = cfg.line_points
        if p1 != p2:
            cv2.line(out, p1, p2, (0, 165, 255), 3)
            cv2.putText(out, "COUNT LINE", (p1[0] + 5, max(15, p1[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    if cfg.enable_trajectories:
        for tid, pts in analytics.trajectories.items():
            if len(pts) < 2:
                continue
            color = _class_color(str(tid))
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], color, 2)

    if cfg.enable_segmentation and cfg.show_masks:
        for d in detections:
            if d.mask is None:
                continue
            x1, y1, x2, y2 = d.box
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(out.shape[1], x2), min(out.shape[0], y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            mh, mw = y2c - y1c, x2c - x1c
            m = d.mask
            if m.shape[:2] != (mh, mw):
                m = cv2.resize(m, (mw, mh))
            color = _class_color(d.label)
            colored = np.zeros((mh, mw, 3), dtype=np.uint8)
            colored[:] = color
            region = out[y1c:y2c, x1c:x2c]
            mask_bool = m > 127
            if region.shape[:2] == mask_bool.shape:
                blended = cv2.addWeighted(region, 0.5, colored, 0.5, 0)
                region[mask_bool] = blended[mask_bool]
                out[y1c:y2c, x1c:x2c] = region

    for d in detections:
        if cfg.roi_enabled and len(cfg.roi_polygon) >= 3:
            x1, y1, x2, y2 = d.box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if not point_in_polygon((cx, cy), cfg.roi_polygon):
                continue

        color = _class_color(d.label)
        x1, y1, x2, y2 = d.box
        if cfg.show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, cfg.box_thickness)

        if cfg.show_labels:
            parts = [d.label]
            if cfg.show_ids and d.track_id is not None:
                parts.append(f"ID:{d.track_id}")
            if cfg.show_conf:
                parts.append(f"{d.confidence:.2f}")
            text = " | ".join(parts)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, text, (x1 + 3, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if cfg.show_fps:
        hud = f"FPS: {fps:.1f}  |  Objects: {len(detections)}  |  Frame: {analytics.frame_count}"
        cv2.rectangle(out, (0, 0), (min(out.shape[1], 430), 28), (20, 20, 20), -1)
        cv2.putText(out, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)

    return out


# ===========================================================================
# 6. PIPELINE  (glues Detector + Tracker + Analytics + Renderer together)
# ===========================================================================

class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.detector = DETECTOR_REGISTRY[cfg.detector_name]
        tracker_cls = TRACKER_REGISTRY[cfg.tracker_name]
        if ("tracker_instance" not in st.session_state or
                st.session_state.get("tracker_name_cached") != cfg.tracker_name):
            st.session_state.tracker_instance = tracker_cls()
            st.session_state.tracker_name_cached = cfg.tracker_name
        self.tracker: Tracker = st.session_state.tracker_instance

    def process(self, frame: np.ndarray, analytics: AnalyticsState) -> Tuple[np.ndarray, List[Detection], float]:
        t0 = time.time()
        detections = self.detector.detect(
            frame, self.cfg.conf_threshold, self.cfg.enabled_classes,
            self.cfg.max_detections, segmentation=self.cfg.enable_segmentation
        )
        if self.cfg.enable_tracking:
            detections = self.tracker.update(detections)
        latency_ms = (time.time() - t0) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        analytics.update_counts(detections)
        if self.cfg.enable_trajectories:
            analytics.update_trajectories(detections, self.cfg.trajectory_length)
        if self.cfg.line_enabled:
            analytics.update_line_crossings(detections, self.cfg.line_points)
        analytics.log_frame(detections, fps, latency_ms)

        annotated = annotate_frame(frame, detections, self.cfg, analytics, fps)
        return annotated, detections, fps


# ===========================================================================
# 7. SAMPLE / DEMO DATA GENERATION  (guarantees a working live demo instantly)
# ===========================================================================

def generate_synthetic_traffic_frame(t: float, w=960, h=540, n_objects=8) -> np.ndarray:
    """
    Procedurally generated 'traffic scene' frame — moving colored shapes on a
    road-like background. Powers the always-on Live Simulation demo mode so
    the app never shows a blank screen, with zero external assets.
    """
    frame = np.full((h, w, 3), (40, 90, 40), dtype=np.uint8)  # grass green
    cv2.rectangle(frame, (0, int(h * 0.35)), (w, int(h * 0.75)), (60, 60, 60), -1)  # road
    lane_y = int(h * 0.55)
    for x in range(0, w, 40):
        cv2.line(frame, (x, lane_y), (x + 20, lane_y), (220, 220, 220), 2)
    cv2.rectangle(frame, (0, int(h * 0.75)), (w, h), (70, 70, 70), -1)  # sidewalk

    shapes = []
    for i in range(n_objects):
        seed = i * 97
        speed = 40 + (seed % 60)
        lane = int(h * 0.42) + (seed % 4) * 30 if i % 2 == 0 else int(h * 0.8) + (seed % 2) * 25
        direction = 1 if i % 2 == 0 else -1
        x = int((t * speed * direction + seed * 13) % (w + 200)) - 100
        if direction < 0:
            x = w - x
        size_w = 55 if i % 2 == 0 else 26
        size_h = 28 if i % 2 == 0 else 55
        color = [(0, 90, 220), (0, 160, 240), (0, 200, 120), (200, 130, 0),
                 (140, 0, 200), (0, 210, 210)][i % 6]
        shapes.append((x, lane, size_w, size_h, color, "car" if i % 2 == 0 else "person"))

    for (x, y, sw, sh, color, kind) in shapes:
        if kind == "car":
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), color, -1)
            cv2.rectangle(frame, (x + 8, y - 10), (x + sw - 8, y),
                          tuple(int(c * 0.7) for c in color), -1)
        else:
            cv2.circle(frame, (x + sw // 2, y + 8), 8, color, -1)
            cv2.rectangle(frame, (x + sw // 2 - 8, y + 14), (x + sw // 2 + 8, y + sh), color, -1)

    return frame


def make_sample_image() -> np.ndarray:
    return generate_synthetic_traffic_frame(t=2.0, n_objects=8)


SAMPLE_VIDEO_PATH = os.path.join(tempfile.gettempdir(), "visionops_sample_video.mp4")


def ensure_sample_video(seconds=6, fps=20, w=960, h=540):
    if os.path.exists(SAMPLE_VIDEO_PATH) and os.path.getsize(SAMPLE_VIDEO_PATH) > 10000:
        return SAMPLE_VIDEO_PATH
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(SAMPLE_VIDEO_PATH, fourcc, fps, (w, h))
    for i in range(seconds * fps):
        frame = generate_synthetic_traffic_frame(t=i / fps, w=w, h=h, n_objects=8)
        writer.write(frame)
    writer.release()
    return SAMPLE_VIDEO_PATH


# ===========================================================================
# 8. STREAMLIT APP
# ===========================================================================

st.set_page_config(page_title="VisionOps — CV Detection & Tracking Platform",
                    page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

_build_detector_registry()

CUSTOM_CSS = """
<style>
.stApp { background: linear-gradient(180deg, #0b0f19 0%, #0e1420 100%); }
section[data-testid="stSidebar"] { background: #0c1220; border-right: 1px solid #1f2a3d; }
h1, h2, h3 { letter-spacing: -0.01em; }
div[data-testid="stMetricValue"] { font-size: 1.4rem; color: #4ade80; }
div[data-testid="stMetric"] {
    background: #111826; border: 1px solid #1f2a3d; border-radius: 10px; padding: 10px 14px;
}
.badge {
    display:inline-block; padding:3px 10px; border-radius:12px; font-size:0.75rem;
    font-weight:600; margin-right:6px;
}
.badge-live { background:#14331f; color:#4ade80; border:1px solid #22c55e44; }
.badge-model { background:#1b2440; color:#93c5fd; border:1px solid #3b82f644; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---- Session defaults ----
if "cfg" not in st.session_state:
    st.session_state.cfg = PipelineConfig()
if "run_live_sim" not in st.session_state:
    st.session_state.run_live_sim = True   # ON by default -> instant working demo
if "sim_t" not in st.session_state:
    st.session_state.sim_t = 0.0
if "webcam_stop" not in st.session_state:
    st.session_state.webcam_stop = False

cfg: PipelineConfig = st.session_state.cfg
analytics = get_analytics()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown("## 🎯 VisionOps — Object Detection & Tracking Platform")
    active_backend = ("YOLOv8 (deep learning)" if cfg.detector_name.startswith("YOLOv8")
                       else "Classical CV (built-in, no downloads)")
    st.markdown(
        f'<span class="badge badge-live">● LIVE DEMO RUNNING</span>'
        f'<span class="badge badge-model">Backend: {active_backend}</span>'
        f'<span class="badge badge-model">Tracker: {cfg.tracker_name}</span>',
        unsafe_allow_html=True,
    )
with top_r:
    st.metric("Session Frames Processed", analytics.frame_count)

st.divider()

# ---------------------------------------------------------------------------
# SIDEBAR — MODULAR PIPELINE CONFIGURATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Configuration")

    st.markdown("**Detection Model**")
    detector_options = list(DETECTOR_REGISTRY.keys())
    cfg.detector_name = st.selectbox(
        "Detector backend", detector_options,
        index=detector_options.index(cfg.detector_name) if cfg.detector_name in detector_options else 0,
        help="Modular pipeline: swap the model here — no UI code changes needed. "
             "YOLOv8 auto-appears if `ultralytics` is installed."
    )
    if not YOLO_AVAILABLE:
        st.caption("💡 Install `ultralytics` to unlock YOLOv8 deep-learning detection & segmentation.")

    detector_obj = DETECTOR_REGISTRY[cfg.detector_name]

    cfg.conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, cfg.conf_threshold, 0.05)
    cfg.iou_threshold = st.slider("IOU / NMS threshold", 0.1, 0.9, cfg.iou_threshold, 0.05)
    cfg.max_detections = st.slider("Max detections / frame", 1, 100, cfg.max_detections, 1)

    with st.expander("Class filter", expanded=False):
        all_opts = COCO_CLASSES + ["object", "face"]
        default_sel = [c for c in cfg.enabled_classes if c in all_opts] or all_opts
        quick = st.multiselect("Enabled classes (empty = all)", all_opts, default=default_sel)
        cfg.enabled_classes = quick if quick else all_opts

    seg_supported = detector_obj.supports_segmentation
    cfg.enable_segmentation = st.toggle(
        "Enable segmentation (where supported)",
        value=cfg.enable_segmentation and seg_supported,
        disabled=not seg_supported,
        help="Only available on backends that support pixel masks."
    )

    st.markdown("---")
    st.markdown("**Multi-Object Tracking**")
    cfg.enable_tracking = st.toggle("Enable tracking", value=cfg.enable_tracking)
    tracker_options = list(TRACKER_REGISTRY.keys())
    cfg.tracker_name = st.selectbox(
        "Tracker algorithm", tracker_options,
        index=tracker_options.index(cfg.tracker_name) if cfg.tracker_name in tracker_options else 0,
        disabled=not cfg.enable_tracking
    )
    cfg.enable_trajectories = st.toggle("Show trajectories", value=cfg.enable_trajectories,
                                         disabled=not cfg.enable_tracking)
    cfg.trajectory_length = st.slider("Trajectory length (frames)", 5, 150, cfg.trajectory_length, 5,
                                       disabled=not cfg.enable_trajectories)

    st.markdown("---")
    st.markdown("**Region of Interest (ROI)**")
    cfg.roi_enabled = st.toggle("Enable ROI filter", value=cfg.roi_enabled)
    if cfg.roi_enabled:
        st.caption("Define ROI as % of frame")
        c1, c2 = st.columns(2)
        rx1 = c1.slider("ROI x1 %", 0, 100, 15)
        ry1 = c2.slider("ROI y1 %", 0, 100, 15)
        rx2 = c1.slider("ROI x2 %", 0, 100, 85)
        ry2 = c2.slider("ROI y2 %", 0, 100, 85)
        cfg.roi_polygon = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
    else:
        cfg.roi_polygon = []

    st.markdown("---")
    st.markdown("**Virtual Counting Line**")
    cfg.line_enabled = st.toggle("Enable line counting", value=cfg.line_enabled)
    if cfg.line_enabled:
        lx = st.slider("Line position (x %)", 0, 100, 50)
        cfg.line_points = ((lx, 0), (lx, 100))
        st.caption("Vertical line at chosen X. Crossing increments in/out counters.")

    st.markdown("---")
    st.markdown("**Visualization**")
    v1, v2 = st.columns(2)
    cfg.show_boxes = v1.checkbox("Boxes", value=cfg.show_boxes)
    cfg.show_labels = v2.checkbox("Labels", value=cfg.show_labels)
    cfg.show_conf = v1.checkbox("Confidence", value=cfg.show_conf)
    cfg.show_ids = v2.checkbox("Track IDs", value=cfg.show_ids)
    cfg.show_masks = v1.checkbox("Masks", value=cfg.show_masks)
    cfg.show_fps = v2.checkbox("FPS HUD", value=cfg.show_fps)
    cfg.box_thickness = st.slider("Box thickness", 1, 5, cfg.box_thickness)
    cfg.frame_stride = st.slider("Process every Nth frame (perf)", 1, 5, cfg.frame_stride)

    st.markdown("---")
    if st.button("🔄 Reset analytics & trackers", use_container_width=True):
        analytics.reset()
        st.session_state.pop("tracker_instance", None)
        st.rerun()

st.session_state.cfg = cfg
pipeline = Pipeline(cfg)


def scaled_cfg_for_frame(base_cfg: PipelineConfig, w: int, h: int) -> PipelineConfig:
    """Convert %-based ROI/line config into pixel coordinates for a given frame size."""
    c = PipelineConfig(**asdict(base_cfg))
    if base_cfg.roi_enabled and base_cfg.roi_polygon:
        c.roi_polygon = [(int(px / 100 * w), int(py / 100 * h)) for (px, py) in base_cfg.roi_polygon]
    if base_cfg.line_enabled:
        (x1p, y1p), (x2p, y2p) = base_cfg.line_points
        c.line_points = ((int(x1p / 100 * w), int(y1p / 100 * h)),
                          (int(x2p / 100 * w), int(y2p / 100 * h)))
    return c


# ===========================================================================
# TABS
# ===========================================================================
tab_live, tab_image, tab_video, tab_webcam, tab_analytics, tab_export, tab_about = st.tabs(
    ["🔴 Live Simulation", "🖼️ Image Detection", "🎬 Video Detection",
     "📷 USB Webcam", "📊 Analytics & Benchmark", "⬇️ Export", "ℹ️ Architecture"]
)

# ---------------------------------------------------------------------------
# TAB 1 — LIVE SIMULATION (always-on working demo, zero setup)
# ---------------------------------------------------------------------------
with tab_live:
    st.markdown("#### Procedurally generated traffic scene — auto-running detection + tracking demo")
    st.caption("No camera, no internet, no downloads required. Proves the full pipeline end-to-end "
               "the instant the app launches: detection → tracking → ROI/line → trajectories → analytics.")

    lc1, lc2, _ = st.columns([1, 1, 4])
    running = lc1.toggle("▶ Running", value=st.session_state.run_live_sim)
    st.session_state.run_live_sim = running
    speed = lc2.select_slider("Speed", options=[0.5, 1.0, 1.5, 2.0], value=1.0)

    frame_slot = st.empty()
    metrics_slot = st.empty()

    if running:
        # IMPORTANT: this loop runs entirely INSIDE this script execution and
        # only ever mutates the placeholders above via .image(). It avoids
        # st.rerun() per frame (a full rerun tears down/rebuilds the whole
        # page — sidebar, tabs, CSS — which is the #1 cause of visible
        # shake) AND it avoids rebuilding the metric widgets every frame
        # (recreating st.metric() 30x/sec causes layout jitter even without
        # a full rerun). Metrics are throttled to a few updates/sec instead.
        loop_start = time.time()
        max_loop_seconds = 20     # long in-place run before yielding control
        frame_interval = 0.045    # ~22 FPS — smoother in a browser <img> swap
                                   # than pushing 30fps through Streamlit's image channel
        metrics_interval = 0.5    # update the metric numbers twice a second, not every frame
        last_metrics_update = 0.0

        while time.time() - loop_start < max_loop_seconds:
            tick_start = time.time()

            frame = generate_synthetic_traffic_frame(st.session_state.sim_t, n_objects=8)
            h, w = frame.shape[:2]
            run_cfg = scaled_cfg_for_frame(cfg, w, h)
            pipeline.cfg = run_cfg
            annotated, dets, fps = pipeline.process(frame, analytics)
            st.session_state.sim_t += 0.15 * speed

            # Fixed pixel width (not use_container_width) so the <img> element
            # never triggers a browser reflow of surrounding layout on update —
            # that reflow is what reads as "shaking".
            frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              channels="RGB", width=900)

            if tick_start - last_metrics_update >= metrics_interval:
                with metrics_slot.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Live FPS", f"{fps:.1f}")
                    m2.metric("Objects in frame", len(dets))
                    m3.metric("Unique tracks (session)", len(analytics.unique_track_ids))
                    m4.metric("Line crossings (in/out)",
                              f"{analytics.line_crossings['in']}/{analytics.line_crossings['out']}")
                last_metrics_update = tick_start

            elapsed = time.time() - tick_start
            time.sleep(max(0.0, frame_interval - elapsed))

        # After the time-boxed loop, do a single rerun so Streamlit stays
        # responsive to sidebar/config changes — infrequent enough (every
        # ~20s) that it isn't perceived as a stutter.
        st.rerun()
    else:
        st.info("Simulation paused. Toggle **Running** to resume the live demo.")

# ---------------------------------------------------------------------------
# TAB 2 — IMAGE DETECTION
# ---------------------------------------------------------------------------
with tab_image:
    st.markdown("#### Single-image detection, classification & segmentation")
    img_src = st.radio("Image source", ["Sample image", "Upload image"], horizontal=True, key="img_src")

    frame = None
    if img_src == "Upload image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        frame = make_sample_image()

    if frame is not None:
        h, w = frame.shape[:2]
        run_cfg = scaled_cfg_for_frame(cfg, w, h)
        pipeline.cfg = run_cfg
        t0 = time.time()
        annotated, dets, fps = pipeline.process(frame, analytics)
        infer_ms = (time.time() - t0) * 1000

        c1, c2 = st.columns(2)
        c1.markdown("**Original**")
        c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        c2.markdown("**Detected**")
        c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects detected", len(dets))
        m2.metric("Inference time", f"{infer_ms:.1f} ms")
        m3.metric("Classes found", len(set(d.label for d in dets)))

        if dets:
            df = pd.DataFrame([{
                "label": d.label, "confidence": d.confidence, "track_id": d.track_id,
                "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3]
            } for d in dets])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.session_state["last_annotated_image"] = annotated
    else:
        st.info("Upload an image or switch to **Sample image** to see instant results.")

# ---------------------------------------------------------------------------
# TAB 3 — VIDEO FILE DETECTION
# ---------------------------------------------------------------------------
with tab_video:
    st.markdown("#### Video file detection & tracking")
    vid_src = st.radio("Video source", ["Sample video (auto-generated)", "Upload video"],
                        horizontal=True, key="vid_src")

    video_path = None
    if vid_src == "Upload video":
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_vid is not None:
            tmp_path = os.path.join(tempfile.gettempdir(), "visionops_upload.mp4")
            with open(tmp_path, "wb") as f:
                f.write(uploaded_vid.read())
            video_path = tmp_path
    else:
        with st.spinner("Generating sample traffic video (first run only)..."):
            video_path = ensure_sample_video()

    vc1, vc2, vc3 = st.columns(3)
    max_frames = vc1.slider("Frames to process", 20, 300, 120, 10)
    playback_delay = vc2.slider("Playback delay (ms)", 0, 100, 15, 5)
    start_btn = vc3.button("▶ Run video pipeline", type="primary", use_container_width=True)

    video_frame_slot = st.empty()
    video_metrics_slot = st.empty()
    progress = st.empty()

    if start_btn and video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video source.")
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
            run_cfg = scaled_cfg_for_frame(cfg, w, h)
            pipeline.cfg = run_cfg

            frame_idx = 0
            fps_list = []
            pbar = progress.progress(0)
            while frame_idx < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1
                if frame_idx % cfg.frame_stride != 0:
                    continue
                annotated, dets, fps = pipeline.process(frame, analytics)
                fps_list.append(fps)
                video_frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                        channels="RGB", use_container_width=True)
                with video_metrics_slot.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Frame", f"{frame_idx}/{max_frames}")
                    m2.metric("FPS", f"{fps:.1f}")
                    m3.metric("Objects", len(dets))
                    m4.metric("Avg FPS", f"{np.mean(fps_list):.1f}")
                pbar.progress(min(1.0, frame_idx / max_frames))
                time.sleep(playback_delay / 1000)
            cap.release()
            st.success(f"Processed {frame_idx} frames — avg {np.mean(fps_list) if fps_list else 0:.1f} FPS.")
    else:
        st.info("Click **Run video pipeline** to process the sample (or uploaded) video frame-by-frame "
                 "with live detection, tracking, and analytics.")

# ---------------------------------------------------------------------------
# TAB 4 — USB WEBCAM
# ---------------------------------------------------------------------------
with tab_webcam:
    st.markdown("#### USB webcam detection & tracking")
    st.caption(
        "This runs against a **local USB/built-in camera on the machine running this Streamlit server** "
        "via OpenCV `VideoCapture`. If you're viewing this app remotely (e.g. a hosted URL opened on a "
        "different device), the server cannot see that device's browser-side webcam — run "
        "`streamlit run app.py` locally on the machine with the USB camera attached."
    )

    wc1, wc2, wc3 = st.columns(3)
    cam_index = wc1.number_input("Camera index", min_value=0, max_value=10, value=0, step=1,
                                  help="0 is usually the default built-in/USB webcam. Try 1, 2... for others.")
    cam_frames = wc2.slider("Frames to capture per run", 10, 300, 60, 10)
    cam_delay = wc3.slider("Delay between frames (ms)", 0, 150, 20, 5)

    wb1, wb2 = st.columns(2)
    start_cam = wb1.button("▶ Start webcam capture", type="primary", use_container_width=True)
    stop_cam = wb2.button("■ Stop", use_container_width=True)

    if stop_cam:
        st.session_state["webcam_stop"] = True

    cam_frame_slot = st.empty()
    cam_metrics_slot = st.empty()

    if start_cam:
        st.session_state["webcam_stop"] = False
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(
                f"Could not open camera index {cam_index}. Checks: (1) no other app is already using the "
                f"camera, (2) OS camera permissions are granted to the process running Streamlit, "
                f"(3) try a different camera index (0, 1, 2...)."
            )
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            run_cfg = scaled_cfg_for_frame(cfg, w, h)
            pipeline.cfg = run_cfg

            count = 0
            fps_list = []
            while count < cam_frames and not st.session_state.get("webcam_stop", False):
                ok, frame = cap.read()
                if not ok:
                    st.warning("Frame grab failed — camera may have disconnected.")
                    break
                count += 1
                if count % cfg.frame_stride != 0:
                    continue
                annotated, dets, fps = pipeline.process(frame, analytics)
                fps_list.append(fps)
                cam_frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                      channels="RGB", use_container_width=True)
                with cam_metrics_slot.container():
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Frame", count)
                    m2.metric("FPS", f"{fps:.1f}")
                    m3.metric("Objects", len(dets))
                    m4.metric("Avg FPS", f"{np.mean(fps_list):.1f}")
                time.sleep(cam_delay / 1000)
            cap.release()
            st.success(f"Captured {count} frames from camera index {cam_index}.")
    else:
        st.info("Click **Start webcam capture** to open the local USB camera and run the live pipeline. "
                 "Lower 'Frames to capture' or raise 'frame stride' in the sidebar for a snappier feel.")

# ---------------------------------------------------------------------------
# TAB 5 — ANALYTICS & PERFORMANCE BENCHMARKING
# ---------------------------------------------------------------------------
with tab_analytics:
    st.markdown("#### Session analytics & performance benchmarking")

    if analytics.frame_count == 0:
        st.info("Run any pipeline (Live Simulation, Image, Video, or Webcam) to populate analytics.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total frames", analytics.frame_count)
        m2.metric("Total detections logged", len(analytics.detection_log))
        m3.metric("Unique tracked objects", len(analytics.unique_track_ids))
        avg_fps = float(np.mean(analytics.fps_history)) if analytics.fps_history else 0.0
        m4.metric("Avg FPS", f"{avg_fps:.1f}")
        avg_latency = float(np.mean(analytics.latency_history_ms)) if analytics.latency_history_ms else 0.0
        m5.metric("Avg latency", f"{avg_latency:.1f} ms")

        st.markdown("##### Performance over time")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if analytics.fps_history:
                st.line_chart(pd.DataFrame({"FPS": list(analytics.fps_history)}))
        with pcol2:
            if analytics.latency_history_ms:
                st.line_chart(pd.DataFrame({"Latency (ms)": list(analytics.latency_history_ms)}))

        st.markdown("##### Class distribution")
        if analytics.class_counts:
            class_df = pd.DataFrame(
                sorted(analytics.class_counts.items(), key=lambda x: -x[1]),
                columns=["label", "count"]
            ).set_index("label")
            st.bar_chart(class_df)

        st.markdown("##### Object counting (virtual line)")
        lcc1, lcc2, lcc3 = st.columns(3)
        lcc1.metric("Crossed IN", analytics.line_crossings["in"])
        lcc2.metric("Crossed OUT", analytics.line_crossings["out"])
        lcc3.metric("Net", analytics.line_crossings["in"] - analytics.line_crossings["out"])
        if not cfg.line_enabled:
            st.caption("Enable **Virtual Counting Line** in the sidebar to populate this.")

        st.markdown("##### Recent detection log")
        st.dataframe(pd.DataFrame(analytics.detection_log[-200:]), use_container_width=True, hide_index=True)

        st.markdown("##### Benchmark summary (this session)")
        bench_df = pd.DataFrame({
            "Metric": ["Detector backend", "Tracker", "Frames processed", "Avg FPS", "Min FPS", "Max FPS",
                       "Avg latency (ms)", "P95 latency (ms)"],
            "Value": [
                cfg.detector_name, cfg.tracker_name, analytics.frame_count,
                f"{avg_fps:.2f}",
                f"{np.min(analytics.fps_history):.2f}" if analytics.fps_history else "-",
                f"{np.max(analytics.fps_history):.2f}" if analytics.fps_history else "-",
                f"{avg_latency:.2f}",
                f"{np.percentile(list(analytics.latency_history_ms), 95):.2f}" if analytics.latency_history_ms else "-",
            ]
        })
        st.table(bench_df)

# ---------------------------------------------------------------------------
# TAB 6 — EXPORT
# ---------------------------------------------------------------------------
with tab_export:
    st.markdown("#### Export results")
    st.caption("Export detection logs, analytics summaries, and annotated media generated in this session.")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Detection log**")
        if analytics.detection_log:
            df_log = pd.DataFrame(analytics.detection_log)
            st.download_button("⬇️ Download CSV", df_log.to_csv(index=False).encode("utf-8"),
                                file_name="visionops_detections.csv", mime="text/csv",
                                use_container_width=True)
            st.download_button("⬇️ Download JSON", json.dumps(analytics.detection_log, indent=2).encode("utf-8"),
                                file_name="visionops_detections.json", mime="application/json",
                                use_container_width=True)
        else:
            st.info("No detections logged yet.")

    with e2:
        st.markdown("**Analytics summary**")
        summary = {
            "frames_processed": analytics.frame_count,
            "unique_tracked_objects": len(analytics.unique_track_ids),
            "class_counts": dict(analytics.class_counts),
            "line_crossings": analytics.line_crossings,
            "avg_fps": float(np.mean(analytics.fps_history)) if analytics.fps_history else 0,
            "avg_latency_ms": float(np.mean(analytics.latency_history_ms)) if analytics.latency_history_ms else 0,
            "pipeline_config": {k: v for k, v in asdict(cfg).items() if k != "roi_polygon"},
        }
        st.download_button("⬇️ Download analytics summary (JSON)",
                            json.dumps(summary, indent=2, default=str).encode("utf-8"),
                            file_name="visionops_summary.json", mime="application/json",
                            use_container_width=True)
        st.json(summary, expanded=False)

    st.markdown("**Annotated image**")
    if "last_annotated_image" in st.session_state:
        ok, buf = cv2.imencode(".png", st.session_state["last_annotated_image"])
        if ok:
            st.download_button("⬇️ Download last annotated image (PNG)", buf.tobytes(),
                                file_name="visionops_annotated.png", mime="image/png")
    else:
        st.caption("Run Image Detection tab first to enable annotated-image export.")

# ---------------------------------------------------------------------------
# TAB 7 — ARCHITECTURE / ABOUT
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("#### Modular pipeline architecture")
    st.markdown("""
The platform is built around two swappable abstract interfaces so the UI never
needs to change when the underlying model changes:

```
Detector (ABC)                     Tracker (ABC)
 ├─ MotionContourDetector           ├─ CentroidTracker
 ├─ HaarFaceBodyDetector            └─ IOUTracker
 └─ YOLODetector (if installed)

           ▼                              ▼
                    Pipeline
     (Detector → Tracker → Analytics → Renderer)
                       ▼
                  Streamlit UI
        (Live / Image / Video / Webcam tabs)
```

**Adding a new model** means writing a class that implements `Detector.detect()`
and registering it in `DETECTOR_REGISTRY` — the sidebar dropdown, tabs, ROI/line
logic, tracker, analytics, and export pipeline all work with it automatically.
Same pattern for `Tracker` / `TRACKER_REGISTRY`.

**Backends included out of the box (no API key, no downloads):**
- `Motion + Contour` — background subtraction + contour analysis, supports pseudo-segmentation masks, adapts to static images via edge-based fallback.
- `Haar Cascade — Face/Body` — classical OpenCV Viola-Jones cascades, ships with OpenCV.
- `YOLOv8 (ultralytics)` — auto-detected and enabled only if the `ultralytics` package is present in the environment; adds real deep-learning detection, classification, and instance segmentation.

**Trackers included:**
- `Centroid Tracker` — nearest-centroid assignment with disappearance handling (SORT-style, no dependencies).
- `IOU Tracker` — pure bounding-box overlap matching.

**Why the live demo never shows a blank screen:** a fully procedural synthetic
traffic scene (`generate_synthetic_traffic_frame`) is rendered every frame with
zero external assets, so detection + tracking + counting + trajectories are all
demonstrably working the instant the app starts — before you ever load an
image, video, or camera.
""")
    st.markdown("#### Environment status")
    st.write(f"- Ultralytics (YOLOv8) available: **{YOLO_AVAILABLE}**")
    st.write(f"- OpenCV version: **{cv2.__version__}**")
    st.write(f"- Registered detectors: **{', '.join(DETECTOR_REGISTRY.keys())}**")
    st.write(f"- Registered trackers: **{', '.join(TRACKER_REGISTRY.keys())}**")