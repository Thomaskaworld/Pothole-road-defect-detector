"""Detection utilities for the Pothole & Road Defect Detector project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def _find_model_path() -> Path:
    """Find best.pt, allowing an explicit environment-variable override."""
    override = os.environ.get("POTHOLE_MODEL_PATH")
    if override:
        return Path(override).expanduser().resolve()

    project_dir = Path(__file__).resolve().parent
    candidates = (
        project_dir.parent / "best.pt",       # Current project layout
        project_dir / "best.pt",              # Model beside detector.py
        project_dir.parent.parent / "best.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Keep a useful expected path for error messages if no candidate exists.
    return candidates[0]


MODEL_PATH = _find_model_path()
_MODEL = None


def _load_model():
    """Load YOLO once per process and reuse the model on subsequent calls."""
    global _MODEL

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Could not find the trained YOLO weights (best.pt).\n"
            f"Expected model path: {MODEL_PATH}\n"
            "Place best.pt in the parent project folder, beside detector.py, "
            "or set the POTHOLE_MODEL_PATH environment variable."
        )

    if _MODEL is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is missing from this Python environment. "
                "Activate your .venv and run: python -m pip install ultralytics"
            ) from exc
        _MODEL = YOLO(str(MODEL_PATH))

    return _MODEL


def _validate_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    """Validate and normalize an RGB image array for inference."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be a NumPy array.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
        raise ValueError("The supplied image is empty.")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_rgb)


def _yolo_candidates(image_rgb: np.ndarray, confidence: float = 0.25):
    """Return YOLO detections after Ultralytics' built-in NMS."""
    model = _load_model()
    result = model.predict(
        source=image_rgb,
        conf=float(confidence),
        iou=0.50,
        device="cpu",
        max_det=100,
        verbose=False,
    )[0]

    candidates = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [
                int(round(value)) for value in box.xyxy[0].tolist()
            ]
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            label = result.names.get(class_id, f"class_{class_id}")
            candidates.append(
                (x1, y1, max(1, x2 - x1), max(1, y2 - y1), (label, score))
            )
    return candidates


def _opencv_candidates(image_rgb: np.ndarray, min_area: int = 700):
    """Basic dark-region prototype; this is not a trained pothole classifier."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=17, sigmaY=17)
    darkness = cv2.subtract(background, gray)
    _, mask = cv2.threshold(darkness, 12, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = gray.shape
    candidates = []

    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)

        if contour_area < int(min_area) or box_width < 12 or box_height < 12:
            continue
        if box_width * box_height > 0.65 * width * height:
            continue

        candidates.append((x, y, box_width, box_height, None))

    return candidates


def calculate_union_area_percent(
    boxes: Iterable[Sequence[int]] | Iterable[dict],
    image_width: int,
    image_height: int,
) -> float:
    """
    Calculate the union coverage of rectangular boxes as a percentage of image
    pixels. Overlapping rectangles are counted only once.

    Each box may be (x, y, width, height) or a detection dictionary containing
    keys x, y, width, height. This is still bounding-box coverage, not the true
    irregular pothole surface area or physical road area.
    """
    image_width = int(image_width)
    image_height = int(image_height)
    if image_width <= 0 or image_height <= 0:
        return 0.0

    events = []
    for item in boxes:
        if isinstance(item, dict):
            x = int(item["x"])
            y = int(item["y"])
            width = int(item["width"])
            height = int(item["height"])
        else:
            x, y, width, height = (int(v) for v in item)

        x1 = max(0, min(x, image_width))
        y1 = max(0, min(y, image_height))
        x2 = max(0, min(x + width, image_width))
        y2 = max(0, min(y + height, image_height))
        if x2 <= x1 or y2 <= y1:
            continue
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))

    if not events:
        return 0.0

    events.sort(key=lambda event: event[0])
    active_intervals = []
    union_area = 0
    previous_x = events[0][0]
    index = 0

    def covered_y_length(intervals):
        if not intervals:
            return 0
        ordered = sorted(intervals)
        start, end = ordered[0]
        total = 0
        for next_start, next_end in ordered[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        return total + end - start

    while index < len(events):
        current_x = events[index][0]
        union_area += (current_x - previous_x) * covered_y_length(active_intervals)

        # Apply all events at the same x-coordinate before moving to the next slab.
        while index < len(events) and events[index][0] == current_x:
            _, action, y1, y2 = events[index]
            interval = (y1, y2)
            if action == 1:
                active_intervals.append(interval)
            else:
                try:
                    active_intervals.remove(interval)
                except ValueError:
                    pass
            index += 1

        previous_x = current_x

    return round((union_area / (image_width * image_height)) * 100.0, 4)


def detect_defects(
    image_rgb: np.ndarray,
    use_yolo: bool = True,
    confidence: float = 0.25,
    min_area: int = 700,
):
    """
    Detect candidate defects and return (annotated_rgb, detections, note).

    Area metrics are based on rectangular boxes in image pixels. They do not
    estimate physical pothole dimensions or the true irregular damaged area.
    """
    image_rgb = _validate_rgb_image(image_rgb)
    image_height, image_width = image_rgb.shape[:2]
    total_pixels = image_width * image_height

    if use_yolo:
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        candidates = _yolo_candidates(image_rgb, confidence=confidence)
        note = (
            "Trained YOLO model used. Ultralytics applies built-in non-maximum "
            "suppression to reduce duplicate predictions. Remaining boxes can "
            "still be misplaced; confidence is not a guarantee of accuracy. "
            "Coverage values describe rectangles in the image, not physical "
            "pothole surface area."
        )
    else:
        candidates = _opencv_candidates(image_rgb, min_area=min_area)
        note = (
            "OpenCV dark-region prototype used. It is not a trained classifier "
            "and may flag shadows, stains, puddles, or road patches. Coverage "
            "values describe rectangles in the image, not physical pothole area."
        )

    annotated_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    detections = []

    for x, y, box_width, box_height, extra in candidates:
        x = max(0, min(int(x), image_width - 1))
        y = max(0, min(int(y), image_height - 1))
        box_width = max(1, min(int(box_width), image_width - x))
        box_height = max(1, min(int(box_height), image_height - y))

        rectangle_area = box_width * box_height
        rectangle_pct = (rectangle_area / total_pixels) * 100.0

        if extra is not None:
            label, score = extra
            score_value = round(float(score), 3)
            display_label = str(label)
            title = f"{display_label} {score_value:.2f}"
        else:
            display_label = "Potential dark region"
            score_value = None
            title = "Potential defect"

        detections.append(
            {
                "Label": display_label,
                "Confidence": score_value if score_value is not None else "N/A",
                "x": x,
                "y": y,
                "Width (px)": box_width,
                "Height (px)": box_height,
                "Bounding box (x, y, width, height)": (
                    f"{x}, {y}, {box_width}, {box_height}"
                ),
                "Box area (pixels)": int(rectangle_area),
                "Box area (% of image)": round(rectangle_pct, 4),
            }
        )

        # Draw a readable label with a filled background.
        color = (50, 220, 50)  # BGR
        x2 = min(image_width - 1, x + box_width)
        y2 = min(image_height - 1, y + box_height)
        cv2.rectangle(annotated_bgr, (x, y), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            title, font, font_scale, thickness
        )
        label_top = max(0, y - text_height - baseline - 8)
        label_right = min(image_width - 1, x + text_width + 10)
        label_bottom = min(image_height - 1, label_top + text_height + baseline + 8)

        cv2.rectangle(
            annotated_bgr,
            (x, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )
        text_y = min(image_height - 1, label_top + text_height + 3)
        cv2.putText(
            annotated_bgr,
            title,
            (x + 5, text_y),
            font,
            font_scale,
            (15, 25, 15),
            thickness,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, detections, note
