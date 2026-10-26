"""Streamlit UI for the Pothole & Road Defect Detector."""

import hashlib
import io
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from detector import MODEL_PATH, calculate_union_area_percent, detect_defects


st.set_page_config(
    page_title="Pothole AI | Road Defect Detector",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { max-width: 1380px; padding-top: 1.4rem; padding-bottom: 3rem; }
    .hero {
        padding: 1.65rem 1.9rem; border-radius: 18px;
        background: linear-gradient(120deg, #142b3b, #176b61);
        color: #fff; margin-bottom: 1.15rem;
    }
    .hero h1 { color: #fff; margin: 0 0 .35rem 0; font-size: 2.1rem; }
    .hero p { color: #e1f4ef; font-size: 1.02rem; margin: 0; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.24);
        padding: 1rem 1.05rem; border-radius: 14px;
        background: rgba(128,128,128,.035);
    }
    .muted { color: #858b94; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛣️ Pothole & Road Defect Detector</h1>
      <p>AI-assisted road inspection · Visual detections · Transparent measurements</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write(
    "Upload a road image, run the detector, inspect each prediction, and export "
    "an annotated image or a measurement report."
)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection settings")
    method = st.radio(
        "Detection method",
        ["Trained YOLO model", "OpenCV prototype"],
        index=0,
        help=(
            "YOLO uses best.pt. The OpenCV prototype identifies dark regions and "
            "is included only as a simple baseline."
        ),
    )
    confidence = st.slider(
        "Minimum YOLO confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower thresholds may find more candidates but can increase false positives.",
    )
    min_area = st.slider(
        "OpenCV minimum region area (pixels)",
        min_value=100,
        max_value=10000,
        value=700,
        step=100,
        help="Only applies when OpenCV prototype is selected.",
    )

    st.divider()
    st.subheader("📌 Model status")
    if MODEL_PATH.is_file():
        st.success("best.pt found")
        st.caption(f"Model path: `{MODEL_PATH}`")
    else:
        st.warning("best.pt was not found at the expected path.")
        st.caption(f"Expected path: `{MODEL_PATH}`")
        st.caption(
            "Place best.pt in the parent project folder or set "
            "POTHOLE_MODEL_PATH to its full path."
        )
    st.caption("Inference device: CPU")
    st.caption("YOLO uses built-in non-maximum suppression.")

uploaded_file = st.file_uploader(
    "📤 Upload a road photograph",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a clear image with visible road damage for the best chance of detection.",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

if uploaded_file is None:
    st.info("Upload an image to get started.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1 · Upload")
        st.write("Choose a JPG, PNG, or WEBP road photograph.")
    with c2:
        st.markdown("### 2 · Detect")
        st.write("Run the trained model and inspect its confidence scores.")
    with c3:
        st.markdown("### 3 · Export")
        st.write("Download the annotated image and CSV measurement report.")
    st.divider()
    st.caption(
        "This is a prototype for visual analysis. Predictions should be reviewed "
        "by a person before being used for maintenance decisions."
    )
    st.stop()

raw_bytes = uploaded_file.getvalue()
if len(raw_bytes) > MAX_UPLOAD_BYTES:
    st.error("This image is larger than 20 MB. Please upload a smaller image.")
    st.stop()

image_hash = hashlib.sha256(raw_bytes).hexdigest()
try:
    pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    image_rgb = np.asarray(pil_image)
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

height, width = image_rgb.shape[:2]
st.caption(f"Image: **{width} × {height} pixels** · File: `{uploaded_file.name}`")

with st.expander("Preview original image", expanded=True):
    st.image(image_rgb, use_container_width=True)

use_yolo = method == "Trained YOLO model"
current_signature = (image_hash, method, float(confidence), int(min_area))

run_col, help_col = st.columns([1, 2])
with run_col:
    run_detection = st.button(
        "🔍 Detect road defects",
        type="primary",
        use_container_width=True,
    )
with help_col:
    st.caption(
        "Changing the image or settings requires a new detection run. "
        "The app does not silently reuse results from different settings."
    )

if run_detection:
    try:
        with st.spinner("Analyzing image on the CPU…"):
            annotated_rgb, detections, note = detect_defects(
                image_rgb=image_rgb,
                use_yolo=use_yolo,
                confidence=confidence,
                min_area=min_area,
            )
        st.session_state["pothole_result"] = {
            "signature": current_signature,
            "annotated": annotated_rgb,
            "detections": detections,
            "note": note,
            "filename": uploaded_file.name,
            "width": width,
            "height": height,
            "method": method,
            "confidence_threshold": confidence,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        st.error(f"Detection could not be completed: {exc}")
        if use_yolo:
            st.info(
                "Check that best.pt exists and that the active .venv has "
                "Ultralytics installed. The expected model path is shown in the sidebar."
            )
        st.stop()

result = st.session_state.get("pothole_result")
if not result:
    st.info("Choose your settings and click **Detect road defects** to view results.")
    st.stop()

if result["signature"] != current_signature:
    st.warning("Your image or settings changed. Click **Detect road defects** to refresh the results.")
    st.stop()

detections = result["detections"]
annotated_rgb = result["annotated"]

rectangles = [
    (int(item["x"]), int(item["y"]), int(item["Width (px)"]), int(item["Height (px)"]))
    for item in detections
]
union_coverage = calculate_union_area_percent(rectangles, width, height)
confidences = [
    float(item["Confidence"])
    for item in detections
    if isinstance(item["Confidence"], (int, float))
]
mean_confidence = f"{np.mean(confidences) * 100:.1f}%" if confidences else "N/A"

st.divider()
st.subheader("📊 Detection overview")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Detections", len(detections))
with m2:
    st.metric("Mean confidence", mean_confidence)
with m3:
    st.metric("Unique box coverage", f"{union_coverage:.2f}%")

st.info(result["note"])

tab_image, tab_table, tab_method = st.tabs(
    ["🟩 Annotated image", "📋 Measurements", "ℹ️ Interpretation"]
)

with tab_image:
    st.image(
        annotated_rgb,
        caption="Predicted regions. Always inspect boxes for false positives or poor localization.",
        use_container_width=True,
    )
    png_buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(png_buffer, format="PNG")
    st.download_button(
        "⬇️ Download annotated image",
        data=png_buffer.getvalue(),
        file_name="pothole_detection_result.png",
        mime="image/png",
        use_container_width=True,
    )

with tab_table:
    if detections:
        df = pd.DataFrame(detections)
        df.insert(0, "Defect #", range(1, len(df) + 1))
        preferred_columns = [
            "Defect #", "Label", "Confidence", "x", "y",
            "Width (px)", "Height (px)", "Box area (pixels)",
            "Box area (% of image)",
        ]
        df = df[preferred_columns]
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download measurements (CSV)",
            data=csv_bytes,
            file_name="pothole_measurements.csv",
            mime="text/csv",
            use_container_width=True,
        )

        report = {
            "image_filename": result["filename"],
            "image_width_px": width,
            "image_height_px": height,
            "method": result["method"],
            "confidence_threshold": result["confidence_threshold"],
            "detection_count": len(detections),
            "mean_confidence": float(np.mean(confidences)) if confidences else None,
            "unique_bounding_box_coverage_percent": union_coverage,
            "area_metric_note": (
                "Bounding-box coverage in image pixels; not physical pothole area."
            ),
            "detections": detections,
            "generated_at": result["created_at"],
        }
        st.download_button(
            "⬇️ Download report (JSON)",
            data=json.dumps(report, indent=2).encode("utf-8"),
            file_name="pothole_detection_report.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.success("No detections passed the current threshold.")
        st.write(
            "This does not prove the image has no potholes. Try a clearer image or "
            "a lower confidence threshold, then review the result manually."
        )

with tab_method:
    st.markdown("#### How to interpret these results")
    st.write(
        "**Confidence** is the model's score for a prediction, not a guarantee "
        "that the box is correct."
    )
    st.write(
        "**Box area (% of image)** is the rectangle's area divided by total image pixels."
    )
    st.write(
        "**Unique box coverage** calculates the union of all detection rectangles, "
        "so overlapping box pixels are counted once. It is still not the true "
        "irregular pothole area and is not a physical measurement of road damage."
    )
    st.write(
        "**Localization quality** must be checked visually. A confident box can "
        "still include surrounding road, water, or background."
    )
    st.markdown("#### Recommended validation")
    st.write(
        "Test several images from different roads, lighting conditions, camera "
        "angles, and pothole sizes. Record false positives and missed potholes "
        "before making claims about accuracy."
    )

st.divider()
st.caption(
    "Pothole AI Mini Project · CPU inference · Human review recommended · "
    "Prototype results are not a substitute for a professional road inspection."
)
