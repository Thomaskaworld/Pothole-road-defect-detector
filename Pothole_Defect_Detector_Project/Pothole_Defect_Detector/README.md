## 🌐 Live Demo

**Try the application here:** [Pothole & Road Defect Detector](https://pothole-road-defect-detector-nvyqpea9qpwavrvvnd8d4b5.streamlit.app/)

Upload a road image to detect potential potholes, view annotated results and bounding-box measurements, and download CSV/JSON reports.

## ✨ Features

- AI-based pothole detection using a trained YOLO model
- Image upload and annotated detection results
- Detection confidence scores and bounding-box measurements
- Estimated image-area coverage
- Downloadable CSV and JSON reports
- Streamlit web interface

## ⚠️ Disclaimer

This is a prototype for demonstration purposes. Detection results depend on image quality and model performance and should not replace professional road inspections.

# Pothole & Road Defect Detector

A Streamlit mini-project that uses a pothole-trained YOLO model to detect candidate potholes in road photographs. It displays annotated images, confidence scores, bounding-box measurements, and downloadable CSV/JSON reports.

## Features

- Upload JPG, JPEG, PNG, and WEBP images.
- Run the trained YOLO model on CPU.
- Optional OpenCV dark-region prototype for comparison.
- Bounding-box overlays and confidence labels.
- Per-detection pixel dimensions and image-area percentages.
- Unique bounding-box coverage calculation that avoids double-counting overlaps.
- Download annotated PNG, CSV measurements, and a JSON report.
- Clear notes about model confidence and measurement limitations.

## Project layout

The current setup expects the weights one folder above the Streamlit app:

```text
Pothole_Defect_Detector_Project/
├── best.pt
└── Pothole_Defect_Detector/
    ├── app.py
    ├── detector.py
    ├── README.md
    ├── requirements.txt
    └── tests/
        └── test_detector_utils.py
```

The `best.pt` model weights are not included in this bundle. Keep your existing trained weights in the parent folder as shown above.

## Setup on Windows

Open Command Prompt in the `Pothole_Defect_Detector` folder.

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Open the local address printed by Streamlit, normally `http://localhost:8502`.

If you already have a working `.venv`, activate it and install any missing requirements rather than creating another environment.

### If the model is stored elsewhere

In Command Prompt, set the model path before starting Streamlit:

```bat
set POTHOLE_MODEL_PATH=C:\full\path\to\best.pt
streamlit run app.py --server.port 8502
```

## How to use

1. Upload a clear road photograph.
2. Choose **Trained YOLO model**.
3. Start with a confidence threshold around `0.50`.
4. Click **Detect road defects**.
5. Inspect the boxes—not just the confidence scores.
6. Download the annotated PNG and CSV/JSON report if useful.
7. Test multiple images and thresholds before drawing conclusions.

Use the OpenCV prototype only as a simple baseline. It detects dark regions and may flag shadows, stains, puddles, and road patches.

## Understanding the measurements

- **Confidence** is the model's score for a prediction. It does not guarantee correctness.
- **Box area (pixels)** is width × height of the rectangular detection.
- **Box area (% of image)** is rectangle area divided by total image pixels.
- **Unique box coverage** is the union of the detection rectangles, counting overlapping pixels once.

These are image-space bounding-box measurements. They are **not** the true irregular pothole surface area, physical dimensions, road-damage severity, or percentage of real road surface damaged. Physical measurement would require additional calibration or depth/scale information and a suitable measurement method.

## Validation checklist

- Test at least 5–10 images with different pothole sizes, lighting, viewpoints, and road textures.
- Include images with no potholes to look for false positives.
- Inspect whether boxes tightly cover the potholes.
- Record missed potholes and incorrect detections.
- Do not claim an accuracy percentage unless you evaluate against a labeled test set using an appropriate metric such as precision, recall, and mAP.

## Utility tests

From the project folder, run:

```bat
python -m unittest discover -s tests -v
```

These tests check the rectangle-union calculation only; they do not measure the trained model's accuracy.

## Limitations

- Detection quality depends on the training data, annotations, and image conditions.
- A box can be confidently predicted but still be too large or misplaced.
- CPU inference may be slower than GPU inference.
- This prototype is for demonstration and should not replace human inspection or engineering assessment.
