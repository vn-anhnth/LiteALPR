# Python API & Usage Guide

LiteALPR provides an intuitive Python interface with flexible options for full end-to-end recognition, detection-only, or recognition-only tasks.

---

## 1. End-to-End Pipeline

The primary entry point is the `LiteALPR` class.

```python
from litealpr import LiteALPR

# 1. Automatic device selection (GPU if available, fallback to CPU)
alpr = LiteALPR()

# Or explicitly choose target device:
# alpr = LiteALPR(device="cpu")
# alpr = LiteALPR(device="cuda:0")

# 2. Process an image file or numpy array
results = alpr.read("test_car.jpg")

# 3. Inspect results
for item in results:
    text = item["text"]  # Predicted license plate string
    score = item["score"]  # Confidence score (float 0.0 - 1.0)
    box = item["box"]  # Bounding box coordinates [x1, y1, x2, y2]
    print(f"Plate: {text} (conf: {score:.3f}) at {box}")
```

### Return Format

`alpr.read()` returns a list of dictionaries, one for each detected license plate:

```python
[{"text": "59P289136", "score": 0.9842, "box": [450, 320, 680, 410]}]
```

---

## 2. Detection Only

If your workflow only requires vehicle/plate bounding box localization:

```python
from litealpr import LiteALPR

# Disable text recognition
alpr = LiteALPR(use_rec=False)

# Detect plates
boxes = alpr.detect("test_car.jpg")

for box in boxes:
    # box format: [x1, y1, x2, y2, confidence, class_id]
    print("Detected box:", box)
```

---

## 3. Recognition Only

If you already have cropped license plate images (e.g., from an external detector or crop camera stream):

```python
import cv2
from litealpr import LiteALPR

# Disable detection
alpr = LiteALPR(use_det=False)

# Pass either image path or loaded BGR numpy array
crop = cv2.imread("cropped_plate.jpg")
text, score = alpr.recognize(crop)

print(f"Plate Text: {text} | Confidence: {score:.4f}")
```

---

## 4. Custom Local Models & Backends

> **Note:** To download the pre-trained `.onnx` weights manually for offline usage, you can fetch them directly from our [HuggingFace Repository](https://huggingface.co/anhone3/LiteALPR/tree/main).

LiteALPR automatically supports both **ONNX Runtime** and **PyTorch** checkpoints based on the file extension:

### Using ONNX Models (Recommended)
```python
alpr = LiteALPR(
    det_model_path="path/to/custom_yolo.onnx",
    rec_model_path="path/to/custom_svtr.onnx",
    device="cuda:0",
)
```

### Using PyTorch Checkpoints
```python
alpr = LiteALPR(
    det_model_path="path/to/weights/best.pt",
    rec_model_path="path/to/weights/best.pth",
    device="cuda:0",
)
```

---

## 5. API Reference Summary

### `LiteALPR(...)` Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `det_model_path` | `str | Path | None` | `None` | Path to detection weights (`.onnx` or `.pt`). Auto-downloaded from HuggingFace if `None`. |
| `rec_model_path` | `str | Path | None` | `None` | Path to recognition weights (`.onnx` or `.pth`). Auto-downloaded from HuggingFace if `None`. |
| `use_det` | `bool` | `True` | Whether to enable the detection stage. Set to `False` for recognition-only mode. |
| `use_rec` | `bool` | `True` | Whether to enable the text recognition stage. Set to `False` for detection-only mode. |
| `device` | `str | torch.device | None` | `None` | Target execution device (`"cuda:0"`, `"cpu"`). Defaults to `"cuda:0"` if CUDA is available, otherwise `"cpu"`. |

### Methods Summary

| Method | Arguments | Returns | Description |
| :--- | :--- | :--- | :--- |
| `read(image, conf_thresh=0.25)` | `image_path` (str) or `img` (numpy array) | `List[Dict]` | Runs end-to-end detection and recognition. Returns list of `{"box": [x1, y1, x2, y2], "text": str, "score": float}`. |
| `detect(image, conf_thresh=0.25)` | `image_path` (str) or `img` (numpy array) | `List[List[int]]` | Runs detection stage only. Returns bounding boxes `[[x1, y1, x2, y2], ...]`. |
| `recognize(crop_img)` | `image_path` (str) or `crop_img` (numpy array) | `Tuple[str, float]` | Runs recognition stage on cropped plate. Returns `(text, confidence)`. |
