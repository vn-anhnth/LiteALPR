# LiteALPR

[![PyPI version](https://badge.fury.io/py/litealpr.svg)](https://pypi.org/project/litealpr/)

<p align="center">
  <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/intro1.png" width="350">
  <br>
  <em>Visual samples of challenging real-world license plates (motion blur, diverse layouts, low light) that LiteALPR is built to handle.</em>
</p>

🚀 **LiteALPR** is an accurate, extremely fast, and flexible End-to-End License Plate Recognition library.

Unlike traditional ALPR (Automatic License Plate Recognition) systems that rely on heavy architectures, LiteALPR introduces structural improvements designed specifically for high-throughput applications. Our framework achieves ultra-fast inference speeds without sacrificing accuracy on blurry or degraded license plates through two major architectural optimizations.

---

## 📑 Table of Contents

- [🧩 LiteALPR Pipeline](#-litealpr-pipeline)
  - [1. YOLOv8n-Efficient for Fast Detection](#1-yolov8n-efficient-for-fast-detection)
  - [2. SVTR26-Tiny for Lightning-Fast Recognition](#2-svtr26-tiny-for-lightning-fast-recognition)
- [🛠 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
  - [1. End-to-End Recognition (Detect & Read)](#1-end-to-end-recognition-detect--read)
  - [2. Flexible API: Detect Only](#2-flexible-api-detect-only)
  - [3. Flexible API: Recognize Only](#3-flexible-api-recognize-only)
  - [4. Using Custom Local Weights](#4-using-custom-local-weights)
  - [5. Selecting Execution Device (CPU vs GPU)](#5-selecting-execution-device-cpu-vs-gpu)
- [🏋️ Training & Evaluation](#️-training--evaluation)
  - [0. Environment Setup](#0-environment-setup)
  - [1. Model Weights Preparation](#1-model-weights-preparation)
  - [2. Data Preparation (Create LMDB)](#2-data-preparation-create-lmdb)
  - [3. Training (Det & Rec)](#3-training-det--rec)
  - [4. Evaluation (Validation)](#4-evaluation-validation)
  - [5. Batch Inference](#5-batch-inference)
  - [6. Export to ONNX](#6-export-to-onnx)
- [🤝 Acknowledgements](#-acknowledgements)
- [📜 License](#-license)
- [📧 Contact](#-contact)

---

## 🧩 LiteALPR Pipeline
The framework is structured as a highly optimized two-stage sequential pipeline:

<p align="center">
  <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/Hung_0060.png" height="150"> ➔ <b>YOLOv8n-Efficient</b> ➔ <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/Hung_0060_crop.png" height="150"> ➔ <b>SVTR26-Tiny</b> ➔ <code>59P289136</code>
</p>
<p align="center"><em>Overview of the proposed highly optimized two-stage ALPR pipeline.</em></p>

### 1. YOLOv8n-Efficient for Fast Detection
We replaced the **original heavy C2f blocks** in the YOLOv8 neck with **lightweight C3Ghost blocks**.

| Original: Heavy C2f Block | Proposed: Lightweight C3Ghost Block |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/generate_c2f.png" height="250"> | <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/generate_c3ghost.png" height="250"> |

Leveraging Ghost modules, this architectural enhancement significantly increases detection speed while maintaining high localization accuracy. By generating more feature maps from cheap operations, it eliminates computational redundancy, enabling ultra-fast performance on consumer-grade hardware without compromising precision.

### 2. SVTR26-Tiny for Lightning-Fast Recognition
To make the SVTR26 OCR model viable for strict high-speed constraints, we applied a key modification:
* **Efficient RCTC Decoder:** We entirely discarded the **Original heavy attention-based RCTC Decoder**. Since license plates have a rigid, horizontally aligned structure, we replaced 2D attention with a simple **Height-wise Average Pooling** operation. This elegantly compresses the 2D features into a 1D sequence, completely bypassing expensive matrix multiplications.

| Original: Heavy RCTC Decoder | Proposed: Efficient RCTC Decoder |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/original_rctc_decoder.png" width="400"> | <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/efficient_rctc_decoder.png" width="400"> |

By integrating these specialized components, **LiteALPR** delivers unmatched production-ready performance, processing frames at blazing speeds!

---

## 🛠 Installation

```bash
# Standard installation (CPU inference)
pip install litealpr[cpu]

# With GPU acceleration (CUDA)
pip install litealpr[gpu]
```

*(Note: To use the auto-download feature for pre-trained weights, please ensure `huggingface_hub` is installed).*

## ⚡ Quick Start

LiteALPR automatically downloads the best pre-trained models from our HuggingFace repository the first time you run it. You don't need to manually configure any paths!

### 1. End-to-End Recognition (Detect & Read)

```python
from litealpr import LiteALPR

# Initialize (auto-downloads weights if not found)
model = LiteALPR()

# Read the plate
results = model.read("sample.jpg")

for res in results:
    print(f"Plate Text: {res['text']} | Confidence: {res['score']:.4f}")
    print(f"Bounding Box: {res['box']}")
```

### 2. Flexible API: Detect Only
If you only need to locate the license plates without reading the text:
```python
# Disable the recognition model
model = LiteALPR(use_rec=False)
boxes = model.detect("sample.jpg")
print("Detected boxes:", boxes)
```

### 3. Flexible API: Recognize Only
If you already have a cropped image of a license plate and just want to read the characters:
```python
import cv2
from litealpr import LiteALPR

# Disable the detection model
model = LiteALPR(use_det=False)

# Pass either image path directly or loaded numpy array
text, score = model.recognize("sample_crop.jpg")
print(f"Text: {text} | Confidence: {score:.4f}")
```

### 4. Using Custom Local Weights
LiteALPR seamlessly supports both **ONNX Runtime** (recommended for ultra-fast deployment) and **PyTorch** checkpoints (`.pt` / `.pth`):

```python
# Option A: Load optimized ONNX models (Ultra-Fast)
model = LiteALPR(
    det_model_path="/path/to/your/yolov8n_efficient/best.onnx",
    rec_model_path="/path/to/your/svtr26_tiny/best.onnx",
)

# Option B: Load native PyTorch checkpoints (.pt / .pth)
model = LiteALPR(
    det_model_path="/path/to/your/yolov8n_efficient/best.pt",
    rec_model_path="/path/to/your/svtr26_tiny/best.pth",
)
```
> **Note:** The pipeline automatically detects the file format based on extension (`.onnx` vs `.pt`/`.pth`) and initializes the corresponding execution backend.

### 5. Selecting Execution Device (CPU vs GPU)
By default, LiteALPR automatically chooses `cuda:0` if an GPU is detected, and falls back to `cpu` otherwise. You can explicitly select the device using the `device` parameter:

```python
# Force execution on CPU
model = LiteALPR(device="cpu")

# Explicitly use GPU (CUDA)
model = LiteALPR(device="cuda:0")
```
* When using `device="cuda:0"` with ONNX models, LiteALPR utilizes `CUDAExecutionProvider`. Ensure `onnxruntime-gpu` is installed (`pip install litealpr[gpu]`).
* When using `device="cpu"`, LiteALPR seamlessly utilizes `CPUExecutionProvider` across all stages.

## 🏋️ Training & Evaluation

LiteALPR provides a complete suite of scripts in the `tools/` directory for dataset preparation, training, evaluation, batch inference, and ONNX export.

### 0. Environment Setup
To use the training and evaluation tools, clone the repository and install the development dependencies:
```bash
git clone https://github.com/vn-anhnth/LiteALPR.git
cd LiteALPR

# Install dependencies (choose CPU or GPU):
pip install -r requirements.txt        # CPU usage
# pip install -r requirements-gpu.txt  # For GPU ONNX acceleration
```

### 1. Model Weights Preparation
Before training or evaluation, download the official pre-trained models from our [HuggingFace Repository](https://huggingface.co/anhone3/LiteALPR) and place them in the following structure:
```
LiteALPR/
├── pretrained_models/
│   ├── yolov8n_efficient/
│   │   └── best.pt
│   └── svtr26_tiny/
│       └── best.pth
```
You can download them manually or use `wget`:
```bash
wget -O pretrained_models/det/yolov8n_efficient/best.pt https://huggingface.co/anhone3/LiteALPR/resolve/main/yolov8n_efficient/best.pt
wget -O pretrained_models/rec/svtr26_tiny/best.pth https://huggingface.co/anhone3/LiteALPR/resolve/main/svtr26_tiny/best.pth
```

### 2. Data Preparation (Create LMDB)
The recognition module requires datasets to be formatted into Lightning Memory-Mapped Databases (LMDB) for fast I/O access during training. Generate the LMDB using our CLI script:
```bash
python tools/create_lmdb_dataset.py \
    --data_dir ./dataset/rec \
    --label_files train_labels.txt val_labels.txt test_labels.txt \
    --output_dir ./dataset/rec/lmdb_data
```

### 3. Training (Det & Rec)
Before training, you must configure your dataset paths, batch sizes, and learning parameters:

**For Detection:**
Open `tools/train_det.py` and modify the parameters inside the `model.train()` function directly:
```python
model.train(
    data='dataset/det/data.yaml', # Point this to your YOLO data.yaml
    epochs=50,
    batch=256,
    ...
)
```

**For Recognition (`configs/rec/svtr26/svtr26_tiny.yml`):**
```yaml
Train:
  dataset:
    name: RatioDataSetTVResize
    data_dir_list: ['./dataset/rec/lmdb_data/train']

Eval:
  dataset:
    name: RatioDataSetTVResize
    data_dir_list: ['./dataset/rec/lmdb_data/val']
```

Once configured, start training:

> [!TIP]
> **Pre-trained Models (Fine-tuning)**
> By default, the training process will load pre-trained weights to speed up convergence. You can change the path or remove it to train from scratch:
> - **For Detection:** Edit the `.load(...)` path directly inside the `tools/train_det.py` script.
> - **For Recognition:** Edit the `Global.pretrained_model` field inside your `.yml` config file (e.g., `configs/rec/svtr26/svtr26_tiny.yml`).

```bash
# Train Detection Model (YOLOv8)
# For multi-GPU training, set device to a list of GPU IDs in train_det.py, e.g., device=[0, 1]
python tools/train_det.py -c configs/det/yolov8/yolov8n_efficient.yml

# Train Recognition Model (SVTR26)
# For multi-GPU training, set nproc_per_node to the number of GPUs being used for training
torchrun --nproc_per_node=1 tools/train_rec.py \
    -c configs/rec/svtr26/svtr26_tiny.yml
```

### 4. Evaluation (Validation)
Evaluate your trained checkpoints on the validation set:
```bash
# Evaluate Detection
python tools/eval_det.py -m output/det/yolov8n_efficient/train/weights/best.pt

# Evaluate Recognition
python tools/eval_rec.py -c configs/rec/svtr26/svtr26_tiny.yml -m output/rec/svtr26_tiny/train/best.pth
```

### 5. Batch Inference
Test your checkpoints directly on directories of images (supports `--save_log` to save predictions):
```bash
# Infer Detection
python tools/infer_det.py -m pretrained_models/det/yolov8n_efficient/best.pt -d dataset/det/test/images --save_log

# Infer Recognition
python tools/infer_rec.py -m pretrained_models/rec/svtr26_tiny/best.pth -d dataset/rec/test --save_log
```

### 6. Export to ONNX
Export your trained PyTorch models to the ONNX format for deployment in production environments (C++, C#, TensorRT, etc.). You can configure the ONNX operator set version via `--opset` (default: 12).

```bash
# Export Detection (default: imgsz=416, opset=12)
# The ONNX file will automatically be saved alongside the original `.pt` file (e.g., best_416.onnx)
python tools/export_det.py -m output/det/yolov8n_efficient/train/weights/best.pt --imgsz 416 --opset 18

# Export Recognition (default: 128x32, opset=12)
# If you need it to accept dynamic width images in production, add the `--dynamic` flag
python tools/export_rec.py -m output/rec/svtr26_tiny/train/best.pth --save_path output/rec/svtr26_tiny/train/best.onnx --opset 18 --dynamic
```

## 🤝 Acknowledgements

- **OpenOCR**: LiteALPR is built upon the robust foundation of [OpenOCR](https://github.com/Topdu/OpenOCR).
- **YOLOv8 & SVTRv2**: This work heavily leverages the architectural innovations from **YOLOv8** for high-speed object detection and **SVTRv2** for accurate text recognition.
  - Read the [YOLOv8 Paper](https://arxiv.org/html/2408.15857v1)
  - Read the [SVTRv2 Paper](https://arxiv.org/html/2411.15858v1)
- **Datasets**: Our evaluation utilizes datasets from [Brazil (RodoSol-ALPR)](https://github.com/raysonlaroca/rodosol-alpr-dataset), [China (CBLPRD-330k)](https://github.com/SunlifeV/CBLPRD-330k), and [Vietnam](https://www.kaggle.com/datasets/duydieunguyen/licenseplates) public collections alongside self-collected traffic footage. We sincerely thank the original authors of these datasets for advancing the ALPR research community.

## 📜 License
This project is open-sourced under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

## 📧 Contact
For any questions or issues, please open an issue or contact: `anhnth.25ai@ou.edu.vn`.
