# LiteALPR

[![PyPI version](https://badge.fury.io/py/litealpr.svg)](https://pypi.org/project/litealpr/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://vn-anhnth.github.io/LiteALPR/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22722292.svg)](https://doi.org/10.5281/zenodo.22722292)

<p align="center">
  <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/intro1.png" width="350">
  <br>
  <em>Visual samples of challenging real-world license plates (motion blur, diverse layouts, low light) that LiteALPR is built to handle.</em>
</p>

🚀 **LiteALPR** is a lightweight, accurate, and flexible End-to-End License Plate Recognition library.

Unlike traditional ALPR (Automatic License Plate Recognition) systems that rely on heavy architectures, LiteALPR introduces structural improvements designed specifically for high-throughput applications. Our framework achieves fast inference speeds without sacrificing accuracy on blurry or degraded license plates through two major architectural optimizations.

---

## 📑 Table of Contents

- [🧩 LiteALPR Pipeline](#-litealpr-pipeline)
  - [1. YOLOv8n-Efficient for Fast Detection](#1-yolov8n-efficient-for-fast-detection)
  - [2. SVTR26-Tiny for Fast Recognition](#2-svtr26-tiny-for-fast-recognition)
- [🛠 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
  - [1. End-to-End Recognition (CPU & GPU)](#1-end-to-end-recognition-cpu--gpu)
  - [2. Flexible API: Detect Only](#2-flexible-api-detect-only)
  - [3. Flexible API: Recognize Only](#3-flexible-api-recognize-only)
  - [4. Using Custom Local Weights](#4-using-custom-local-weights)
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

Leveraging Ghost modules, this architectural enhancement significantly increases detection speed while maintaining high localization accuracy. By generating more feature maps from cheap operations, it eliminates computational redundancy, enabling highly efficient performance on consumer-grade hardware without compromising precision.

### 2. SVTR26-Tiny for Fast Recognition
To make the SVTR26 OCR model viable for strict high-speed constraints, we applied a key modification:
* **Efficient RCTC Decoder:** We entirely discarded the **Original heavy attention-based RCTC Decoder**. Since license plates have a rigid, horizontally aligned structure, we replaced 2D attention with a simple **Height-wise Average Pooling** operation. This elegantly compresses the 2D features into a 1D sequence, completely bypassing expensive matrix multiplications.

| Original: Heavy RCTC Decoder | Proposed: Efficient RCTC Decoder |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/original_rctc_decoder.png" width="400"> | <img src="https://raw.githubusercontent.com/vn-anhnth/LiteALPR/main/docs/figures/efficient_rctc_decoder.png" width="400"> |

By integrating these specialized components, **LiteALPR** delivers robust production-ready performance, processing frames efficiently in high-throughput pipelines.

---

## 🛠 Installation

```bash
# Standard installation (auto-detects & configures runtime at first run)
pip install litealpr

# Or explicitly specify your target runtime environment:
pip install litealpr[cpu]  # CPU (ONNX) Inference
pip install litealpr[gpu]  # NVIDIA GPU (ONNX) Acceleration
```

*(Note: When using standard `pip install litealpr`, LiteALPR automatically detects your device hardware and configures the corresponding ONNX Runtime execution engine and Hugging Face pre-trained weights upon initial execution).*

## ⚡ Quick Start

LiteALPR automatically downloads the best pre-trained models from our Hugging Face repository the first time you run it. You don't need to manually configure any paths!

### 1. End-to-End Recognition (CPU & GPU)

```python
from litealpr import LiteALPR

# Option A: Automatic device selection (GPU if available, else CPU)
model = LiteALPR()

# Option B: Explicitly select target execution device
# model = LiteALPR(device="cpu")     # CPU (ONNX) Inference
# model = LiteALPR(device="cuda:0")  # NVIDIA GPU (ONNX) Acceleration

# Read the license plate (auto-downloads pre-trained weights if not found)
results = model.read("sample.jpg")

for res in results:
    print(f"Plate Text: {res['text']} | Confidence: {res['score']:.4f}")
    print(f"Bounding Box: {res['box']}")
```
> **Device Selection:** By default, LiteALPR uses `device="cuda:0"` if a GPU is available, and seamlessly falls back to `device="cpu"` otherwise. When using GPU acceleration with ONNX models, ensure `litealpr[gpu]` is installed (`pip install litealpr[gpu]`).

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
# Option A: Load optimized ONNX models (High-Throughput)
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

## 🏋️ Training & Evaluation

LiteALPR provides a complete suite of scripts in the `tools/` directory for dataset preparation, training, evaluation, batch inference, and ONNX export.

### 0. Environment Setup
To use the training and evaluation tools, clone the repository and install the development dependencies:
```bash
git clone https://github.com/vn-anhnth/LiteALPR.git
cd LiteALPR

# Choose based on your runtime environment:
pip install -r requirements.lock       # Exact reference environment for reproducibility
pip install -r requirements.txt        # CPU (ONNX) Inference
pip install -r requirements-gpu.txt    # NVIDIA GPU (ONNX) Acceleration
```

### 1. Model Weights Preparation
Before training or evaluation, download the official pre-trained models from our [Hugging Face repository](https://huggingface.co/anhone3/LiteALPR/tree/main) and place them in the following structure:
```text
LiteALPR/
└── pretrained_models/
    ├── det/
    │   └── yolov8n_efficient/
    │       └── best.pt
    └── rec/
        └── svtr26_tiny/
            └── best.pth
```
You can download them using `wget` or `curl`:
```bash
# Download Detection pre-trained weights
wget -O pretrained_models/det/yolov8n_efficient/best.pt https://huggingface.co/anhone3/LiteALPR/resolve/main/yolov8n_efficient/best.pt

# Download Recognition pre-trained weights
wget -O pretrained_models/rec/svtr26_tiny/best.pth https://huggingface.co/anhone3/LiteALPR/resolve/main/svtr26_tiny/best.pth
```

### 2. Data Preparation (Create LMDB)
The recognition module requires datasets to be formatted into Lightning Memory-Mapped Databases (LMDB) for fast I/O access during training.

Generate the LMDB using our CLI script:
```bash
python tools/create_lmdb_dataset.py \
    --data_dir ./dataset/rec \
    --label_files train_labels.txt val_labels.txt test_labels.txt \
    --output_dir ./dataset/rec/lmdb_data
```

### 3. Training (Det & Rec)
Before training, you must configure your dataset paths, batch sizes, and learning parameters:

**For Detection (`configs/det/yolov8/yolov8n_efficient.yml`):**
Configure your dataset paths, batch sizes, and training hyperparameters under the `Global:` section:
```yaml
Global:
  pretrained_model: "pretrained_models/det/yolov8n_efficient/best.pt"  # or null to train from scratch
  data: "dataset/det/data.yaml"
  epochs: 50
  imgsz: 640
  batch: 256
  device: 0  # GPU ID (e.g. 0), or list for multi-GPU (e.g. [0, 1] or more)
  project: "output/det/yolov8n_efficient"
  workers: 8
```

**For Recognition (`configs/rec/svtr26/svtr26_tiny.yml`):**
Configure your training hyperparameters under `Global:` and `Train:` sections:
```yaml
Global:
  device: gpu
  epoch_num: 150
  pretrained_model: "./pretrained_models/rec/svtr26_tiny/best.pth"  # or null to train from scratch
  output_dir: "./output/rec/svtr26_tiny/train"

Train:
  dataset:
    name: RatioDataSetTVResize
    data_dir_list: ['./dataset/rec/lmdb_data/train']
  sampler:
    first_bs: &bs 256             # Batch size per GPU
  loader:
    batch_size_per_card: *bs
    num_workers: 4

Eval:
  dataset:
    name: RatioDataSetTVResize
    data_dir_list: ['./dataset/rec/lmdb_data/val']
```

Once configured, start training:

> [!TIP]
> **Pre-trained Models (Fine-tuning)**
> By default, the training process will load pre-trained weights to speed up convergence. You can change the path or remove it to train from scratch:
> - **For Detection:** Edit the `Global.pretrained_model` field inside `configs/det/yolov8/yolov8n_efficient.yml`.
> - **For Recognition:** Edit the `Global.pretrained_model` field inside `configs/rec/svtr26/svtr26_tiny.yml`.

```bash
# Train Detection Model (YOLOv8)
# For multi-GPU training, set device to GPU IDs (e.g. [0, 1] or more) in configs/det/yolov8/yolov8n_efficient.yml
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
python tools/infer_det.py -m output/det/yolov8n_efficient/train/weights/best.pt -d dataset/det/test/images --save_log

# Infer Recognition
python tools/infer_rec.py -m output/rec/svtr26_tiny/train/best.pth -d dataset/rec/test --save_log
```

### 6. Export to ONNX
Export your trained PyTorch models to the ONNX format for deployment in production environments (C++, C#, TensorRT, etc.). You can configure the ONNX operator set version via `--opset` (default: 12).

```bash
# Export Detection (default: imgsz=416, opset=12)
# The ONNX file will automatically be saved alongside the original `.pt` file (e.g., best_416.onnx)
python tools/export_det.py -m output/det/yolov8n_efficient/train/weights/best.pt --imgsz 416 --opset 18
# -> Expected output: output/det/yolov8n_efficient/train/weights/best_416.onnx

# Export Recognition (default: 128x32, opset=12)
# If you need it to accept dynamic width images in production, add the `--dynamic` flag
python tools/export_rec.py -m output/rec/svtr26_tiny/train/best.pth --save_path output/rec/svtr26_tiny/train/best.onnx --opset 18 --dynamic
# -> Expected output: output/rec/svtr26_tiny/train/best.onnx
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
