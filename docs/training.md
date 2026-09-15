# Training, Evaluation & Export Guide

LiteALPR includes a modular suite of scripts under `tools/` for preparing datasets, training custom detectors or recognizers, running evaluations, and exporting to optimized ONNX models.

---

## 0. Environment & Setup

Clone the repository and install all dependencies:

```bash
git clone https://github.com/vn-anhnth/LiteALPR.git
cd LiteALPR

# Choose based on your runtime environment:
pip install -r requirements.lock       # Exact reference environment for reproducibility
pip install -r requirements.txt        # CPU (ONNX) Inference
pip install -r requirements-gpu.txt    # NVIDIA GPU (ONNX) Acceleration
```

---

## 1. Model Weights Preparation

Before training (fine-tuning) or evaluation, download the official pre-trained weights from our [Hugging Face repository](https://huggingface.co/anhone3/LiteALPR/tree/main) and place them in `pretrained_models/`:

```
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

---

## 2. Dataset Preparation

### Detection Dataset
Detection models use the standard YOLO dataset format (`data.yaml` pointing to image and label folders).

### Recognition Dataset (LMDB)
The recognition module requires datasets to be formatted into Lightning Memory-Mapped Databases (LMDB) for fast I/O access during training.

Generate the LMDB using our CLI script:

```bash
python tools/create_lmdb_dataset.py \
    --data_dir ./dataset/rec \
    --label_files train_labels.txt val_labels.txt test_labels.txt \
    --output_dir ./dataset/rec/lmdb_data
```

---

## 3. Training Models

Before training, configure your dataset paths, batch sizes, and learning parameters:

### Train YOLOv8n-Efficient Detector
Configure your dataset paths, batch sizes, and training hyperparameters inside the `Global:` section of `configs/det/yolov8/yolov8n_efficient.yml`:

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

Start training with the CLI:

```bash
# Single GPU training (set device: 0 in YAML)
python tools/train_det.py -c configs/det/yolov8/yolov8n_efficient.yml

# Multi-GPU training (set device to GPU IDs, e.g. [0, 1] or [0, 1, 2, 3] in configs/det/yolov8/yolov8n_efficient.yml)
python tools/train_det.py -c configs/det/yolov8/yolov8n_efficient.yml
```

### Train SVTR26-Tiny Recognizer
Configure your training hyperparameters in `Global:` and `Train:` sections of `configs/rec/svtr26/svtr26_tiny.yml`:

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

Start training with `torchrun`:

```bash
# Single GPU training
torchrun --nproc_per_node=1 tools/train_rec.py -c configs/rec/svtr26/svtr26_tiny.yml

# Multi-GPU training (e.g. set --nproc_per_node to number of GPUs, such as 2, 4, 8)
torchrun --nproc_per_node=2 tools/train_rec.py -c configs/rec/svtr26/svtr26_tiny.yml
```

!!! tip "Pre-trained Models (Fine-tuning)"
    By default, the training process will load pre-trained weights to speed up convergence. You can change the path or remove it to train from scratch:

    * **For Detection:** Edit the `Global.pretrained_model` field inside `configs/det/yolov8/yolov8n_efficient.yml`.
    * **For Recognition:** Edit the `Global.pretrained_model` field inside `configs/rec/svtr26/svtr26_tiny.yml`.


---

## 4. Evaluation (Validation)

Evaluate your trained checkpoints on the validation set:

### Evaluate Detector
```bash
python tools/eval_det.py -m output/det/yolov8n_efficient/train/weights/best.pt
```

### Evaluate Recognizer
```bash
python tools/eval_rec.py \
    -c configs/rec/svtr26/svtr26_tiny.yml \
    -m output/rec/svtr26_tiny/train/best.pth
```

---

## 5. Batch Inference

Test your checkpoints directly on directories of images. Pass `--save_log` to persist prediction logs:

### Infer Detection
```bash
python tools/infer_det.py \
    -m output/det/yolov8n_efficient/train/weights/best.pt \
    -d dataset/det/test/images \
    --save_log
```

### Infer Recognition
```bash
python tools/infer_rec.py \
    -m output/rec/svtr26_tiny/train/best.pth \
    -d dataset/rec/test \
    --save_log
```

---

## 6. Exporting to ONNX

Export your trained PyTorch models to the ONNX format for deployment in production environments (C++, C#, TensorRT, etc.). You can configure the ONNX operator set version via `--opset` (default: 12).

### Export Detector
The ONNX file will automatically be saved alongside the original `.pt` file (e.g. `best_416.onnx`):

```bash
python tools/export_det.py \
    -m output/det/yolov8n_efficient/train/weights/best.pt \
    --imgsz 416 \
    --opset 18
```

### Export Recognizer
If you need it to accept dynamic width images in production, include the `--dynamic` flag:

```bash
python tools/export_rec.py \
    -m output/rec/svtr26_tiny/train/best.pth \
    --save_path output/rec/svtr26_tiny/train/best.onnx \
    --opset 18 \
    --dynamic
```
