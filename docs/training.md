# Training, Evaluation & Export Guide

LiteALPR includes a modular suite of scripts under `tools/` for preparing datasets, training custom detectors or recognizers, running evaluations, and exporting to optimized ONNX models.

---

## 1. Environment & Setup

Clone the repository and install all dependencies:

```bash
git clone https://github.com/vn-anhnth/LiteALPR.git
cd LiteALPR

# Choose based on your runtime environment:
pip install -r requirements.txt        # CPU environment
pip install -r requirements-gpu.txt    # GPU ONNX acceleration
```

---

## 2. Dataset Preparation

### Detection Dataset
Detection models use the standard YOLO dataset format (`data.yaml` pointing to image and label folders).

### Recognition Dataset (LMDB)
The recognition stage uses Lightning Memory-Mapped Databases (LMDB) for maximum I/O throughput during training.

Convert text label files into LMDB format:

```bash
python tools/create_lmdb_dataset.py \
    --data_dir ./dataset/rec \
    --label_files train_labels.txt val_labels.txt test_labels.txt \
    --output_dir ./dataset/rec/lmdb_data
```

---

## 3. Training Models

### Train YOLOv8n-Efficient Detector
Edit training configurations inside `configs/det/yolov8/yolov8n_efficient.yml` or customize hyperparameters in `tools/train_det.py`:

```bash
python tools/train_det.py -c configs/det/yolov8/yolov8n_efficient.yml
```

### Train SVTR26-Tiny Recognizer
Update your dataset paths in `configs/rec/svtr26/svtr26_tiny.yml`:

```bash
# Single GPU training
torchrun --nproc_per_node=1 tools/train_rec.py -c configs/rec/svtr26/svtr26_tiny.yml

# Multi-GPU training (e.g. 2 GPUs)
torchrun --nproc_per_node=2 tools/train_rec.py -c configs/rec/svtr26/svtr26_tiny.yml
```

---

## 4. Evaluation

Evaluate model checkpoints on test/validation sets:

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

## 5. Exporting to ONNX

Once models are trained, export them to ONNX for production deployment:

### Export Detector
```bash
python tools/export_det.py \
    -m output/det/yolov8n_efficient/train/weights/best.pt \
    --imgsz 416 \
    --opset 18
```

### Export Recognizer
```bash
python tools/export_rec.py \
    -m output/rec/svtr26_tiny/train/best.pth \
    --save_path output/rec/svtr26_tiny/train/best.onnx \
    --opset 18 \
    --dynamic
```
