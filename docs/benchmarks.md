# Benchmarks & Experimental Results

LiteALPR was trained and evaluated on two task-specific corpora spanning three regional groups:

- **[Brazil (RodoSol-ALPR)](https://github.com/raysonlaroca/rodosol-alpr-dataset)**: 20,000 scenes (detection) and 20,000 crops (recognition)
- **China**: 20,644 [CCPD](https://github.com/detectRecog/CCPD) scenes (detection) and 20,000 [CBLPRD-330k](https://github.com/SunlifeV/CBLPRD-330k) crops (recognition)
- **[Vietnam](https://www.kaggle.com/datasets/duydieunguyen/licenseplates) & Auxiliary**: 11,651 scenes (detection) and 12,595 crops (recognition: 4,556 public Vietnamese, 5,547 synthetic Vietnamese, 1,036 self-collected Vietnamese, and 1,456 [auxiliary European](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset) crops)

### Dataset Splits Summary

| Task / Partition | Brazil | China | Vietnam & Aux. | Total |
| :--- | ---: | ---: | ---: | ---: |
| **Full-Image Plate Detection (Scenes)** | **20,000** | **20,644** | **11,651** | **52,295** |
| Training | 15,990 | 16,502 | 9,344 | 41,836 |
| Validation | 1,999 | 2,038 | 1,192 | 5,229 |
| Testing | 2,011 | 2,104 | 1,115 | 5,230 |
| **Cropped Character Recognition (Crops)** | **20,000** | **20,000** | **12,595** | **52,595** |
| Training | 9,454 | 9,570 | 5,976 | 25,000 |
| Validation | 986 | 980 | 629 | 2,595 |
| Testing | 9,560 | 9,450 | 5,990 | 25,000 |

- **Plates per scene (Detection test, 5,230 scenes)**: 5,156 single-plate (98.59%) and 74 multi-plate (1.41%: 52 with 2, 16 with 3, 4 with 4, and 2 with 5 plates).
- **End-to-End Holdout Set (5,230 full-scene images)**: Distinct from the detection test split; combines 3,981 detection-test scenes having verified character labels (2,104 `ccpd_test`, 1,742 `rodosol_test`, 135 `vn_test`) with 1,249 additional `ccpd_extra` scenes strictly disjoint from detector training/validation.
- **Reproducible Split Manifests**: Provided in `manifests/` (`detection_split_manifest.csv`, `ocr_split_manifest.csv` with `unseen_plate` flag, and `e2e_holdout_manifest.csv`).

!!! info "Benchmarking Protocol"
    Workstation benchmarks run in ONNX FP32 precision ($N=1$ batch size, $416 \times 416$ detector deployment resolution, 50 warm-up runs discarded, explicit `cudaDeviceSynchronize()` barriers on GPU, and 6 intra-op CPU threads on the AMD Ryzen 5 4600G). Latency ($\text{mean} \pm \text{std}$) is measured per-image over 1,000 samples, while FPS standard deviation is computed across 5 consecutive 1,000-image passes.

## 1. End-to-End Pipeline Performance

Full-pipeline latency, throughput, exact-match accuracy (`E2E Acc.`), and Character Error Rate (`E2E CER`) on the 5,230-scene holdout set (missed detections with $\text{IoU} < 0.50$ count as full-length CER deletions):

| Platform | Model / Pipeline | Params | Pre (ms) | Infer (ms) | Post (ms) | Total (ms) | FPS | E2E Acc. (%) | E2E CER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPU** *(RTX 3060 12GB)* | [fast-alpr (out-of-the-box)](https://github.com/ankandrew/fast-alpr) | 2.68M | 4.58 ± 0.85 | 15.06 ± 3.27 | 5.62 ± 2.96 | 25.27 ± 5.67 | 39.57 ± 4.30 | 39.67 | 34.79 |
| | YOLOv8n + SVTRv2-Tiny (Baseline) | 8.12M | **1.54 ± 0.41** | 12.38 ± 2.43 | 1.55 ± 0.20 | 15.84 ± 2.79 | 63.13 ± 0.70 | **78.60** | **5.60** |
| | **LiteALPR (Ours)** | **6.22M** | 1.63 ± 0.40 | **11.57 ± 5.87** | **1.49 ± 0.12** | **15.05 ± 6.01** | **66.45 ± 1.80** | 77.55 | 6.22 |
| **CPU** *(Ryzen 5 4600G)* | [fast-alpr (out-of-the-box)](https://github.com/ankandrew/fast-alpr) | 2.68M | **5.10 ± 1.18** | **24.02 ± 3.29** | 4.17 ± 2.44 | **33.30 ± 4.92** | **30.03 ± 0.80** | 39.67 | 34.79 |
| | YOLOv8n + SVTRv2-Tiny (Baseline) | 8.12M | 5.26 ± 3.55 | 58.53 ± 13.50 | **0.95 ± 0.78** | 65.08 ± 14.28 | 15.37 ± 0.30 | **78.60** | **5.60** |
| | **LiteALPR (Ours)** | **6.22M** | 6.52 ± 3.99 | 34.04 ± 9.32 | 1.80 ± 2.06 | 42.67 ± 8.66 | 23.44 ± 0.20 | 77.55 | 6.22 |

### Per-Subset End-to-End Breakdown (4-Bucket Resizing)

| Holdout Subset | Scenes | Baseline (%) | LiteALPR (%) |
| :--- | ---: | :---: | :---: |
| Standard Chinese (`ccpd_extra`) | 1,249 | **96.80** | 94.72 |
| Brazilian (`rodosol_test`) | 1,742 | **88.29** | 87.43 |
| Vietnamese (`vn_test`) | 135 | 73.33 | **81.48** |
| Tilted/Blurred Chinese (`ccpd_test`) | 2,104 | **60.12** | 58.94 |
| **Overall (E2E Acc. / CER)** | **5,230** | **78.60 / 5.60** | **77.55 / 6.22** |

> **Key Takeaway:** Against the domain-matched 4-bucket baseline (`YOLOv8n + SVTRv2-Tiny`, 8.12M params), LiteALPR reduces parameters by **23.4%** (6.22M) and speeds up inference by **52.5% on CPU** (23.44 vs. 15.37 FPS) and **5.3% on GPU** (66.45 vs. 63.13 FPS), trailing the baseline by 1.05 percentage points in overall E2E accuracy (99.90% detection recall, 16 FPs). Heavily tilted/blurred CCPD scenes (`ccpd_test`, 58.94%) account for most of the gap between full-pipeline E2E (77.55%) and ground-truth crop OCR (89.14%). Out-of-the-box `fast-alpr` (2.68M params, 2.79 GFLOPs) is smaller and faster on CPU (30.03 FPS) but reaches 39.67% E2E accuracy without domain adaptation.

### Reproduce Desktop Latency & E2E Benchmarks

```bash
# Latency & Throughput Benchmark
python test_performance/benchmark_latency.py \
  --images_dir dataset/det/test/images \
  --det_model_path pretrained_models/det/yolov8n_efficient/best_416.onnx \
  --rec_model_path pretrained_models/rec/svtr26_tiny/best.onnx \
  --device cuda:0  # or 'cpu'

# Full-Pipeline End-to-End Accuracy & CER Evaluation
python test_performance/eval_end2end_full.py \
  --dataset_dir dataset/e2e_holdout \
  --det_model pretrained_models/det/yolov8n_efficient/best_416.onnx \
  --rec_model pretrained_models/rec/svtr26_tiny/best.onnx
```

## 2. Detection Module Ablation (YOLOv8n-Efficient)

Models were trained at $640 \times 640$; all evaluation metrics (GFLOPs, ONNX latency over 1,000 samples with 50 warm-up runs, and mAP over 3 random seeds on the 5,230-image test set) are measured uniformly at the **$416 \times 416$ deployment resolution** ("Head" follows Ultralytics YAML terminology for the PANet neck and detection layers):

| Configuration | Params (M) | GFLOPs ($416\times416$) | GPU Lat. (ms) | CPU Lat. (ms) | $\text{mAP}_{50}$ (%) | $\text{mAP}_{50-95}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline YOLOv8n | 3.01 | 3.41 | 10.02 ± 1.29 | 40.92 ± 5.44 | 99.44 ± 0.03 | 88.01 ± 0.11 |
| Head (C3Ghost) | 2.52 | 2.99 | 9.54 ± 1.24 | 19.69 ± 4.26 | 99.43 ± 0.06 | **88.98 ± 0.14** |
| Backbone (C3Ghost) | 2.49 | 2.78 | 9.04 ± 0.51 | 19.59 ± 3.38 | 99.44 ± 0.02 | 88.60 ± 0.12 |
| **YOLOv8n-Efficient (Ours)** | **2.00** | **2.36** | **9.02 ± 0.98** | **17.45 ± 1.62** | **99.45 ± 0.03** | 88.07 ± 0.12 |

> Replacing C2f with C3Ghost across both backbone and neck reduces $416 \times 416$ compute by **30.8%** ($3.41 \rightarrow 2.36$ GFLOPs), cuts parameters by **33.6%** ($3.01\text{M} \rightarrow 2.00\text{M}$), and delivers a **2.34× CPU speedup** ($40.92 \rightarrow 17.45$ ms) while matching the baseline in $\text{mAP}_{50}$ (99.45%) and $\text{mAP}_{50-95}$ (88.07%).

## 3. Recognition Module Ablation (SVTR26-Tiny)

Impact of pruning 2D attention in favor of the Efficient RCTC Decoder (Height-wise Average Pooling) and synthetic degradation augmentations (latency measured over 1,000 samples with 50 warm-up runs; accuracy and CER over 3 random seeds):

| Decoder Architecture | Synthetic Aug. | Params (M) | GPU Lat. (ms) | CPU Lat. (ms) | Accuracy (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (2D Attention) | No | 5.11 | 5.55 ± 0.92 | 10.97 ± 1.26 | 86.72 ± 0.12 | 4.13 ± 0.09 |
| Baseline (2D Attention) | Yes | 5.11 | 5.85 ± 0.89 | 10.84 ± 0.76 | 88.48 ± 0.11 | 3.46 ± 0.06 |
| w/o Self-Attention | No | 4.33 | 5.95 ± 1.16 | 9.79 ± 0.74 | 85.98 ± 0.15 | 4.29 ± 0.11 |
| w/o Self-Attention | Yes | 4.33 | 5.58 ± 0.77 | 9.81 ± 0.73 | 88.52 ± 0.12 | 3.48 ± 0.05 |
| **Efficient RCTC (Ours)** | No | **4.22** | 5.29 ± 1.19 | 9.05 ± 1.34 | 86.55 ± 0.09 | 4.16 ± 0.06 |
| **Efficient RCTC (Ours)** | **Yes** | **4.22** | **5.03 ± 0.51** | **8.32 ± 0.98** | **89.14 ± 0.10** | **3.28 ± 0.04** |

> Overall across the 25,000 test crops, SVTR26-Tiny attains **89.14% sequence accuracy** (3.28% CER), or **87.96%** across the 22,350 real-image crops when excluding the 2,650 synthetic Vietnamese samples.

## 4. Regional Subsets & Unseen-Plate Diagnostic Audit

All baselines were fine-tuned on the same 25,000-crop training split using the same 38-token vocabulary (A–Z, Vietnamese `Đ`, 0–9, and CTC blank; Chinese provincial prefixes stripped; two-line plates concatenated row-wise) and synthetic augmentations, evaluated at each model's native input configuration (4-bucket resizing for `SVTR26-Tiny` and `SVTRv2-Tiny`; $48 \times 320$ for `PP-OCRv4-Mobile`; $32 \times 128$ for `CRNN`; $384 \times 384$ for `TrOCR`):

| Model | Params | Vietnam & Aux. Acc (%) | Brazil Acc (%) | China Acc (%) | Vietnam & Aux. CER (%) | Brazil CER (%) | China CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [CRNN (ResNet34)](https://arxiv.org/abs/1507.05717) | 24.50M | 35.17 | 55.96 | 67.88 | 36.28 | 18.31 | 11.64 |
| [PP-OCRv4-Mobile](https://github.com/PaddlePaddle/PaddleOCR) | **1.89M** | 64.50 | 87.28 | 92.05 | 13.00 | 3.02 | 1.84 |
| [TrOCR](http://dx.doi.org/10.1609/aaai.v37i11.26538) | 333.92M | 70.20 | 90.85 | 91.45 | 11.44 | 2.50 | 2.03 |
| [SVTRv2-Tiny](https://arxiv.org/abs/2411.15858) | 5.11M | 73.39 | 90.54 | 95.96 | 8.55 | 2.19 | 0.83 |
| **SVTR26-Tiny (Ours)** | 4.22M | **74.56** | **91.03** | **96.49** | **8.23** | **2.06** | **0.72** |

### Diagnostic Breakdown of SVTR26-Tiny (Unseen-Plate Audit & Per-Source Accuracy)

| Diagnostic Subset | Crops | Acc. (%) | CER (%) |
| :--- | ---: | :---: | :---: |
| **Strictly Unseen-Plate Audit (Zero Training Overlap)** | **20,036** | **89.25** | **3.08** |
| China (CBLPRD, unseen) | 9,448 | 96.49 | 0.72 |
| Brazil (RodoSol-ALPR, unseen) | 5,917 | 88.03 | 2.80 |
| Vietnam & Auxiliary (unseen) | 4,671 | 76.17 | 7.14 |
| **Homogeneous CBLPRD Clean vs. Degraded (Unseen)** | | | |
| Clean plates | 4,718 | 99.45 | 0.10 |
| Synthetically degraded plates | 4,730 | 93.53 | 1.33 |
| **Vietnam & Auxiliary Per-Source Breakdown (Full Test, $n=5,990$)** | | | |
| Synthetic Vietnamese | 2,650 | 99.13 | 0.11 |
| Public Vietnamese (`vietnam_3173`) | 1,503 | 68.13 | 8.56 |
| Public Vietnamese (`vietnam_1383`) | 680 | 52.06 | 13.87 |
| Self-collected Vietnamese | 487 | 38.81 | 17.12 |
| [Auxiliary European](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset) (original) | 338 | 75.15 | 4.89 |
| [Auxiliary European](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset) (heavily degraded) | 332 | 5.42 | 58.17 |

To reproduce the unseen-plate audit using `manifests/ocr_split_manifest.csv`:

```bash
python test_performance/audit_unseen_plates.py \
  --manifest manifests/ocr_split_manifest.csv \
  --labels dataset/rec/test_labels.txt \
  --dataset_dir dataset/rec
```
