# Benchmarks & Experimental Results

LiteALPR was trained and rigorously benchmarked on a large multi-national dataset (**52,595 total images**), comprising:

- **[Brazil (RodoSol-ALPR)](https://github.com/raysonlaroca/rodosol-alpr-dataset)**: 20,000 images
- **[China (CBLPRD-330k)](https://github.com/SunlifeV/CBLPRD-330k)**: 20,000 images
- **[Vietnam](https://www.kaggle.com/datasets/duydieunguyen/licenseplates)**: 12,595 images (public dataset + self-collected traffic footage)

### Detection Dataset
The full-resolution images were split for the YOLOv8n-Efficient detector:

- **Training**: 42,136 images
- **Validation**: 5,229 images
- **Test**: 5,230 images

### Recognition Dataset
The **52,595** localized license plates were cropped out. To ensure the model is robust to real-world conditions, **50% of the crops were artificially degraded** (e.g., motion blur, noise).

- **Training**: 25,000 crops
- **Validation**: 2,595 crops
- **Test**: 25,000 crops

**Cross-Regional Test Set Breakdown** (25,000 crops):

- **Brazil**: 9,560 plates
- **China**: 9,450 plates
- **Vietnam**: 5,990 plates

!!! info "Benchmarking Methodology"
    All benchmarks were evaluated in native ONNX FP32 precision. To ensure statistical reliability, **latency and throughput (mean ± std)** were measured over **1,000 real-world test images** (or cropped plates) across **5 consecutive execution passes**.

---

## 1. End-to-End Pipeline Performance

Comparison against existing lightweight open-source ALPR pipelines:

| Platform | Model / Pipeline | Pre (ms) | Infer (ms) | Post (ms) | Total (ms) | Throughput (FPS) | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPU** *(RTX 3060 12GB)* | YOLOv8n + SVTRv2 (Baseline) | **1.54 ± 0.41** | 12.38 ± 2.43 | 1.55 ± 0.20 | 15.84 ± 2.79 | 63.13 ± 0.70 | 88.25% |
| | [fast-alpr (out-of-the-box)](https://github.com/ankandrew/fast-alpr) | 4.58 ± 0.85 | 15.06 ± 3.27 | 5.62 ± 2.96 | 25.27 ± 5.67 | 39.57 ± 4.30 | 22.67% |
| | **LiteALPR (Ours)** | 1.63 ± 0.40 | **11.57 ± 5.87** | **1.49 ± 0.12** | **15.05 ± 6.01** | **66.45 ± 1.80** | **89.15%** |
| **CPU** *(Ryzen 5 4600G)* | YOLOv8n + SVTRv2 (Baseline) | 5.26 ± 3.55 | 58.53 ± 13.50 | **0.95 ± 0.78** | 65.08 ± 14.28 | 15.37 ± 0.30 | 88.25% |
| | [fast-alpr (out-of-the-box)](https://github.com/ankandrew/fast-alpr) | **5.10 ± 1.18** | **24.02 ± 3.29** | 4.17 ± 2.44 | **33.30 ± 4.92** | **30.03 ± 0.80** | 22.67% |
| | **LiteALPR (Ours)** | 6.52 ± 3.99 | 34.04 ± 9.32 | 1.80 ± 2.06 | 42.67 ± 8.66 | 23.44 ± 0.20 | **89.15%** |

> **Key Takeaway:** While off-the-shelf pipelines such as [`fast-alpr (out-of-the-box)`](https://github.com/ankandrew/fast-alpr) degrade down to 22.67% accuracy without domain retraining, LiteALPR achieves **89.15% sequence accuracy** while maintaining ultra-high throughput (**66.45 ± 1.80 FPS** on GPU, **23.44 ± 0.20 FPS** on CPU), delivering a steady 5.26% GPU speedup and 52.5% CPU speedup over the matched baseline (63.13 FPS on GPU, 15.37 FPS on CPU, 88.25% accuracy).

### Reproduce Desktop Benchmarks
To evaluate the end-to-end pipeline latency and throughput on your own hardware, run the following benchmark script:

```bash
python test_performance/benchmark_latency.py \
  --images_dir dataset/det/test/images \
  --det_model_path output/det/yolov8n_efficient/train/weights/best_416.onnx \
  --rec_model_path output/rec/svtr26_tiny/train/best.onnx \
  --device cuda:0  # or use 'cpu'
```

---

## 2. Detection Module Ablation (YOLOv8n-Efficient)

Impact of progressively replacing C2f blocks with lightweight C3Ghost modules across head and backbone (mAP is reported as mean ± std over 3 random seeds):

| Configuration | Parameters (M) | GFLOPs | GPU Latency (ms) | CPU Latency (ms) | $\text{mAP}_{50}$ (%) | $\text{mAP}_{50-95}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline YOLOv8n | 3.01 | 8.19 | 10.02 ± 1.29 | 40.92 ± 5.44 | **99.45 ± 0.04%** | 88.48 ± 0.12% |
| Head (C3Ghost) | 2.52 | 7.19 | 9.54 ± 1.24 | 19.69 ± 4.26 | 99.44 ± 0.05% | **89.33 ± 0.15%** |
| Backbone (C3Ghost) | 2.49 | 6.70 | 9.04 ± 0.51 | 19.59 ± 3.38 | **99.45 ± 0.04%** | 89.20 ± 0.14% |
| **YOLOv8n-Efficient (Ours)** | **2.00** | **5.69** | **9.02 ± 0.98** | **17.45 ± 1.62** | **99.45 ± 0.05%** | 88.41 ± 0.16% |

> Replacing C2f with C3Ghost across both backbone and head achieves a **2.35× speedup on CPU** (~$40.92 \rightarrow$ ~17.45 ms) and **33.5% parameter reduction** ($3.01\text{M} \rightarrow 2.00\text{M}$) with **zero accuracy drop** (maintains ~99.45% $\text{mAP}_{50}$).

---

## 3. Recognition Module Ablation (SVTR26-Tiny)

Impact of replacing 2D attention matrices with Height-wise Average Pooling (HAP) and data synthesis augmentation (Accuracy and CER are reported as mean ± std over 3 random seeds):

| Decoder Architecture | Synthetic Degradation | Params (M) | GPU Lat. (ms) | CPU Lat. (ms) | Accuracy (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Full Attention) | No | 5.11 | 5.55 ± 0.92 | 10.97 ± 1.26 | 86.71 ± 0.12% | 4.13 ± 0.09% |
| Baseline (Full Attention) | Yes | 5.11 | 5.85 ± 0.89 | 10.84 ± 0.76 | 88.25 ± 0.11% | 3.56 ± 0.06% |
| w/o Self-Attention | No | 4.33 | 5.95 ± 1.16 | 9.79 ± 0.74 | 85.98 ± 0.15% | 4.29 ± 0.11% |
| w/o Self-Attention | Yes | 4.33 | 5.58 ± 0.77 | 9.81 ± 0.73 | 88.52 ± 0.12% | 3.48 ± 0.05% |
| **HAP (Ours)** | No | **4.22** | 5.29 ± 1.19 | 9.05 ± 1.34 | 86.55 ± 0.09% | 4.16 ± 0.06% |
| **HAP + Degrade (Ours)** | **Yes** | **4.22** | **5.03 ± 0.51** | **8.32 ± 0.98** | **89.15 ± 0.10%** | **3.28 ± 0.04%** |

> Height-wise Average Pooling (HAP) completely replaces 2D attention matrices, reducing CPU recognition latency from **~10.84 ms** down to **~8.32 ms** while maintaining a sequence recognition accuracy of **89.15%** and lowering Character Error Rate (CER) to **3.28%**.

---

## 4. Multi-National Robustness across Regional Subsets

Evaluation of SVTR26-Tiny against widely-adopted sequence recognition baselines across 3 diverse geographic subsets (**25,000 total test samples**: 5,990 from Vietnam, 9,560 from Brazil, and 9,450 from China), fine-tuned under matched training conditions using identical crop resolution ($128 \times 32$), vocabulary, and augmentations.

> **Evaluation Vocabulary:** All models are evaluated on a unified 38-token Latin alphanumeric vocabulary (26 uppercase letters **A–Z**, Vietnamese **Đ**, 10 digits **0–9**, and 1 CTC blank token). To ensure a fair regional comparison, non-Latin provincial logograms (e.g., Chinese prefix characters) are excluded during label normalization, and multi-line plates are concatenated into a contiguous 1D sequence.

The table below presents the performance of the evaluated lightweight architectures across regional benchmarks:

| No. | Model | Params | Vietnam Acc (%) | Brazil Acc (%) | China Acc (%) | Vietnam CER (%) | Brazil CER (%) | China CER (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [CRNN](https://arxiv.org/abs/1507.05717) | 24.50M | 35.17% | 55.96% | 67.88% | 36.28% | 18.31% | 11.64% |
| 2 | [PP-OCRv4-Mobile](https://github.com/PaddlePaddle/PaddleOCR) | 19.30M | 64.50% | 87.28% | 92.05% | 13.00% | 3.02% | 1.84% |
| 3 | [TrOCR](http://dx.doi.org/10.1609/aaai.v37i11.26538) | 333.92M | 70.20% | 90.85% | 91.45% | 11.44% | 2.50% | 2.03% |
| 4 | [SVTRv2-Tiny](https://arxiv.org/abs/2411.15858) | 5.11M | 81.36% | 90.54% | 96.00% | 5.05% | 2.19% | 0.82% |
| 5 | **SVTR26-Tiny (Ours)** | **4.22M** | **82.93%** | **90.98%** | **96.51%** | **4.69%** | **2.06%** | **0.71%** |

> **Summary:** SVTR26-Tiny attains top performance among the evaluated architectures while requiring the smallest parameter footprint (**4.22M**). It demonstrates multi-national robustness across disparate plate aspect ratios, font morphologies, and line layouts within the pooled distribution.
