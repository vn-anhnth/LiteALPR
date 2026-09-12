# Benchmarks & Experimental Results

LiteALPR was rigorously benchmarked on a comprehensive multi-national dataset totaling **52,595 images**:

- **[Brazil (RodoSol-ALPR)](https://github.com/raysonlaroca/rodosol-alpr-dataset)**: **20,000 images** (with 9,560 evaluated in test set)
- **[China (CBLPRD-330k)](https://github.com/SunlifeV/CBLPRD-330k)**: **20,000 images** (with 9,450 evaluated in test set)
- **[Vietnam](https://www.kaggle.com/datasets/duydieunguyen/licenseplates)**: **12,595 images** (public dataset + self-collected traffic footage, with 5,990 evaluated in test set)

For the text recognition task, the full corpus of **52,595 localized plate crops** was used. To simulate challenging real-world environments and ensure the OCR engine's resilience, **half (50%) of these crops were artificially distorted** to emulate out-of-focus optics and high-speed motion blur. The recognition dataset was distributed into:

- **Training set**: 25,000 crops
- **Validation set**: 2,595 crops
- **Test set**: 25,000 crops (comprising 9,560 from Brazil, 9,450 from China, and 5,990 from Vietnam)

All benchmarks were evaluated in native ONNX FP32 precision, reporting latency and throughput as $\text{mean} \pm \text{std}$ evaluated over 1,000 real-world test images or cropped plates across 5 consecutive execution passes.

---

## 1. End-to-End Pipeline Performance

Comparison against existing state-of-the-art and lightweight open-source ALPR pipelines:

| Platform | Model / Pipeline | Pre (ms) | Infer (ms) | Post (ms) | Total (ms) | Throughput (FPS) | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPU** *(RTX 3060 12GB)* | YOLOv8n + SVTRv2 (Baseline) | **1.54 ± 0.41** | 12.38 ± 2.43 | 1.55 ± 0.20 | 15.84 ± 2.79 | 63.1 ± 0.7 | 88.60% |
| | [fast-alpr](https://github.com/ankandrew/fast-alpr) | 4.58 ± 0.85 | 15.06 ± 3.27 | 5.62 ± 2.96 | 25.27 ± 5.67 | 40.1 ± 4.3 | 22.67% |
| | **LiteALPR (Ours)** | 1.63 ± 0.40 | **11.57 ± 5.87** | **1.49 ± 0.12** | **15.05 ± 6.01** | **66.5 ± 1.8** | **89.15%** |
| **CPU** *(Ryzen 5 4600G)* | YOLOv8n + SVTRv2 (Baseline) | 5.26 ± 3.55 | 58.53 ± 13.50 | **0.95 ± 0.78** | 65.08 ± 14.28 | 15.4 ± 0.3 | 88.60% |
| | [fast-alpr](https://github.com/ankandrew/fast-alpr) | **5.10 ± 1.18** | **24.02 ± 3.29** | 4.17 ± 2.44 | **33.30 ± 4.92** | **30.1 ± 0.8** | 22.67% |
| | **LiteALPR (Ours)** | 6.52 ± 3.99 | 34.04 ± 9.32 | 1.80 ± 2.06 | 42.67 ± 8.66 | 23.4 ± 0.2 | **89.15%** |

> **Key Takeaway:** While alternative lightweight pipelines such as [`fast-alpr`](https://github.com/ankandrew/fast-alpr) degrade down to 22.67% accuracy under blur, low light, and tilt angles, LiteALPR preserves **89.15% sequence accuracy** while maintaining ultra-high throughput (**66.5 FPS** on GPU, **23.4 FPS** on CPU).

---

## 2. Detection Module Ablation (YOLOv8n-Efficient)

Impact of progressively replacing C2f blocks with lightweight C3Ghost modules across head and backbone:

| Configuration | Parameters (M) | GFLOPs | GPU Latency (ms) | CPU Latency (ms) | mAP@50 (%) | mAP@50-95 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline YOLOv8n | 3.01 | 8.19 | 10.02 ± 1.29 | 40.92 ± 5.44 | **99.45%** | 88.48% |
| Head (C3Ghost) | 2.52 | 7.19 | 9.54 ± 1.24 | 19.69 ± 4.26 | 99.44% | **89.33%** |
| Backbone (C3Ghost) | 2.49 | 6.70 | 9.04 ± 0.51 | 19.59 ± 3.38 | **99.45%** | 89.20% |
| **YOLOv8n-Efficient (Ours)** | **2.00** | **5.69** | **9.02 ± 0.98** | **17.45 ± 1.62** | **99.45%** | 88.41% |

> Replacing C2f with C3Ghost across both backbone and head achieves a **2.35× speedup on CPU** ($40.92 \rightarrow 17.45\text{ ms}$) and **33.5% parameter reduction** ($3.01\text{M} \rightarrow 2.00\text{M}$) with **zero accuracy drop** (maintains 99.45% mAP@50).

---

## 3. Recognition Module Ablation (SVTR26-Tiny)

Ablation analysis of decoders (Height-wise Average Pooling) and synthetic degradation training on **25,000 test crops** (containing 50% synthetically degraded plates):

| Decoder Architecture | Synthetic Degradation | Params (M) | GPU Lat. (ms) | CPU Lat. (ms) | Accuracy (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (Full Attention) | No | 5.11 | 5.55 ± 0.92 | 10.97 ± 1.26 | 86.71% | 4.13% |
| Baseline (Full Attention) | Yes | 5.11 | 5.85 ± 0.89 | 10.84 ± 0.76 | 88.25% | 3.56% |
| w/o Self-Attention | No | 4.33 | 5.95 ± 1.16 | 9.79 ± 0.74 | 85.98% | 4.29% |
| w/o Self-Attention | Yes | 4.33 | 5.58 ± 0.77 | 9.81 ± 0.73 | 88.52% | 3.48% |
| **HAP (Ours)** | No | **4.22** | 5.29 ± 1.19 | 9.05 ± 1.34 | 86.55% | 4.16% |
| **HAP + Degrade (Ours)** | **Yes** | **4.22** | **5.03 ± 0.51** | **8.32 ± 0.98** | **89.15%** | **3.28%** |

> Height-wise Average Pooling (HAP) completely replaces 2D attention matrices, slashing CPU recognition latency from **10.97 ms** down to **8.32 ms** while synthetic degradation training boosts sequence recognition accuracy up to **89.15%** and lowers Character Error Rate (CER) to **3.28%**.

---

## 4. Cross-Regional Generalization Benchmark

Evaluation of SVTR26-Tiny against state-of-the-art sequence recognition baselines across 3 diverse geographic subsets (**25,000 total test samples**: 5,990 from Vietnam, 9,560 from Brazil, and 9,450 from China).

> **Alphanumeric Vocabulary:** All evaluated recognition models transcribe standard Latin uppercase alphabets and numeric digits (**A–Z**, **Đ**, **0–9**, totaling 37 alphanumeric characters plus 1 CTC blank token). For multi-line plates (e.g. Vietnam and Brazil formats), Chinese provincial Hanzi characters in CBLPRD-330k were normalized or evaluated on the alphanumeric sequence, ensuring a fair, unified Latin-character benchmark across all international datasets.

The table below presents the performance of all 30 evaluated text recognition architectures, ranked in progressive order of regional generalization:

| No. | Model | Vietnam Acc (%) | Brazil Acc (%) | China Acc (%) | Vietnam CER (%) | Brazil CER (%) | China CER (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [RobustScanner](https://doi.org/10.1007/978-3-030-58529-7_9) | 6.93% | 48.43% | 58.65% | 46.19% | 18.85% | 15.39% |
| 2 | [RFL](https://doi.org/10.1007/978-3-030-86549-8_19) | 8.85% | 55.65% | 46.31% | 49.95% | 23.87% | 35.28% |
| 3 | [SPIN](https://doi.org/10.1609/aaai.v35i4.16442) | 9.95% | 72.68% | 59.95% | 44.52% | 12.32% | 22.32% |
| 4 | [StarNet](https://doi.org/10.5244/c.30.43) | 19.21% | 44.29% | 62.86% | 39.65% | 21.76% | 12.06% |
| 5 | [Rosetta](https://doi.org/10.1145/3219819.3219861) | 23.85% | 36.41% | 61.31% | 39.27% | 27.96% | 17.10% |
| 6 | [RARE](https://doi.org/10.1109/cvpr.2016.452) | 25.76% | 63.04% | 71.68% | 37.08% | 13.76% | 11.47% |
| 7 | [NRTR](https://doi.org/10.1109/icdar.2019.00130) | 32.71% | 62.30% | 35.24% | 35.72% | 15.13% | 36.57% |
| 8 | [CRNN](https://arxiv.org/abs/1507.05717) | 35.17% | 55.96% | 67.88% | 36.28% | 18.31% | 11.64% |
| 9 | [PREN](https://doi.org/10.1109/cvpr46437.2021.00035) | 43.63% | 82.87% | 84.21% | 23.85% | 5.82% | 4.42% |
| 10 | [ViTSTR](https://doi.org/10.1007/978-3-030-86549-8_21) | 45.18% | 73.60% | 66.31% | 26.77% | 8.97% | 12.62% |
| 11 | [SATRN](https://doi.org/10.1109/cvprw50498.2020.00236) | 45.21% | 75.11% | 79.06% | 17.06% | 5.24% | 3.91% |
| 12 | [SRN](https://doi.org/10.1109/cvpr42600.2020.01213) | 46.71% | 73.74% | 69.55% | 23.93% | 8.77% | 11.63% |
| 13 | [SEED](https://doi.org/10.1109/cvpr42600.2020.01354) | 48.44% | 79.06% | 79.93% | 22.55% | 6.45% | 6.06% |
| 14 | [PP-OCRv3-Mobile](https://github.com/PaddlePaddle/PaddleOCR) | 56.59% | 85.00% | 83.85% | 16.89% | 3.90% | 3.76% |
| 15 | [RepSVTR](https://arxiv.org/abs/2411.15858) | 61.43% | 88.40% | 94.00% | 12.84% | 2.65% | 1.42% |
| 16 | [PP-OCRv6-Small](https://github.com/PaddlePaddle/PaddleOCR) | 63.74% | 86.01% | 83.72% | 12.15% | 3.57% | 3.85% |
| 17 | [PP-OCRv5-Mobile](https://github.com/PaddlePaddle/PaddleOCR) | 64.35% | 89.07% | 84.95% | 13.29% | 2.86% | 3.43% |
| 18 | [PP-OCRv4-Mobile](https://github.com/PaddlePaddle/PaddleOCR) | 64.50% | 87.28% | 92.05% | 13.00% | 3.02% | 1.84% |
| 19 | [TrOCR](http://dx.doi.org/10.1609/aaai.v37i11.26538) | 70.20% | 90.85% | 91.45% | 11.44% | 2.50% | 2.03% |
| 20 | [SVTR](https://doi.org/10.3390/electronics13234756) | 70.52% | 90.04% | 93.57% | 11.11% | 2.39% | 1.44% |
| 21 | [PP-OCRv5-Server](https://github.com/PaddlePaddle/PaddleOCR) | 70.70% | 91.17% | 91.53% | 11.51% | 2.11% | 1.96% |
| 22 | [SAR](https://doi.org/10.1609/aaai.v33i01.33018610) | 72.30% | 89.04% | 95.00% | 9.86% | 3.35% | 1.06% |
| 23 | [PP-OCRv6-Medium](https://github.com/PaddlePaddle/PaddleOCR) | 73.32% | 89.72% | 82.71% | 9.42 | 2.61% | 4.27% |
| 24 | [VisionLAN](https://doi.org/10.1109/iccv48922.2021.01393) | 73.37% | **92.26%** | 95.14% | 10.72% | 1.87% | 1.11% |
| 25 | [PARSeq](https://doi.org/10.1007/978-3-031-19815-1_11) | 74.01% | 91.53% | 95.71% | 10.08% | 2.11% | 0.94% |
| 26 | [PP-OCRv4-Server](https://github.com/PaddlePaddle/PaddleOCR) | 74.34% | 92.08% | 86.65% | 10.38% | **1.81%** | 3.19% |
| 27 | [ABINet](https://doi.org/10.1109/cvpr46437.2021.00702) | 76.05% | 91.07% | 95.39% | 8.17% | 2.19% | 1.03% |
| 28 | [CPPD](https://doi.org/10.1109/tpami.2025.3545453) | 76.22% | 90.80% | 95.16% | 8.14% | 2.11% | 1.08% |
| 29 | [SVTRv2-Tiny](https://arxiv.org/abs/2411.15858) | 81.36% | 90.54% | 96.00% | 5.05% | 2.19% | 0.82% |
| 30 | **SVTR26-Tiny (Ours)** | **82.93%** | 90.98% | **96.51%** | **4.69%** | 2.06% | **0.71%** |

> **Summary:** SVTR26-Tiny attains top performance on the challenging Vietnam benchmark (**82.93%**) and China benchmark (**96.51%** with **0.71% CER**), while achieving competitive **90.98%** accuracy in Brazil.
