# Reference Environment

The performance metrics and evaluations reported in the *SoftwareX* manuscript and this documentation were benchmarked on the following workstation and embedded platforms.

## Workstation Hardware Specifications
- **CPU**: AMD Ryzen 5 4600G with Radeon Graphics (6 physical cores, 12 threads, 3.7 GHz; benchmarked with 6 intra-op threads)
- **GPU**: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **RAM**: 32 GB DDR4

## Workstation Software Environment
- **Operating System**: Windows 11 Pro (64-bit) / Ubuntu 22.04 LTS
- **Python**: Python 3.8+ (tested on Python 3.8.10)
- **CUDA Toolkit**: CUDA 11.8
- **Benchmark Runtime (Manuscript)**: PyTorch 2.1.0, ONNX Runtime 1.16.3
- **Pinned Lockfile (`requirements.lock`)**: `torch==2.2.0+cu118`, `onnxruntime-gpu==1.17.1`, `onnxruntime==1.19.2`, `ultralytics==8.4.98`

## Embedded Edge Platform (Jetson Nano)
- **Hardware**: NVIDIA Jetson Nano Developer Kit (4 GB LPDDR4, 128-core Maxwell GPU, Quad-core Cortex-A57)
- **Software**: Ubuntu 18.04 (JetPack 4.6.1), CUDA 10.2, cuDNN 8.2, TensorRT 8.2.1, Python 3.6.9

## Dependency Lockfile
To reconstruct the pinned Python development environment, install from `requirements.lock` in the repository root:

```bash
pip install -r requirements.lock
```
