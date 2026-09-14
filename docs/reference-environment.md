# Reference Environment

The performance metrics and evaluations presented in the LiteALPR documentation were generated under the following reference environment to ensure reproducibility.

## Hardware Specifications
- **CPU**: AMD Ryzen 5 4600G with Radeon Graphics (6 Cores, 12 Threads, 3.7 GHz)
- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **RAM**: 32 GB DDR4

## Software Environment
- **Operating System**: Windows 11 Pro (64-bit) / Ubuntu 22.04 LTS
- **Python Version**: Python 3.8.10
- **NVIDIA Driver**: 536.23 (or compatible)
- **CUDA Toolkit**: 12.1
- **cuDNN**: 8.9.x
- **PyTorch**: 2.4.1+cu121
- **ONNX Runtime**: onnxruntime-gpu 1.19.2

## Dependency Lockfile
For an exact reconstruction of the Python environment used during benchmarking, please refer to the `requirements.lock` file provided in the repository root.

You can replicate this environment via:
```bash
pip install -r requirements.lock
```
