# Getting Started

LiteALPR is distributed via PyPI and supports Python 3.8 through 3.12 across Windows, Linux, and macOS.

---

## Installation

You can install LiteALPR easily with `pip`. Choose the option that fits your deployment target:

=== "Standard (Auto-detect)"
    ```bash
    pip install litealpr
    ```
    *LiteALPR will automatically detect available acceleration hardware and run with default CPU/GPU fallbacks.*

=== "CPU Optimized"
    ```bash
    pip install litealpr[cpu]
    ```
    *Installs standard `onnxruntime` for optimal lightweight CPU inference.*

=== "CUDA GPU Accelerated"
    ```bash
    pip install litealpr[gpu]
    ```
    *Installs `onnxruntime-gpu` for CUDA execution provider acceleration on NVIDIA GPUs.*

=== "From Source"
    ```bash
    git clone https://github.com/vn-anhnth/LiteALPR.git
    cd LiteALPR
    pip install -e .
    ```

---

## Hardware Requirements

LiteALPR is designed to be lightweight, running efficiently on both standard CPU environments and GPU accelerators:

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 20.04+, macOS | Ubuntu 20.04/22.04 LTS, Windows 64-bit |
| **Python** | Python 3.8+ | Python 3.10 or 3.11 |
| **CPU** | Dual-core x86_64 / ARM64 | 4+ cores (Intel Core i5+, AMD Ryzen) |
| **RAM / Memory** | 2 GB (~350 MB memory footprint during inference) | 4 GB+ |
| **Storage** | ~100 MB for library and model weights | 1 GB+ (SSD recommended) |
| **GPU (Optional)** | None (runs smoothly on CPU) | NVIDIA GPU with CUDA 11.8 / 12.x |

---

## Automatic Model Downloads

When you first instantiate `LiteALPR()` in your code:

```python
from litealpr import LiteALPR

model = LiteALPR()
```

LiteALPR checks your local cache directory (`~/.cache/huggingface/hub/` or custom paths) for the official ONNX models. If they are not found, it automatically downloads the latest models from our HuggingFace repository:

- **Detection Model**: `yolov8n_efficient/best_416.onnx` (~8.1 MB)
- **Recognition Model**: `svtr26_tiny/best.onnx` (~17.0 MB)

You do not need to download or place model files manually!
