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

=== "CPU (ONNX) Inference"
    ```bash
    pip install litealpr[cpu]
    ```
    *Installs standard `onnxruntime` for optimal lightweight CPU inference.*

=== "NVIDIA GPU (ONNX) Acceleration"
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

## Automatic Model Downloads

When you first instantiate `LiteALPR()` in your code:

```python
from litealpr import LiteALPR

model = LiteALPR()
```

LiteALPR checks your local cache directory (`~/.cache/huggingface/hub/` or custom paths) for the official ONNX models. If they are not found, it automatically downloads the latest models from our [Hugging Face repository](https://huggingface.co/anhone3/LiteALPR/tree/main):

- **Detection Model**: `yolov8n_efficient/best_416.onnx` (~8.1 MB)
- **Recognition Model**: `svtr26_tiny/best.onnx` (~17.0 MB)

You do not need to download or place model files manually!
