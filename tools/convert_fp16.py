import onnx
from onnxconverter_common import float16

# Load ONNX FP32 file
model = onnx.load("output/rec/svtr26_tiny/train/best.onnx")

# Automatically convert to FP16 but retain FP32 for sensitive functions.
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

# Save ONNX FP16 file
onnx.save(model_fp16, "output/rec/svtr26_tiny/train/best_fp16.onnx")
print("Convert to FP16 successfully!")
