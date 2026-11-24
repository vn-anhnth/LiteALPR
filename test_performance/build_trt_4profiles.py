import argparse
import os
import sys

import tensorrt as trt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_int8_calib import SVTREntropyCalibrator, load_calib_paths, set_workspace_size

SHAPES = [
    (1, 3, 64, 64),
    (1, 3, 48, 96),
    (1, 3, 40, 112),
    (1, 3, 32, 128),
]


def build_engine(onnx_path, engine_path, precision, calib_file=None, labels_file=None, dataset_dir=None):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    set_workspace_size(config, 1 << 30)  # 1 GB

    input_name = network.get_input(0).name
    for shape in SHAPES:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, shape, shape, shape)
        config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if not calib_file or not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration cache not found: {calib_file}")
        calib_profile = builder.create_optimization_profile()
        calib_shape = (1, 3, 64, 64)
        calib_profile.set_shape(input_name, calib_shape, calib_shape, calib_shape)
        config.set_calibration_profile(calib_profile)
        paths = load_calib_paths(labels_file, dataset_dir, 500) if (labels_file and dataset_dir) else []
        calibrator = SVTREntropyCalibrator(paths, calib_file, batch_size=1, target_w=64, target_h=64)
        config.int8_calibrator = calibrator

    print(f"[INFO] Building {engine_path} ({precision.upper()}) with 4 static profiles...")
    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"Failed to build {engine_path}")
        out_dir = os.path.dirname(os.path.abspath(engine_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)
    finally:
        if calibrator is not None:
            calibrator.free()

    print(f"[DONE] Saved {engine_path} ({os.path.getsize(engine_path) / (1024 * 1024):.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines with 4 static SVTR26-Tiny aspect-ratio profiles")
    parser.add_argument("--onnx", default="pretrained_models/rec/svtr26_tiny/best.onnx")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8", "all"], default="all")
    parser.add_argument("--calib", default="svtr26_tiny.calib")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--dataset-dir", default=None)
    args = parser.parse_args()

    targets = ["fp32", "fp16", "int8"] if args.precision == "all" else [args.precision]
    for prec in targets:
        out_engine = os.path.join(args.output_dir, f"svtr26_tiny_{prec}.engine")
        build_engine(
            args.onnx,
            out_engine,
            prec,
            calib_file=args.calib,
            labels_file=args.labels_file,
            dataset_dir=args.dataset_dir,
        )


if __name__ == "__main__":
    main()
