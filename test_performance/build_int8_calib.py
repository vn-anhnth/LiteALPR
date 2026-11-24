import argparse
import os
import random

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image


def set_workspace_size(config, workspace_bytes=1 << 30):
    """Set TensorRT workspace limit across TRT 8.2 (Jetson Nano) and TRT 8.5+."""
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes


def preprocess_image(img_bgr, target_w=64, target_h=64):
    """Preprocess BGR image using PIL BICUBIC to match LiteALPR."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_resized = np.array(img_pil.resize((target_w, target_h), Image.BICUBIC))
    img_float = img_resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
    img_float = (img_float - 0.5) / 0.5
    return np.ascontiguousarray(img_float, dtype=np.float32)


def load_calib_paths(labels_file, dataset_dir, num_images, seed=42):
    paths = []
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 1:
                continue
            img_path = os.path.join(dataset_dir, parts[0])
            if os.path.isfile(img_path):
                paths.append(img_path)

    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:num_images]


class SVTREntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, cache_file, batch_size=1, target_w=64, target_h=64):
        super().__init__()
        self.image_paths = image_paths
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.target_w = target_w
        self.target_h = target_h
        self.current_index = 0

        nbytes = batch_size * 3 * target_h * target_w * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_paths):
            return None

        batch_imgs = []
        while len(batch_imgs) < self.batch_size and self.current_index < len(self.image_paths):
            img_path = self.image_paths[self.current_index]
            self.current_index += 1
            img = cv2.imread(img_path)
            if img is None:
                continue
            batch_imgs.append(preprocess_image(img, self.target_w, self.target_h))

        if len(batch_imgs) < self.batch_size:
            return None

        batch_np = np.ascontiguousarray(np.stack(batch_imgs, axis=0), dtype=np.float32)
        cuda.memcpy_htod(self.device_input, batch_np)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        out_dir = os.path.dirname(os.path.abspath(self.cache_file))
        os.makedirs(out_dir, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[INFO] Saved calibration cache to: {self.cache_file}")

    def free(self):
        if self.device_input is not None:
            self.device_input.free()
            self.device_input = None


def main():
    parser = argparse.ArgumentParser(description="Generate TensorRT INT8 calibration cache for SVTR26-Tiny")
    parser.add_argument("--onnx", default="pretrained_models/rec/svtr26_tiny/best.onnx")
    parser.add_argument("--calib-out", default="svtr26_tiny.calib")
    parser.add_argument("--labels-file", default="dataset/rec/val_labels.txt")
    parser.add_argument("--dataset-dir", default="dataset/rec")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.calib_out):
        os.remove(args.calib_out)

    image_paths = load_calib_paths(args.labels_file, args.dataset_dir, args.num_images, seed=args.seed)
    if not image_paths:
        raise RuntimeError(f"No valid images found from {args.labels_file}")
    print(f"[INFO] Loaded {len(image_paths)} calibration images from {args.labels_file}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    trt_parser = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as f:
        if not trt_parser.parse(f.read()):
            for i in range(trt_parser.num_errors):
                print(trt_parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX model: {args.onnx}")

    config = builder.create_builder_config()
    set_workspace_size(config, 1 << 30)  # 1 GB
    config.set_flag(trt.BuilderFlag.INT8)

    input_name = network.get_input(0).name
    opt_profile = builder.create_optimization_profile()
    opt_profile.set_shape(
        input_name,
        (1, 3, 32, 32),
        (1, 3, 64, 64),
        (1, 3, 128, 128),
    )
    config.add_optimization_profile(opt_profile)

    calib_profile = builder.create_optimization_profile()
    calib_shape = (args.batch_size, 3, args.height, args.width)
    calib_profile.set_shape(input_name, calib_shape, calib_shape, calib_shape)
    config.set_calibration_profile(calib_profile)

    calibrator = SVTREntropyCalibrator(
        image_paths=image_paths,
        cache_file=args.calib_out,
        batch_size=args.batch_size,
        target_w=args.width,
        target_h=args.height,
    )
    config.int8_calibrator = calibrator

    print("[INFO] Running INT8 calibration (this may take a few minutes on Jetson Nano)...")
    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("Calibration / build_serialized_network returned None")
    finally:
        calibrator.free()

    print(f"[DONE] Calibration file ready: {args.calib_out}")


if __name__ == "__main__":
    main()
