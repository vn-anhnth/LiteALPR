"""
Benchmark Script for LiteALPR Pipeline Latency & Throughput (FPS).
Measures:
- Pre-processing (ms)
- Inference (ms) [Detection + Recognition]
- Post-processing (ms)
- Total Pipeline Latency (ms)
- Throughput (FPS)
Strictly evaluated on ONNX FP32 runtime at 416x416 resolution.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

# Ensure LiteALPR root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
litealpr_root = os.path.abspath(os.path.join(current_dir, '..'))
if litealpr_root not in sys.path:
    sys.path.insert(0, litealpr_root)

# Disable Ultralytics auto check/update
os.environ['YOLO_AUTOINSTALL'] = 'False'
import ultralytics

ultralytics.checks = lambda: None

# Attempt to dynamically add torch/lib to PATH for CUDA DLL discovery on Windows
try:
    import torch
    torch_cuda_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_cuda_lib):
        os.environ['PATH'] = torch_cuda_lib + os.pathsep + os.environ.get('PATH', '')
except ImportError:
    pass

from litealpr import LiteALPR


def benchmark_pipeline_dataset(image_dir,
                               det_model_path=None,
                               rec_model_path=None,
                               num_samples=1000,
                               num_runs=5,
                               num_warmup=10,
                               device="cpu"):
    print("=" * 68)
    print(f" LiteALPR FULL PIPELINE BENCHMARK (Device: {device.upper()})")
    print(f" Dataset : {image_dir}")
    print(f" Samples : {num_samples} images | Iterations : {num_runs} passes")
    print("=" * 68)

    # Resolve local default ONNX model paths if not provided
    if det_model_path is None:
        det_model_path = os.path.join(litealpr_root, "output", "det", "yolov8n_efficient", "train", "weights", "best_416.onnx")
    if rec_model_path is None:
        rec_model_path = os.path.join(litealpr_root, "output", "rec", "svtr26_tiny", "train", "best.onnx")

    print(f"Det Model: {det_model_path}")
    print(f"Rec Model: {rec_model_path}")

    # Collect test images
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    all_files.sort()

    if len(all_files) < num_samples:
        print(f"Warning: Found {len(all_files)} images, which is less than requested {num_samples}.")
        selected_files = all_files
    else:
        selected_files = all_files[:num_samples]

    print(f"Loading {len(selected_files)} images into memory for benchmarking...")
    images = []
    for f in selected_files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        print("No images found! Exiting.")
        return

    print(f"Successfully loaded {len(images)} images.")

    # Initialize LiteALPR
    alpr = LiteALPR(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        device=device
    )

    # Warm-up runs
    print(f"\nWarm-up: executing {num_warmup} warm-up runs...")
    for i in range(num_warmup):
        _ = alpr.read(images[i % len(images)].copy())

    all_pre = []
    all_det_infer = []
    all_post = []
    all_rec_infer = []
    all_total_infer = []
    all_total = []
    all_pass_fps = []

    print(f"\nExecuting {num_runs} benchmark passes over {len(images)} images...")
    for pass_idx in range(1, num_runs + 1):
        pass_pre = []
        pass_det_infer = []
        pass_post = []
        pass_rec_infer = []
        pass_total_infer = []
        pass_total = []

        t_pass_start = time.perf_counter()
        pbar = tqdm(images, desc=f"Pass {pass_idx}/{num_runs}", unit="img", ncols=90)
        for img in pbar:
            t_start = time.perf_counter()

            # 1. Detection Pre-processing (Letterbox / Normalization)
            # Ultralytics preprocess
            device_arg = 0 if 'cuda' in str(alpr.device) else 'cpu'
            # Measure detection including YOLO's internal timing
            det_results = alpr.det_model(img, verbose=False, conf=0.25, imgsz=416, device=device_arg)[0]

            speed = det_results.speed
            t_det_pre = speed.get('preprocess', 0.0)
            t_det_infer = speed.get('inference', 0.0)
            t_det_post = speed.get('postprocess', 0.0)

            boxes = det_results.boxes.data.cpu().numpy()
            h, w, _ = img.shape

            t_rec_pre_total = 0.0
            t_rec_infer_total = 0.0
            t_rec_post_total = 0.0

            # 2. Recognition across detected license plates
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
                x2_c, y2_c = min(w, int(x2)), min(h, int(y2))
                crop_img = img[y1_c:y2_c, x1_c:x2_c]

                if crop_img.size > 0:
                    t_rp0 = time.perf_counter()
                    tensor = alpr._preprocess_crop(crop_img)
                    input_numpy = tensor.detach().cpu().numpy()
                    t_rec_pre_total += (time.perf_counter() - t_rp0) * 1000

                    t_ri0 = time.perf_counter()
                    preds = alpr.rec_session.run(None, {alpr.rec_input_name: input_numpy})[0]
                    t_rec_infer_total += (time.perf_counter() - t_ri0) * 1000

                    t_rpost0 = time.perf_counter()
                    _ = alpr.post_process_class(preds)
                    t_rec_post_total += (time.perf_counter() - t_rpost0) * 1000

            t_total = (time.perf_counter() - t_start) * 1000

            # Breakdown matching paper:
            # Pre = Detection Pre-process + Recognition Crop Pre-process
            # Infer = Detection Infer + Recognition Infer
            # Post = Detection NMS/Post + Recognition CTC Decode Post
            t_pipeline_pre = t_det_pre + t_rec_pre_total
            t_pipeline_infer = t_det_infer + t_rec_infer_total
            t_pipeline_post = t_det_post + t_rec_post_total

            pass_pre.append(t_pipeline_pre)
            pass_det_infer.append(t_det_infer)
            pass_rec_infer.append(t_rec_infer_total)
            pass_total_infer.append(t_pipeline_infer)
            pass_post.append(t_pipeline_post)
            pass_total.append(t_total)

        pass_duration = time.perf_counter() - t_pass_start
        pass_fps = len(images) / pass_duration if pass_duration > 0 else 0.0

        all_pre.extend(pass_pre)
        all_det_infer.extend(pass_det_infer)
        all_rec_infer.extend(pass_rec_infer)
        all_total_infer.extend(pass_total_infer)
        all_post.extend(pass_post)
        all_total.extend(pass_total)
        all_pass_fps.append(pass_fps)

        print(f"Pass {pass_idx}/{num_runs}: Pre={np.mean(pass_pre):.2f}ms | DetInfer={np.mean(pass_det_infer):.2f}ms | RecInfer={np.mean(pass_rec_infer):.2f}ms | Infer={np.mean(pass_total_infer):.2f}ms | Post={np.mean(pass_post):.2f}ms | Total={np.mean(pass_total):.2f}ms | FPS={pass_fps:.2f}")

    # Calculate statistics
    mean_pre, std_pre = float(np.mean(all_pre)), float(np.std(all_pre))
    mean_infer, std_infer = float(np.mean(all_total_infer)), float(np.std(all_total_infer))
    mean_det_infer, std_det_infer = float(np.mean(all_det_infer)), float(np.std(all_det_infer))
    mean_rec_infer, std_rec_infer = float(np.mean(all_rec_infer)), float(np.std(all_rec_infer))
    mean_post, std_post = float(np.mean(all_post)), float(np.std(all_post))
    mean_total, std_total = float(np.mean(all_total)), float(np.std(all_total))
    mean_fps, std_fps = float(np.mean(all_pass_fps)), float(np.std(all_pass_fps))

    print("\n" + "=" * 72)
    print(f" SUMMARY BENCHMARK RESULTS ON {device.upper()} (Mean ± Std across {num_runs} passes x {len(images)} images)")
    print("=" * 72)
    print(f" Pre-processing Latency   : {mean_pre:6.2f} ± {std_pre:4.2f} ms")
    print(f" Detection Inference      : {mean_det_infer:6.2f} ± {std_det_infer:4.2f} ms")
    print(f" Recognition Inference    : {mean_rec_infer:6.2f} ± {std_rec_infer:4.2f} ms")
    print(f" Total Inference Latency  : {mean_infer:6.2f} ± {std_infer:4.2f} ms")
    print(f" Post-processing Latency  : {mean_post:6.2f} ± {std_post:4.2f} ms")
    print("-" * 72)
    print(f" Total Pipeline Latency   : {mean_total:6.2f} ± {std_total:4.2f} ms")
    print(f" Throughput (FPS)         : {mean_fps:6.2f} ± {std_fps:4.2f} FPS")
    print("=" * 72)

    return {
        "pre_ms": f"{mean_pre:.2f} ± {std_pre:.2f}",
        "infer_ms": f"{mean_infer:.2f} ± {std_infer:.2f}",
        "post_ms": f"{mean_post:.2f} ± {std_post:.2f}",
        "total_ms": f"{mean_total:.2f} ± {std_total:.2f}",
        "fps": f"{mean_fps:.2f} ± {std_fps:.2f}"
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiteALPR Benchmark Script")
    parser.add_argument("--images_dir", type=str, default=os.path.join(litealpr_root, "dataset", "det", "train", "images"), help="Path to test images")
    parser.add_argument("--det_model_path", type=str, default=os.path.join(litealpr_root, "pretrained_models", "det", "yolov8n_efficient", "best_416.onnx"), help="Path to the detection model")
    parser.add_argument("--rec_model_path", type=str, default=os.path.join(litealpr_root, "pretrained_models", "rec", "svtr26_tiny", "best.onnx"), help="Path to the recognition model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g. 'cpu' or 'cuda:0')")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of images to sample")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark passes")
    args = parser.parse_args()

    benchmark_pipeline_dataset(args.images_dir, det_model_path=args.det_model_path, rec_model_path=args.rec_model_path, num_samples=args.num_samples, num_runs=args.num_runs, num_warmup=args.warmup, device=args.device)
