import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import argparse

from litealpr.rec.modeling import build_model
from litealpr.rec.postprocess import build_post_process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pretrained_models/rec/svtr26_tiny/best.pth', help='Path to model .pth')
    parser.add_argument('-d', '--dir', type=str, default='dataset/rec/test', help='Image directory')
    parser.add_argument('--save_log', action='store_true', help='Save inference predictions to log file')
    args = parser.parse_args()
    return args


def preprocess_image(image_path, target_h=32, target_w=128):
    """
    Barebones inference preprocessor to mimic SVTR transforms.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Resize
    img = cv2.resize(img, (target_w, target_h))
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize 0-1
    img = img.astype(np.float32) / 255.0
    # Standard Normalize (x - 0.5) / 0.5
    img -= 0.5
    img /= 0.5
    # HWC to CHW
    img = img.transpose((2, 0, 1))
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img)

def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing inference on device: {device.type.upper()}")

    print(f"[INFO] Loading model checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    if 'config' not in checkpoint:
        print("[ERROR] No config found in checkpoint! Please ensure the .pth was saved by LiteALPR trainer.")
        return

    cfg = checkpoint['config']
    cfg['Global']['character_dict_path'] = './tools/utils/license_plate_dict.txt'

    # Build post_process to get character numbers (out_channels)
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = post_process_class.get_character_num()

    # Build and load model
    model = build_model(cfg['Architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device)
    model.eval()

    # Discover images
    image_paths = glob.glob(os.path.join(args.dir, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(args.dir, '*.[pP][nN][gG]'))

    if not image_paths:
        print(f"[ERROR] No images found in {args.dir}")
        return

    print(f"[INFO] Found {len(image_paths)} images in {args.dir}.")
    print("[INFO] Pre-loading images into tensors to isolate pure model inference time...")

    # Pre-load tensors to strictly measure model performance without I/O bottleneck
    tensors = []
    for p in image_paths:
        t = preprocess_image(p)
        if t is not None:
            tensors.append(t.to(device))

    if not tensors:
        return

    # Warmup
    print("[INFO] Starting warmup (10 iterations)...")
    with torch.no_grad():
        for i in range(min(10, len(tensors))):
            _ = model(tensors[i])

    print("[INFO] Warmup completed. Measuring FPS...")
    # Measure Inference Time
    total_time = 0.0
    total_samples = len(tensors)

    results = []

    with torch.no_grad():
        for idx, tensor in enumerate(tensors):
            # Sync GPU before measuring time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # Pure model inference
            preds = model(tensor)

            # Sync GPU after measuring time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time)

            if args.save_log:
                post_result = post_process_class(preds)
                # post_result[0] contains (text, score) for the first item in batch
                text, score = post_result[0]
                results.append((image_paths[idx], text, score))

    avg_time_ms = (total_time / total_samples) * 1000
    fps = total_samples / total_time

    summary_lines = [
        "[INFO] ========================================",
        "[INFO] Inference Test",
        f"[INFO] Device        : {device.type.upper()}",
        f"[INFO] Total Images  : {total_samples}",
        f"[INFO] Avg Time/Img  : {avg_time_ms:.2f} ms",
        f"[INFO] FPS           : {fps:.2f} frames/sec"
    ]

    if args.save_log:
        # Determine infer output dir dynamically based on model's folder name
        model_name = os.path.basename(os.path.dirname(args.model))
        if not model_name or model_name == 'weights':
            model_name = 'svtr26_tiny'

        infer_dir = os.path.join('output', 'rec', model_name, 'infer')
        os.makedirs(infer_dir, exist_ok=True)
        log_path = os.path.join(infer_dir, 'infer.log')
        summary_lines.append(f"[INFO] Log saved     : {log_path}")

    summary_lines.append("[INFO] ========================================")

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}\n")

    if args.save_log:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(summary_text + "\n\n")
            f.write("FileName\tPrediction\tConfidence\n")
            f.write("----------------------------------------\n")
            for path, text, score in results:
                f.write(f"{os.path.basename(path)}\t{text}\t{score:.4f}\n")

if __name__ == '__main__':
    # python tools/infer_rec.py
    # python tools/infer_rec.py --save_log
    # python tools/infer_rec.py -m pretrained_models/rec/svtr26_tiny/best.pth -d dataset/rec/test --save_log
    main()
