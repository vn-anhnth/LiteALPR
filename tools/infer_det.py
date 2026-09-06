import argparse
import glob
import os
import sys
import time

import cv2
import torch

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='pretrained_models/det/yolov8n_efficient/best.pt', help='path to model.pt')
    parser.add_argument('-d', '--dir', type=str, default='dataset/det/test/images', help='Image directory')
    parser.add_argument('--save_log', action='store_true', help='Save inference predictions to log file')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing inference on device: {str(device).upper()}")

    pretrained_model = args.model
    print(f"[INFO] Loading model: {pretrained_model}")
    model = YOLO(pretrained_model)
    model.to(device)

    # Discover images
    image_paths = glob.glob(os.path.join(args.dir, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(args.dir, '*.[pP][nN][gG]'))

    if not image_paths:
        print(f"[ERROR] No images found in {args.dir}")
        return

    print(f"[INFO] Found {len(image_paths)} images in {args.dir}.")
    print("[INFO] Pre-loading images into memory...")

    images = []
    valid_paths = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(img)
            valid_paths.append(p)

    if not images:
        return

    # Warmup
    print("[INFO] Starting warmup (10 iterations)...")
    for i in range(min(10, len(images))):
        _ = model(images[i], verbose=False)

    print("[INFO] Warmup completed. Measuring FPS...")

    # Measure Inference Time
    total_time = 0.0
    total_samples = len(images)
    results_list = []

    for idx, img in enumerate(images):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # Inference
        res = model(img, verbose=False)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        total_time += (end_time - start_time)

        if args.save_log:
            # Format predictions: class, conf, bbox
            preds = []
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                # name = model.names[cls_id]
                preds.append(f"[{cls_id}, {conf:.4f}, {xyxy}]")
            results_list.append((valid_paths[idx], " | ".join(preds)))

    avg_time_ms = (total_time / total_samples) * 1000
    fps = total_samples / total_time

    summary_lines = [
        "[INFO] ========================================",
        "[INFO] Inference Test",
        f"[INFO] Device        : {str(device).upper()}",
        f"[INFO] Total Images  : {total_samples}",
        f"[INFO] Avg Time/Img  : {avg_time_ms:.2f} ms",
        f"[INFO] FPS           : {fps:.2f} frames/sec"
    ]

    if args.save_log:
        # Determine infer output dir dynamically based on model's folder name
        model_name = os.path.basename(os.path.dirname(args.model))
        if not model_name or model_name == 'weights':  # if it's nested in a weights folder
            model_name = 'yolov8n_efficient'

        infer_dir = os.path.join('output', 'det', model_name, 'infer')
        os.makedirs(infer_dir, exist_ok=True)
        log_path = os.path.join(infer_dir, 'infer.log')
        summary_lines.append(f"[INFO] Log saved     : {log_path}")

    summary_lines.append("[INFO] ========================================")

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}\n")

    if args.save_log:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(summary_text + "\n\n")
            f.write("FileName\tPredictions [Class, Conf, BBox]\n")
            f.write("----------------------------------------\n")
            for path, preds in results_list:
                f.write(f"{os.path.basename(path)}\t{preds}\n")

if __name__ == '__main__':
    # python tools/infer_det.py
    # python tools/infer_det.py --save_log
    # python tools/infer_det.py -m pretrained_models/det/yolov8n_efficient/best.pt -d dataset/det/test/images --save_log
    main()
