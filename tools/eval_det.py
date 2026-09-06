import argparse
import os
import sys

import torch
from ultralytics import YOLO

# Add project root to path so we can import 'det'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='output/det/yolov8n_efficient/train/weights/best.pt', help='path to the pretrained model (overrides config)')
    return parser.parse_args()


def main():
    args = parse_args()

    print('[INFO] Starting evaluation.')
    model = YOLO(args.model)
    project_path = 'output/det/yolov8n_efficient'
    metrics = model.val(
        data='dataset/det/data.yaml',
        split='val',
        imgsz=640,
        batch=32,
        device='0' if torch.cuda.is_available() else 'cpu',
        verbose=False,
        project=project_path
    )

    print('\n[INFO] ========================================')
    print('[INFO] Evaluation completed!')
    print(f'[INFO] Precision  : {metrics.box.mp:.4f}')
    print(f'[INFO] Recall     : {metrics.box.mr:.4f}')
    print(f'[INFO] mAP50      : {metrics.box.map50:.4f}')
    print(f'[INFO] mAP50-95   : {metrics.box.map:.4f}')
    print(f'[INFO] Results saved to: {os.path.join(project_path, "eval")}')
    print('[INFO] ========================================')


if __name__ == '__main__':
    # python tools/eval_det.py
    # python tools/eval_det.py -m output/det/yolov8n_efficient/train/weights/best.pt
    main()
