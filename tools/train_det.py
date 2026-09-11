import argparse
import os

from ultralytics import YOLO, settings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
settings.update({"datasets_dir": project_root})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/det/yolov8/yolov8n_efficient.yml",
        help="configuration file to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.config).load("pretrained_models/det/yolov8n_efficient/best.pt")
    model.train(
        data=os.path.abspath("dataset/det/data.yaml"),
        epochs=50,
        imgsz=640,
        batch=256,
        device=0,  # device=[0, 1] for 2 GPUs
        project="output/det/yolov8n_efficient",
        workers=8,
    )


if __name__ == "__main__":
    # python tools/train_det.py
    # python tools/train_det.py -c configs/det/yolov8/yolov8n_efficient.yaml
    main()
