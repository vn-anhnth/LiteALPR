import argparse
import os

import yaml
from ultralytics import YOLO, settings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
settings.update({"datasets_dir": project_root})


def parse_args():
    parser = argparse.ArgumentParser(description="LiteALPR Detection Training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/det/yolov8/yolov8n_efficient.yml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("Global", cfg.get("Train", {}))

    model = YOLO(config_path)

    pretrained = train_cfg.pop("pretrained_model", None)
    if pretrained and os.path.isfile(pretrained):
        model = model.load(pretrained)

    if "data" in train_cfg and not os.path.isabs(train_cfg["data"]):
        train_cfg["data"] = os.path.abspath(train_cfg["data"])

    model.train(**train_cfg)


if __name__ == "__main__":
    # python tools/train_det.py
    # python tools/train_det.py -c configs/det/yolov8/yolov8n_efficient.yml
    main()
