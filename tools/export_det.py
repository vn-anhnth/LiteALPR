import os
import argparse

import torch
import onnx
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 Detection Model to ONNX")
    parser.add_argument("-m", "--model", type=str, default="output/det/yolov8n_efficient/train/weights/best.pt", help="Path to trained model .pt")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size for export")
    return parser.parse_args()


def main():
    args = parse_args()
    opset_version = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Initializing export on device: {device}")
    print(f"[INFO] Loading detection model: {args.model}")

    # Load YOLO model
    yolo = YOLO(args.model)

    model = yolo.model
    model.to(device)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1,3,args.imgsz,args.imgsz,device=device)

    # Output path
    output_dir = os.path.dirname(args.model)
    output_name = f"best_{args.imgsz}.onnx"
    save_path = os.path.join(output_dir, output_name)

    print(f"[INFO] Exporting to ONNX (imgsz={args.imgsz}, opset={opset_version}, FP32)...")
    exported_path = yolo.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=opset_version,
        dynamic=False,
        simplify=False
    )

    # Ultralytics exports as best.onnx; rename to save_path (e.g. best_416.onnx)
    if os.path.exists(save_path):
        os.remove(save_path)
    os.replace(exported_path, save_path)

    print("[INFO] ONNX export successful!")
    print(f"[INFO] ONNX model: {save_path}")


if __name__ == "__main__":
    main()
