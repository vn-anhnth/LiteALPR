import argparse
import os
import sys

import torch
import onnx

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from litealpr.rec.modeling import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Export SVTR Recognition Model to ONNX")
    parser.add_argument('-m', '--model', type=str, default='output/rec/svtr26_tiny/train/best.pth', help='Path to model .pth')
    parser.add_argument('--save_path', type=str, default='output/rec/svtr26_tiny/train/best.onnx', help='Path to save onnx')
    parser.add_argument('--imgH', type=int, default=32, help='Image height')
    parser.add_argument('--imgW', type=int, default=128, help='Image width')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic width axes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    opset_version = 12
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Initializing export on device: {device.type.upper()}")

    # Load checkpoint
    print(f"[INFO] Loading model checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    if 'config' not in checkpoint:
        print("[ERROR] No config found in checkpoint! Please provide a valid LiteALPR checkpoint.")
        sys.exit(1)

    config = checkpoint['config']
    model = build_model(config['Architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, args.imgH, args.imgW, device=device)

    # Output path
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 1: 'width'}}

    # Export using legacy PyTorch ONNX exporter
    print(f"[INFO] Exporting to ONNX (input={args.imgH}x{args.imgW}, opset={opset_version}, FP32, dynamic={args.dynamic})...")
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )

    # Verify ONNX
    print("[INFO] Checking ONNX model...")
    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)
    actual_opset = model_onnx.opset_import[0].version

    print("[INFO] ONNX export successful!")
    print(f"[INFO] ONNX opset: {actual_opset}")
    print(f"[INFO] ONNX model: {save_path}")


if __name__ == '__main__':
    main()
