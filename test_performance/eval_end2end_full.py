import argparse
import os
import string
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torchvision import ops as tv_ops
from torchvision import transforms as T
from torchvision.transforms import functional as TF

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALLOWED_CHARS = set(string.digits + string.ascii_letters + "Đđ")
BASE_SHAPE = [[64, 64], [96, 48], [112, 40], [128, 32]]


def normalize_text(text: str) -> str:
    return "".join(ch for ch in text if ch in ALLOWED_CHARS).lower()


def load_ctc_chars(dict_path: str):
    chars = ["blank"]
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip("\n").strip("\r\n")
            if c:
                chars.append(c)
    return chars


def ctc_greedy_decode(probs: np.ndarray, chars):
    idxs = probs.argmax(axis=2)[0]
    out, prev = [], -1
    for idx in idxs:
        idx = int(idx)
        if idx != 0 and idx != prev and idx < len(chars):
            out.append(chars[idx])
        prev = idx
    return "".join(out)


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def letterbox(img, new_shape=416):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_shape - new_unpad[0]) / 2.0, (new_shape - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, left, top


def detect_onnx(sess, input_name, img_bgr, imgsz=416, conf_thresh=0.25, nms_iou=0.70):
    h0, w0 = img_bgr.shape[:2]
    lb, r, left, top = letterbox(img_bgr, imgsz)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    inp = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    out = sess.run(None, {input_name: inp[None, ...]})[0]
    preds = out[0].T
    scores = preds[:, 4]
    mask = scores >= conf_thresh
    if not np.any(mask):
        return [], []
    preds, scores = preds[mask], scores[mask]
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack(
        [(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r, (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r],
        axis=1,
    )
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)
    keep = tv_ops.nms(
        torch.from_numpy(boxes.astype(np.float32)),
        torch.from_numpy(scores.astype(np.float32)),
        nms_iou,
    ).numpy()
    return boxes[keep].tolist(), scores[keep].tolist()


def preprocess_crop(img_bgr, max_ratio=4, base_h=32, fixed_hw=None):
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if fixed_hw is not None:
        imgH, imgW = fixed_hw
    else:
        w, h = img.size
        ratio = min(max(int(round(float(w) / float(h))), 1), max_ratio)
        imgW, imgH = BASE_SHAPE[ratio - 1]
    resized = TF.resize(img, (imgH, imgW), interpolation=T.InterpolationMode.BICUBIC)
    t = TF.to_tensor(resized)
    t.sub_(0.5).div_(0.5)
    return t.unsqueeze(0).numpy()


def load_gt_boxes(lbl_path, w, h):
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = (xc - bw / 2.0) * w
                y1 = (yc - bh / 2.0) * h
                x2 = (xc + bw / 2.0) * w
                y2 = (yc + bh / 2.0) * h
                boxes.append([x1, y1, x2, y2])
            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                coords = list(map(float, parts[1:]))
                xs = [c * w for c in coords[0::2]]
                ys = [c * h for c in coords[1::2]]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(ROOT_DIR, "dataset", "e2e_holdout"),
    )
    parser.add_argument(
        "--det_model",
        default=os.path.join(ROOT_DIR, "pretrained_models", "det", "yolov8n_efficient", "best_416.onnx"),
    )
    parser.add_argument(
        "--rec_model",
        default=os.path.join(ROOT_DIR, "pretrained_models", "rec", "svtr26_tiny", "best.onnx"),
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--iou_thresh", type=float, default=0.50)
    args = parser.parse_args()

    img_dir = os.path.join(args.dataset_dir, "images")
    lbl_dir = os.path.join(args.dataset_dir, "labels")
    ocr_txt = os.path.join(args.dataset_dir, "ocr_labels.txt")

    samples = []
    with open(ocr_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                src = parts[2] if len(parts) >= 3 else "unknown"
                samples.append((parts[0], parts[1], src))
    print(f"Loaded {len(samples)} full-scene test samples from {args.dataset_dir}", flush=True)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 6
    det_sess = ort.InferenceSession(args.det_model, sess_options=so, providers=["CPUExecutionProvider"])
    det_in = det_sess.get_inputs()[0].name
    rec_sess = ort.InferenceSession(args.rec_model, sess_options=so, providers=["CPUExecutionProvider"])
    rec_in = rec_sess.get_inputs()[0].name
    rec_shape = rec_sess.get_inputs()[0].shape
    fixed_hw = (int(rec_shape[2]), int(rec_shape[3])) if (len(rec_shape) == 4 and isinstance(rec_shape[2], int) and isinstance(rec_shape[3], int)) else None
    print(f"Rec input shape: {rec_shape} -> fixed_hw={fixed_hw}", flush=True)
    chars = load_ctc_chars(os.path.join(ROOT_DIR, "litealpr", "license_plate_dict.txt"))

    total = len(samples)
    det_matched = 0
    missed_no_det = 0
    missed_low_iou = 0
    total_preds = 0
    total_fp = 0
    multi_det_images = 0

    e2e_exact_matched = 0
    e2e_exact_top1 = 0
    e2e_exact_strict = 0
    total_edits_all, total_chars_all = 0, 0
    total_edits_det, total_chars_det = 0, 0

    per_src = {}
    t0 = time.time()

    for i, (fn, gt_text, src) in enumerate(samples):
        if src not in per_src:
            per_src[src] = {"n": 0, "det": 0, "e2e": 0, "fp": 0, "edits": 0, "chars": 0}
        per_src[src]["n"] += 1

        img = cv2.imread(os.path.join(img_dir, fn))
        h, w = img.shape[:2]
        gt_boxes = load_gt_boxes(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"), w, h)
        gt_box = gt_boxes[0] if gt_boxes else [0, 0, w, h]

        boxes_xyxy, _ = detect_onnx(det_sess, det_in, img, imgsz=args.imgsz, conf_thresh=args.conf_thresh)
        n_pred = len(boxes_xyxy)
        total_preds += n_pred
        if n_pred > 1:
            multi_det_images += 1

        gt_norm = normalize_text(gt_text)
        gt_len = len(gt_norm)
        total_chars_all += gt_len
        per_src[src]["chars"] += gt_len

        if n_pred == 0:
            missed_no_det += 1
            total_edits_all += gt_len
            per_src[src]["edits"] += gt_len
            continue

        # Match against any GT box (for single-plate images, gt_boxes has 1 box)
        ious = [max(compute_iou(pb, gb) for gb in gt_boxes) for pb in boxes_xyxy]
        best_idx = int(np.argmax(ious))
        best_iou = ious[best_idx]

        if best_iou >= args.iou_thresh:
            det_matched += 1
            per_src[src]["det"] += 1
            fp_here = max(0, n_pred - len(gt_boxes))
            total_fp += fp_here
            per_src[src]["fp"] += fp_here

            x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[best_idx]]
            x1_c, y1_c, x2_c, y2_c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2_c > x1_c and y2_c > y1_c:
                inp = preprocess_crop(img[y1_c:y2_c, x1_c:x2_c], fixed_hw=fixed_hw)
                probs = rec_sess.run(None, {rec_in: inp})[0]
                pred_norm = normalize_text(ctc_greedy_decode(probs, chars))
            else:
                pred_norm = ""

            dist = Levenshtein.distance(gt_norm, pred_norm)
            total_edits_det += dist
            total_chars_det += gt_len
            total_edits_all += dist
            per_src[src]["edits"] += dist

            if pred_norm == gt_norm:
                e2e_exact_matched += 1
                per_src[src]["e2e"] += 1
                if best_idx == 0:
                    e2e_exact_top1 += 1
                if fp_here == 0:
                    e2e_exact_strict += 1
        else:
            missed_low_iou += 1
            total_fp += n_pred
            per_src[src]["fp"] += n_pred
            total_edits_all += gt_len
            per_src[src]["edits"] += gt_len

        if (i + 1) % 1000 == 0 or (i + 1) == total:
            dt = time.time() - t0
            print(
                f"[{i+1}/{total}] elapsed={dt:.1f}s | det_recall={det_matched/(i+1)*100:.2f}% | "
                f"e2e_exact={e2e_exact_matched/(i+1)*100:.2f}% | fp={total_fp}",
                flush=True,
            )

    dt = time.time() - t0
    print("\n================ END-TO-END RESULTS (5,230 Full-Scene Holdout) ================")
    print(f"Detector: {os.path.basename(args.det_model)} (imgsz={args.imgsz}) | Recognizer: {os.path.basename(args.rec_model)}")
    print(f"Total full-scene test images:       {total}")
    print(f"Detection Recall (IoU >= {args.iou_thresh:.2f}):     {det_matched}/{total} = {det_matched/total*100:.2f}%")
    print(f"  Missed (no box):                  {missed_no_det} ({missed_no_det/total*100:.2f}%)")
    print(f"  Missed (IoU < {args.iou_thresh:.2f}):              {missed_low_iou} ({missed_low_iou/total*100:.2f}%)")
    print(f"Total predicted boxes:              {total_preds} | False Positives: {total_fp} ({total_fp/total*100:.2f}% per img)")
    print(f"Detection Precision:                {det_matched/total_preds*100:.2f}%")
    print(f"Images with >1 detected box:        {multi_det_images} ({multi_det_images/total*100:.2f}%)")
    print(f"End-to-End Exact Match (IoU>=0.5):  {e2e_exact_matched}/{total} = {e2e_exact_matched/total*100:.2f}%")
    print(f"End-to-End Exact Match (Top-1 box): {e2e_exact_top1}/{total} = {e2e_exact_top1/total*100:.2f}%")
    print(f"End-to-End Strict (no FP):          {e2e_exact_strict}/{total} = {e2e_exact_strict/total*100:.2f}%")
    print(f"OCR Acc on Detected Crops:          {e2e_exact_matched}/{det_matched} = {e2e_exact_matched/det_matched*100:.2f}%")
    print(f"End-to-End CER (all imgs, miss=del):{total_edits_all/total_chars_all*100:.2f}%")
    print(f"OCR CER on Detected Crops:          {total_edits_det/total_chars_det*100:.2f}%")
    print(f"Total time: {dt:.1f}s ({total/dt:.1f} img/s)")
    print("\nPer-source breakdown:")
    for src, st in sorted(per_src.items()):
        print(
            f"  {src:14s} (n={st['n']:4d}): DetRec={st['det']/st['n']*100:6.2f}% | "
            f"E2E={st['e2e']/st['n']*100:6.2f}% | CER={st['edits']/st['chars']*100:5.2f}% | FP={st['fp']}"
        )


if __name__ == "__main__":
    main()
