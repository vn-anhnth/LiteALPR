"""Audit OCR accuracy on strictly unseen plates using ocr_split_manifest.csv."""

import argparse
import csv
import os
import string

import cv2
import onnxruntime as ort
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torchvision import transforms as T
from torchvision.transforms import functional as TF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ALLOWED_CHARS = set(string.digits + string.ascii_letters + "Đđ")
BASE_SHAPE = [[64, 64], [96, 48], [112, 40], [128, 32]]


def normalize_text(text: str) -> str:
    return "".join(ch for ch in text if ch in ALLOWED_CHARS).lower()


def load_ctc_chars(dict_path: str):
    chars = ["blank"]
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip("\r\n")
            if c:
                chars.append(c)
    return chars


def main():
    parser = argparse.ArgumentParser(description="Audit LiteALPR recognizer on unseen plates from ocr_split_manifest.csv")
    parser.add_argument("--manifest", default=os.path.join(ROOT_DIR, "manifests", "ocr_split_manifest.csv"))
    parser.add_argument("--labels", required=True, help="Path to OCR test_labels.txt (rel_path<TAB>label)")
    parser.add_argument("--dataset_dir", required=True, help="Root directory containing OCR crop images")
    parser.add_argument("--rec_model", default=os.path.join(ROOT_DIR, "pretrained_models", "rec", "svtr26_tiny", "best.onnx"))
    parser.add_argument("--dict_path", default=os.path.join(ROOT_DIR, "litealpr", "license_plate_dict.txt"))
    parser.add_argument("--cpu_threads", type=int, default=6)
    args = parser.parse_args()

    unseen_paths = set()
    with open(args.manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row.get("unseen_plate") == "True" and row["relative_path"] != "REDACTED":
                unseen_paths.add(row["relative_path"].replace("\\", "/"))

    chars = load_ctc_chars(args.dict_path)
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.cpu_threads
    sess = ort.InferenceSession(args.rec_model, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    n = ok = edits = total_ch = 0
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            rel = parts[0].replace("\\", "/")
            if rel not in unseen_paths:
                continue
            img = cv2.imread(os.path.join(args.dataset_dir, rel))
            if img is None:
                continue
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            w, h = pil.size
            ratio = min(max(int(round(float(w) / float(h))), 1), 4)
            imgW, imgH = BASE_SHAPE[ratio - 1]
            t = TF.to_tensor(TF.resize(pil, (imgH, imgW), interpolation=T.InterpolationMode.BICUBIC))
            t.sub_(0.5).div_(0.5)
            idxs = sess.run(None, {in_name: t.unsqueeze(0).numpy()})[0].argmax(axis=2)[0]
            out, prev = [], -1
            for idx in idxs:
                idx = int(idx)
                if idx != 0 and idx != prev and idx < len(chars):
                    out.append(chars[idx])
                prev = idx
            pred_n, gt_n = normalize_text("".join(out)), normalize_text(parts[1])
            n += 1
            if pred_n == gt_n:
                ok += 1
            edits += Levenshtein.distance(gt_n, pred_n)
            total_ch += len(gt_n)

    print(f"Unseen-plate public crops evaluated: {n}")
    if n > 0:
        print(f"Acc: {ok / n * 100:.2f}% | CER: {edits / max(total_ch, 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
