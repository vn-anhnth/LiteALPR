import os
import sys
import warnings

import cv2
import torch

warnings.filterwarnings("ignore")

# This ensures that we can import det and rec if running from this file
__dir__ = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(__dir__, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ultralytics import YOLO

from litealpr.rec.modeling import build_model
from litealpr.rec.postprocess import build_post_process


def download_from_hf(filename):
    try:
        from huggingface_hub import hf_hub_download

        print(f"[LiteALPR] Downloading/Verifying {filename} from HuggingFace...")
        return hf_hub_download(repo_id="anhone3/LiteALPR", filename=filename)
    except ImportError:
        raise ImportError(
            "Please install huggingface_hub to auto-download models: pip install huggingface_hub"
        )


class LiteALPR:
    def __init__(
        self, use_det=True, use_rec=True, det_model_path=None, rec_model_path=None, device=None
    ):

        self.device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        print(f"[LiteALPR] Initializing on {self.device}...")

        self.det_model = None
        self.rec_model = None
        self.rec_session = None
        self.rec_input_name = None
        self.post_process_class = None

        # 1. Initialize DET (YOLO ONNX by default)
        if use_det:
            if det_model_path is None:
                det_model_path = download_from_hf("yolov8n_efficient/best_416.onnx")

            print(f"[LiteALPR] Loading Detection Model: {det_model_path}")
            if str(det_model_path).endswith(".onnx"):
                self.det_model = YOLO(det_model_path, task="detect")
            else:
                self.det_model = YOLO(det_model_path)
                self.det_model.to(self.device)
                self.det_model.fuse()

        # 2. Initialize REC (SVTR26 ONNX by default)
        if use_rec:
            if rec_model_path is None:
                rec_model_path = download_from_hf("svtr26_tiny/best.onnx")

            print(f"[LiteALPR] Loading Recognition Model: {rec_model_path}")
            dict_path = os.path.join(os.path.dirname(__file__), "license_plate_dict.txt")

            if str(rec_model_path).endswith(".onnx"):
                import onnxruntime as ort

                from litealpr.rec.postprocess.ctc_postprocess import CTCLabelDecode

                self.post_process_class = CTCLabelDecode(
                    character_dict_path=dict_path, use_space_char=False
                )

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "cuda" in str(self.device)
                    else ["CPUExecutionProvider"]
                )
                self.rec_session = ort.InferenceSession(str(rec_model_path), providers=providers)
                self.rec_input_name = self.rec_session.get_inputs()[0].name
                self.rec_model = True  # Flag to indicate model is loaded
            else:
                checkpoint = torch.load(rec_model_path, map_location="cpu")
                cfg = checkpoint["config"]
                cfg["Global"]["character_dict_path"] = dict_path

                self.post_process_class = build_post_process(cfg["PostProcess"], cfg["Global"])
                cfg["Architecture"]["Decoder"]["out_channels"] = (
                    self.post_process_class.get_character_num()
                )

                self.rec_model = build_model(cfg["Architecture"])
                self.rec_model.load_state_dict(checkpoint["state_dict"], strict=True)
                self.rec_model.to(self.device)
                self.rec_model.eval()

        if self.det_model or self.rec_model:
            print("[LiteALPR] Models loaded successfully!")
        else:
            print(
                "[LiteALPR] WARNING: No models loaded. Please provide det_model_path or rec_model_path."
            )

    def _preprocess_crop(self, img_crop, max_ratio=12, base_shape=None, base_h=32):
        """Preprocesses cropped image for SVTR with RatioRecTVResize logic using PIL (to match training)."""
        from PIL import Image
        from torchvision import transforms as T
        from torchvision.transforms import functional as F

        # Convert OpenCV BGR to PIL RGB
        if base_shape is None:
            base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]
        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        w, h = img.size
        gen_ratio = int(float(w) / float(h)) + 1
        ratio_resize = min(gen_ratio, max_ratio)

        if ratio_resize <= 4:
            imgW, imgH = base_shape[ratio_resize - 1]
        else:
            imgW, imgH = [base_h * ratio_resize, base_h]

        # SVTR is sensitive to interpolation algorithms; we MUST use PIL BICUBIC like during training
        resized_image = F.resize(img, (imgH, imgW), interpolation=T.InterpolationMode.BICUBIC)

        transforms = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

        tensor = transforms(resized_image)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def detect(self, img, conf_thresh=0.25):
        """
        Detect license plates in an image.
        Returns: list of bounding boxes [[x1, y1, x2, y2], ...]
        """
        if self.det_model is None:
            raise ValueError("Detection model not loaded! Initialize with det_model_path.")

        if isinstance(img, str):
            img = cv2.imread(img)

        device_arg = 0 if "cuda" in str(self.device) else "cpu"
        det_results = self.det_model(img, verbose=False, conf=conf_thresh, device=device_arg)[0]
        boxes = det_results.boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        results = []
        for box in boxes:
            x1, y1, x2, y2, _conf, _cls = box
            results.append([int(x1), int(y1), int(x2), int(y2)])
        return results

    def recognize(self, crop_img):
        """
        Recognize text in a cropped license plate image.
        Returns: tuple (text, confidence_score)
        """
        if self.rec_model is None:
            raise ValueError("Recognition model not loaded! Initialize with rec_model_path.")

        if isinstance(crop_img, str):
            crop_img = cv2.imread(crop_img)
            if crop_img is None:
                raise ValueError(f"Could not read image: {crop_img}")

        if crop_img.size == 0:
            return "", 0.0

        tensor = self._preprocess_crop(crop_img)

        if self.rec_session is not None:
            # ONNX Runtime Inference
            input_numpy = tensor.detach().cpu().numpy()
            preds = self.rec_session.run(None, {self.rec_input_name: input_numpy})[0]
        else:
            # PyTorch Model Inference
            with torch.no_grad():
                preds = self.rec_model(tensor)

        post_result = self.post_process_class(preds)
        text, score = post_result[0]
        return text, float(score)

    def read(self, image_path, conf_thresh=0.25):
        """
        End-to-End inference.
        Returns: list of dicts [{'box': [x1,y1,x2,y2], 'text': '51F1234', 'score': 0.99}]
        """
        if self.det_model is None or self.rec_model is None:
            raise ValueError(
                "End-to-End read() requires BOTH det_model_path and rec_model_path to be loaded."
            )

        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            img = image_path

        boxes = self.detect(img, conf_thresh)
        final_results = []

        h, w, _ = img.shape
        for box in boxes:
            x1, y1, x2, y2 = box

            # Ensure bounds
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)

            crop_img = img[y1_c:y2_c, x1_c:x2_c]

            text, score = self.recognize(crop_img)

            final_results.append({"box": box, "text": text, "score": score})

        return final_results
