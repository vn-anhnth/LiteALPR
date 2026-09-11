import glob

import cv2

from litealpr import LiteALPR


def test_end_to_end():
    print("\n" + "=" * 50)
    print("TEST 1: END-TO-END (DET + REC)")
    print("=" * 50)

    # Load both models (default)
    model = LiteALPR()

    test_images = glob.glob("dataset/wild/*.jpg") + glob.glob("dataset/wild/*.png")
    if not test_images:
        print("No test images found for DET.")
        return

    print(f"[Run] Processing Image: {test_images[3]}")
    results = model.read(test_images[3])

    for i, res in enumerate(results):
        print(
            f"  -> Plate {i + 1}: Text='{res['text']}' | Conf={res['score']:.4f} | Box={res['box']}"
        )


def test_detect_only():
    print("\n" + "=" * 50)
    print("TEST 2: DETECTION ONLY (YOLO ONLY)")
    print("=" * 50)

    # Load ONLY detection model, disable recognition
    model = LiteALPR(use_rec=False)

    test_images = glob.glob("dataset/det/test/images/*.jpg") + glob.glob(
        "dataset/det/test/images/*.png"
    )
    if not test_images:
        print("No test images found for DET.")
        return

    img = cv2.imread(test_images[0])
    print(f"[Run] Detecting plates in: {test_images[0]}")

    boxes = model.detect(img)
    for i, box in enumerate(boxes):
        print(f"  -> Box {i + 1}: {box}")


def test_recognize_only():
    print("\n" + "=" * 50)
    print("TEST 3: RECOGNITION ONLY (SVTR ONLY)")
    print("=" * 50)

    # Load ONLY recognition model, disable detection
    model = LiteALPR(use_det=False)

    # Use an already cropped plate image from the REC dataset
    test_crops = glob.glob("dataset/rec/test/*.*")
    if not test_crops:
        print("No cropped test images found for REC.")
        return

    crop_img = cv2.imread(test_crops[0])
    print(f"[Run] Recognizing text in cropped image: {test_crops[0]}")

    text, score = model.recognize(crop_img)
    print(f"  -> Result: Text='{text}' | Conf={score:.4f}")


def test_custom_models():
    print("\n" + "=" * 50)
    print("TEST 4: USING CUSTOM LOCAL MODELS")
    print("=" * 50)

    # User provides their own local paths instead of auto-downloading
    custom_det = "pretrained_models/det/yolov8n_efficient/best.pt"
    custom_rec = "pretrained_models/rec/svtr26_tiny/best.pth"

    print(f"[Run] Initializing with custom det: {custom_det}")
    print(f"[Run] Initializing with custom rec: {custom_rec}")

    model = LiteALPR(det_model_path=custom_det, rec_model_path=custom_rec)

    test_images = glob.glob("dataset/det/test/images/*.jpg") + glob.glob(
        "dataset/det/test/images/*.png"
    )
    if not test_images:
        print("No test images found for DET.")
        return

    print(f"[Run] Processing Image: {test_images[0]}")
    results = model.read(test_images[0])

    for i, res in enumerate(results):
        print(
            f"  -> Plate {i + 1}: Text='{res['text']}' | Conf={res['score']:.4f} | Box={res['box']}"
        )


if __name__ == "__main__":
    test_end_to_end()
    test_detect_only()
    test_recognize_only()
    test_custom_models()
