import argparse
import csv
import logging
import os
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format='%(asctime)s INFO: %(message)s', datefmt='%H:%M:%S')

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.memory_allocated = False

    def allocate_memory(self, input_shape):
        self.context.set_binding_shape(0, input_shape)
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            shape = self.context.get_binding_shape(idx)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
        self.memory_allocated = True

    def infer(self, input_data):
        if not self.memory_allocated or tuple(self.inputs[0]['shape']) != input_data.shape:
            if self.memory_allocated:
                for inp in self.inputs:
                    inp['device'].free()
                for out in self.outputs:
                    out['device'].free()
                self.inputs = []
                self.outputs = []
                self.bindings = []
            self.allocate_memory(input_data.shape)

        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'].reshape(out['shape']) for out in self.outputs]

    def destroy(self):
        try:
            self.stream.synchronize()
            for inp in self.inputs:
                inp['device'].free()
            for out in self.outputs:
                out['device'].free()
        except Exception:
            pass

def preprocess_yolo(img, img_size=416):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_float = img_resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
    return np.expand_dims(img_float, axis=0)

def preprocess_svtr(crops):
    base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]
    base_h = 32
    max_ratio = 4
    tensors = []
    for img in crops:
        h, w = img.shape[:2]
        gen_ratio = max(1, int(np.round(float(w) / float(h))))
        ratio_resize = min(gen_ratio, max_ratio)
        if ratio_resize <= 4:
            imgW, imgH = base_shape[ratio_resize - 1]
        else:
            imgW, imgH = [base_h * ratio_resize, base_h]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
        img_float = img_resized.astype(np.float32).transpose((2, 0, 1)) / 255.0
        img_float = (img_float - 0.5) / 0.5
        tensors.append(img_float)
    if tensors:
        return np.stack(tensors, axis=0)
    return None

def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def postprocess_yolo(output, orig_shape, conf_thres=0.25, det_size=416):
    out = output[0][0].T
    scores = out[:, 4]
    mask = scores > conf_thres
    out = out[mask]
    scores = scores[mask]
    if len(out) == 0:
        return []
    boxes = out[:, :4]
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes[:, 0] = x_c - w/2
    boxes[:, 1] = y_c - h/2
    boxes[:, 2] = x_c + w/2
    boxes[:, 3] = y_c + h/2
    keep = nms(boxes, scores)
    final_boxes = boxes[keep]
    h_orig, w_orig = orig_shape[:2]
    scale_x, scale_y = w_orig / float(det_size), h_orig / float(det_size)
    final_boxes[:, [0, 2]] *= scale_x
    final_boxes[:, [1, 3]] *= scale_y
    return final_boxes.astype(int)


def greedy_decode(preds, char_list, blank_idx=0):
    preds_idx = np.argmax(preds, axis=2)[0]
    text = ""
    for i in range(len(preds_idx)):
        idx = preds_idx[i]
        if idx != blank_idx and (not (i > 0 and idx == preds_idx[i - 1])):
            text += char_list[idx]
    return text

def get_jetson_temp():

    try:
        with open('/sys/devices/virtual/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000.0
        return temp
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="LiteALPR Jetson Nano Stress Test Script")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to test images")
    parser.add_argument("--det_model_path", type=str, required=True, help="Path to the detection TensorRT engine (.engine)")
    parser.add_argument("--rec_model_path", type=str, required=True, help="Path to the recognition TensorRT engine (.engine)")
    parser.add_argument("--dict_path", type=str, default="dataset/license_plates_ocr/license_plate_dict.txt", help="Path to the character dictionary")
    parser.add_argument("--duration", type=int, default=30, help="Stress test duration in minutes")
    parser.add_argument("--output_csv", type=str, default="run_stress_test_results.csv", help="Output CSV file path")
    args = parser.parse_args()

    logging.info(f"Starting Stress Test for {args.duration} minutes...")

    with open(args.dict_path, 'r', encoding='utf-8') as f:
        char_list = [line.strip() for line in f.readlines()]
    char_list = ['blank'] + char_list

    image_files = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:50]
    images = [cv2.imread(f) for f in image_files]
    if not images:
        logging.error("No images found in dataset!")
        return

    # Initialize models
    detector = TRTWrapper(args.det_model_path)
    ocr_model = TRTWrapper(args.rec_model_path)
    det_size = 416

    # Warmup
    logging.info("Warming up...")
    for img in images[:10]:
        inp = preprocess_yolo(img, img_size=det_size)
        det_out = detector.infer(inp)
        boxes = postprocess_yolo(det_out, img.shape, det_size=det_size)

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crops.append(img[y1:y2, x1:x2])

        if crops:
            inp_ocr = preprocess_svtr([crops[0]])
            if inp_ocr is not None:
                ocr_model.infer(inp_ocr)

    logging.info("Stress Test Started!")

    with open(args.output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time (min)', 'FPS', 'Latency (ms)', 'Temp (C)', 'Plate'])

        start_time = time.time()
        end_time = start_time + args.duration * 60

        last_log_time = start_time
        frames_since_last_log = 0
        img_idx = 0

        while time.time() < end_time:
            img = images[img_idx % len(images)]
            img_idx += 1

            # Inference pipeline
            inp = preprocess_yolo(img, img_size=det_size)
            det_out = detector.infer(inp)
            boxes = postprocess_yolo(det_out, img.shape, det_size=det_size)

            crops = []
            for box in boxes:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    crops.append(img[y1:y2, x1:x2])

            last_plate_text = ""
            if crops:
                inp_ocr = preprocess_svtr([crops[0]])
                if inp_ocr is not None:
                    rec_out = ocr_model.infer(inp_ocr)
                    last_plate_text = greedy_decode(rec_out[0], char_list)

            frames_since_last_log += 1
            current_time = time.time()

            # Log every 10 seconds
            if current_time - last_log_time >= 10.0:
                elapsed_min = (current_time - start_time) / 60.0
                fps = frames_since_last_log / (current_time - last_log_time)
                latency = 1000.0 / fps if fps > 0 else 0.0
                temp = get_jetson_temp()

                writer.writerow([f"{elapsed_min:.2f}", f"{fps:.2f}", f"{latency:.2f}", f"{temp:.1f}", last_plate_text])
                f.flush()

                logging.info(f"[{elapsed_min:.2f} min] FPS: {fps:.2f} | Latency: {latency:.2f} ms | Temp: {temp:.1f} C | Plate: {last_plate_text}")

                last_log_time = current_time
                frames_since_last_log = 0

    logging.info(f"Stress test completed! Data saved to {args.output_csv}")
    detector.destroy()
    ocr_model.destroy()

if __name__ == '__main__':
    main()
