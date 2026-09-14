import argparse
import logging
import os
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

logging.basicConfig(level=logging.INFO, format='%(asctime)s INFO: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def push_range(msg):
    pass

def pop_range():
    pass

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

        t0 = time.perf_counter()
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        h2d_time = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()
        infer_time = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        d2h_time = (time.perf_counter() - t2) * 1000.0

        outputs = [out['host'].reshape(out['shape']) for out in self.outputs]
        return outputs, h2d_time, d2h_time, infer_time


    def infer_fast(self, input_data):
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

def main():
    parser = argparse.ArgumentParser(description="LiteALPR Jetson Nano TensorRT Benchmark Script")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to test images")
    parser.add_argument("--det_model_path", type=str, required=True, help="Path to the detection TensorRT engine (.engine)")
    parser.add_argument("--rec_model_path", type=str, required=True, help="Path to the recognition TensorRT engine (.engine)")
    args = parser.parse_args()

    image_files = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        logging.error("No images found in dataset!")
        return

    images_1k = [cv2.imread(f) for f in image_files[:1000]]

    results = []

    scenario_name = "End-to-End Jetson Nano TensorRT Pipeline"
    logging.info("="*60)
    logging.info(f"Evaluating: {scenario_name}")

    if not os.path.exists(args.det_model_path) or not os.path.exists(args.rec_model_path):
        logging.error("Engine missing!")
        return

    detector = TRTWrapper(args.det_model_path)
    ocr_model = TRTWrapper(args.rec_model_path)

    det_size = 416

    logging.info("Warming up (50 iterations)...")
    for i in range(50):
        img = images_1k[i]
        inp = preprocess_yolo(img, img_size=det_size)
        det_out, _, _, _ = detector.infer(inp)
        boxes = postprocess_yolo(det_out, img.shape, det_size=det_size)
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                inp_ocr = preprocess_svtr([crop])
                ocr_model.infer(inp_ocr)

    metrics = {
        'prep': 0.0,
        'detect': 0.0,
        'crop_resize': 0.0,
        'recognize': 0.0,
        'memcpy': 0.0,
    }

    total_images = len(images_1k)
    logging.info(f"Running timed evaluation on {total_images} frames...")

    loop_start = time.perf_counter()
    for img in images_1k:
        t_start = time.perf_counter()
        inp = preprocess_yolo(img, img_size=det_size)
        metrics['prep'] += (time.perf_counter() - t_start) * 1000.0

        det_out, h2d1, d2h1, infer_time1 = detector.infer(inp)
        metrics['memcpy'] += h2d1 + d2h1
        metrics['detect'] += infer_time1

        t_start = time.perf_counter()
        boxes = postprocess_yolo(det_out, img.shape, det_size=det_size)
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crops.append(img[y1:y2, x1:x2])

        inp_ocr = None
        if crops:
            inp_ocr = preprocess_svtr([crops[0]])
        metrics['crop_resize'] += (time.perf_counter() - t_start) * 1000.0

        if inp_ocr is not None:
            rec_out, h2d2, d2h2, infer_time2 = ocr_model.infer(inp_ocr)
            metrics['memcpy'] += h2d2 + d2h2
            metrics['recognize'] += infer_time2

    avg_prep = metrics['prep'] / total_images
    avg_det = metrics['detect'] / total_images
    avg_crop = metrics['crop_resize'] / total_images
    avg_rec = metrics['recognize'] / total_images
    avg_memcpy = metrics['memcpy'] / total_images

    total_pipe = avg_prep + avg_det + avg_crop + avg_rec + avg_memcpy
    fps = 1000.0 / total_pipe if total_pipe > 0 else 0

    logging.info("Running actual end-to-end evaluation (without timer overhead)...")
    loop_start = time.perf_counter()
    for img in images_1k:
        inp = preprocess_yolo(img, img_size=det_size)
        det_out = detector.infer_fast(inp)
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
                ocr_model.infer_fast(inp_ocr)

    actual_fps = total_images / (time.perf_counter() - loop_start)

    detector.destroy()
    ocr_model.destroy()

    res_str = (
        f">>> FINAL RESULTS for {scenario_name}:\n"
        f"    Preprocess : {avg_prep:.2f} ms\n"
        f"    Detection  : {avg_det:.2f} ms\n"
        f"    Crop+Resize: {avg_crop:.2f} ms\n"
        f"    Recognition: {avg_rec:.2f} ms\n"
        f"    Memcpy     : {avg_memcpy:.2f} ms\n"
        f"    TOTAL PIPE : {total_pipe:.2f} ms\n"
        f"    FPS (Measured component sum) : {fps:.2f}\n" + f"    FPS (Actual end-to-end)      : {actual_fps:.2f}\n"
    )
    logging.info("\n" + res_str)
    results.append(res_str)

    with open("run_test_predict_time_pipeline_efficient_our.log", "a") as f:
        f.write("\n\n" + "="*80 + "\n")
        f.write(f"TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for res in results:
            f.write(res + "\n")

if __name__ == '__main__':
    main()
