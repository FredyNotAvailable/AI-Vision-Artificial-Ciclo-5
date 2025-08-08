import threading
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

MODEL_PATH = r"C:\Users\fredy\Documents\UIDE\Inteligencia Artificial\Parcial 3\Semana 14\models\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\saved_model"

class CameraStream:
    def __init__(self, cam_index=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {cam_index}")

        self.queue = deque(maxlen=1)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.queue.append(frame)

    def read(self):
        if self.queue:
            return self.queue[-1]
        return None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()

def load_detector():
    detect_fn = tf.saved_model.load(MODEL_PATH)
    return detect_fn.signatures['serving_default']

def draw_box(img, box, label=None, color=(0,255,0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def detectar_personas(frame_bgr, infer, conf_thresh=0.4):
    h, w, _ = frame_bgr.shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)[tf.newaxis, ...]
    outputs = infer(input_tensor)

    num = int(outputs["num_detections"][0])
    boxes   = outputs["detection_boxes"][0][:num].numpy()
    classes = outputs["detection_classes"][0][:num].numpy().astype(np.int32)
    scores  = outputs["detection_scores"][0][:num].numpy()

    mask = (classes == 1) & (scores >= conf_thresh)
    sel_boxes  = boxes[mask]
    sel_scores = scores[mask]

    person_count = sel_boxes.shape[0]

    for i in range(person_count):
        ymin, xmin, ymax, xmax = sel_boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        draw_box(frame_bgr, (x1, y1, x2, y2), label=f"Persona {i+1} ({sel_scores[i]:.2f})")

    cv2.putText(frame_bgr, f"Personas: {person_count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)

    return frame_bgr, person_count
