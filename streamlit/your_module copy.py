import os
import time
import threading
from collections import deque
from pathlib import Path
import argparse

import cv2
import numpy as np
import tensorflow as tf

# ==== CONFIGURACIÓN DEL MODELO (usa tu ruta local existente) ====
# Cambia esto a tu ruta absoluta al saved_model
MODEL_PATH = r"C:\Users\fredy\Documents\UIDE\Inteligencia Artificial\Parcial 3\Semana 14\models\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8\saved_model"

# ==== HILO DE CAPTURA (siempre mantiene el último frame disponible) ====
class CameraStream:
    def __init__(self, cam_index=0, width=1280, height=720, target_fps=30, use_directshow=True):
        self.cam_index = cam_index
        flags = cv2.CAP_DSHOW if use_directshow else 0  # DirectShow acelera en Windows
        self.cap = cv2.VideoCapture(cam_index, flags)

        # Intenta mejorar la captura
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          target_fps)
        # MJPG suele dar más FPS en webcams USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        # Reduce el buffering interno (si el backend lo soporta)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara. Prueba con --cam 1 o revisa permisos.")

        self.queue = deque(maxlen=1)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                continue
            self.queue.append(frame)

    def read(self):
        # Devuelve el último frame disponible, o None si aún no hay
        return self.queue[-1] if self.queue else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

# ==== UTILIDADES ====
def draw_box(img, box_xyxy, label=None, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box_xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

def load_detector(model_path):
    print(f"[INFO] Cargando modelo: {model_path}")
    detect_fn = tf.saved_model.load(model_path)
    infer = detect_fn.signatures['serving_default']
    print("[OK] Modelo cargado")
    return infer

# ==== MAIN ====
def main():
    parser = argparse.ArgumentParser(description="Contador de personas con video fluido (SSD-MNV2-FPNLite-640 TF2)")
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara (0,1,...)")
    parser.add_argument("--conf", type=float, default=0.40, help="Umbral de confianza")
    parser.add_argument("--width", type=int, default=1280, help="Ancho de captura")
    parser.add_argument("--height", type=int, default=720, help="Alto de captura")
    parser.add_argument("--fps", type=int, default=30, help="FPS deseados de cámara")
    parser.add_argument("--decimate", type=int, default=1, help="Procesar detección cada N frames (1 = cada frame)")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Escala de visualización (ej. 0.75)")
    args = parser.parse_args()

    cv2.setUseOptimized(True)

    infer = load_detector(MODEL_PATH)

    stream = CameraStream(
        cam_index=0,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
        use_directshow=True  # en Linux/Mac no afecta
    ).start()

    print("[INFO] Presiona ESC para salir.")
    conf_thresh = args.conf
    frame_count = 0
    fps = 0.0
    t_prev = time.time()

    try:
        while True:
            frame_bgr = stream.read()
            if frame_bgr is None:
                # Aún no hay frame; evita bloquear
                cv2.waitKey(1)
                continue

            frame_count += 1
            h, w, _ = frame_bgr.shape

            do_infer = (frame_count % args.decimate == 0)

            person_count = 0
            boxes_to_draw = []
            scores_to_draw = []

            if do_infer:
                # Convertir a RGB uint8
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)[tf.newaxis, ...]
                outputs = infer(input_tensor)

                num = int(outputs["num_detections"][0])
                boxes   = outputs["detection_boxes"][0][:num].numpy()
                classes = outputs["detection_classes"][0][:num].numpy().astype(np.int32)
                scores  = outputs["detection_scores"][0][:num].numpy()

                # Filtra solo personas (COCO id=1) y por confianza
                mask = (classes == 1) & (scores >= conf_thresh)
                sel_boxes  = boxes[mask]
                sel_scores = scores[mask]

                person_count = sel_boxes.shape[0]

                # Convierte normalizado -> pixeles
                if person_count > 0:
                    ymins, xmins, ymaxs, xmaxs = sel_boxes[:,0], sel_boxes[:,1], sel_boxes[:,2], sel_boxes[:,3]
                    x1 = (xmins * w).astype(np.int32)
                    y1 = (ymins * h).astype(np.int32)
                    x2 = (xmaxs * w).astype(np.int32)
                    y2 = (ymaxs * h).astype(np.int32)

                    for i in range(person_count):
                        boxes_to_draw.append((x1[i], y1[i], x2[i], y2[i]))
                        scores_to_draw.append(sel_scores[i])

            # Dibuja (si no hubo inferencia este frame, mantiene el conteo previo en 0 visual,
            # si quieres persistir último conteo, puedes recordar 'person_count' en una variable global)
            for i, box in enumerate(boxes_to_draw):
                draw_box(frame_bgr, box, label=f"Persona {i+1} ({scores_to_draw[i]:.2f})", color=(0,255,0))

            # Texto HUD
            cv2.putText(frame_bgr, f"Personas: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            # FPS (suavizado exponencial)
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2, cv2.LINE_AA)

            # Escala de visualización (para bajar carga de render)
            if args.display_scale != 1.0:
                frame_display = cv2.resize(frame_bgr, None, fx=args.display_scale, fy=args.display_scale,
                                           interpolation=cv2.INTER_AREA)
            else:
                frame_display = frame_bgr

            cv2.imshow("Contador de Personas (fluido)", frame_display)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()