import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = 'C:/Users/fredy/Documents/UIDE/Inteligencia Artificial/Parcial 3/Semana 12/MobileNetV2/ssd_mobilenet_v2_coco_2018_03_29/saved_model'

detect_fn = tf.saved_model.load(MODEL_PATH)
infer = detect_fn.signatures['serving_default']

category_index = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
    65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

# Variables globales para el área de conteo
counting_area = None
drawing = False
start_point = None

def mouse_callback(event, x, y, flags, param):
    global counting_area, drawing, start_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if start_point:
            counting_area = {
                'x1': min(start_point[0], x),
                'y1': min(start_point[1], y),
                'x2': max(start_point[0], x),
                'y2': max(start_point[1], y)
            }
            print(f"Área de conteo definida: {counting_area}")

def is_in_counting_area(center_x, center_y, area):
    """Verifica si un punto está dentro del área de conteo"""
    if area is None:
        return False
    return (area['x1'] <= center_x <= area['x2'] and 
            area['y1'] <= center_y <= area['y2'])

def get_center_point(box, width, height):
    """Obtiene el centro de la caja delimitadora"""
    ymin, xmin, ymax, xmax = box
    center_x = int((xmin + xmax) * width / 2)
    center_y = int((ymin + ymax) * height / 2)
    return center_x, center_y

cap = cv2.VideoCapture(0)

# Configurar callback del mouse
cv2.namedWindow("Contador de Personas")
cv2.setMouseCallback("Contador de Personas", mouse_callback)

print("Instrucciones:")
print("1. Haz clic y arrastra para definir el área de conteo")
print("2. Presiona 'r' para resetear el área")
print("3. Presiona ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = infer(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape
    person_count = 0

    # Dibujar área de conteo si está definida
    if counting_area:
        cv2.rectangle(frame, 
                     (counting_area['x1'], counting_area['y1']), 
                     (counting_area['x2'], counting_area['y2']), 
                     (255, 0, 0), 2)
        cv2.putText(frame, "AREA DE CONTEO", 
                   (counting_area['x1'], counting_area['y1'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Procesar detecciones
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:  # Solo personas con confianza > 50%
            ymin, xmin, ymax, xmax = boxes[i]
            start_box = (int(xmin * width), int(ymin * height))
            end_box = (int(xmax * width), int(ymax * height))
            
            # Obtener centro de la persona
            center_x, center_y = get_center_point(boxes[i], width, height)
            
            # Verificar si está en el área de conteo
            in_area = is_in_counting_area(center_x, center_y, counting_area)
            
            if in_area:
                person_count += 1
                # Dibujar rectángulo verde para personas en el área
                cv2.rectangle(frame, start_box, end_box, (0, 255, 0), 3)
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            else:
                # Dibujar rectángulo gris para personas fuera del área
                cv2.rectangle(frame, start_box, end_box, (128, 128, 128), 2)
                cv2.circle(frame, (center_x, center_y), 3, (128, 128, 128), -1)
            
            # Etiqueta con confianza
            label = f"Persona: {int(scores[i]*100)}%"
            cv2.putText(frame, label, (start_box[0], start_box[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Mostrar contador
    counter_text = f"Personas en area: {person_count}"
    cv2.putText(frame, counter_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Mostrar instrucciones
    if counting_area is None:
        cv2.putText(frame, "Haz clic y arrastra para definir area de conteo", 
                   (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Contador de Personas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    elif key == ord('r'):  # 'r' para resetear área
        counting_area = None
        print("Área de conteo reseteada")

cap.release()
cv2.destroyAllWindows()