import cv2
import numpy as np
import time
import os

# ---------- Procesamiento ----------
def preprocesar_frame(frame):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gris

# ---------- Detección de patrones ----------
anterior = None
def detectar_patrones(frame_actual):
    global anterior
    resultados = []

    if anterior is None:
        anterior = frame_actual
        return resultados

    flujo = cv2.calcOpticalFlowFarneback(anterior, frame_actual, None,
                                         0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = frame_actual.shape
    paso = 16
    for y in range(0, h, paso):
        for x in range(0, w, paso):
            fx, fy = flujo[y, x]
            if abs(fx) > 1 or abs(fy) > 1:
                resultados.append(((x, y), (int(x + fx), int(y + fy))))

    anterior = frame_actual
    return resultados

# ---------- Visualización ----------
def dibujar_sobre_frame(frame, resultados):
    for (x1, y1), (x2, y2) in resultados:
        cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return frame

# ---------- Guardar captura ----------
def guardar_captura(frame):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"captura_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Captura guardada como: {filename}")

# ---------- Captura de cámara principal ----------
def capturar_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_procesado = preprocesar_frame(frame)
        resultados = detectar_patrones(frame_procesado)
        frame_marcado = dibujar_sobre_frame(frame.copy(), resultados)

        cv2.imshow("Visión en vivo", frame_marcado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('s'):
            guardar_captura(frame_marcado)

    cap.release()
    cv2.destroyAllWindows()

# ---------- Ejecutar ----------
if __name__ == "__main__":
    capturar_video()
