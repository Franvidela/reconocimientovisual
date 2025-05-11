import cv2
import numpy as np
import os
import mediapipe as mp
from collections import deque

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Parámetros
SECUENCIAS = 30  # muestras por letra
FRAMES_POR_SECUENCIA = 30
LETRAS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Carpeta base para guardar dataset
DATASET_PATH = "dataset"
os.makedirs(DATASET_PATH, exist_ok=True)
for letra in LETRAS:
    os.makedirs(os.path.join(DATASET_PATH, letra), exist_ok=True)

# Función para extraer landmarks como array plano
def extraer_landmarks(results):
    if results.multi_hand_landmarks:
        mano = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in mano.landmark])
    return None

# Captura de datos
cap = cv2.VideoCapture(0)

for letra in LETRAS:
    print(f"\nPreparando para capturar la letra: {letra}")
    input("Presiona Enter cuando estés listo...")

    muestras_capturadas = 0
    while muestras_capturadas < SECUENCIAS:
        secuencia = []
        print(f"Grabando secuencia {muestras_capturadas + 1} de {SECUENCIAS} para {letra}...")

        frames_grabados = 0
        while frames_grabados < FRAMES_POR_SECUENCIA:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmarks = extraer_landmarks(results)
            if landmarks is not None:
                secuencia.append(landmarks)
                frames_grabados += 1

                # Dibujar puntos
                mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"Letra: {letra} - Frame {frames_grabados}/{FRAMES_POR_SECUENCIA}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Mano no detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Captura de datos", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Guardar secuencia
        np.save(os.path.join(DATASET_PATH, letra, f"seq_{muestras_capturadas}.npy"), np.array(secuencia))
        muestras_capturadas += 1

print("\n¡Captura completa para todas las letras!")
cap.release()
cv2.destroyAllWindows()
