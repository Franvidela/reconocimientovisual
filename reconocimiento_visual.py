import cv2
import mediapipe as mp
import time
from collections import deque

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Historial de posiciones de la palma para detectar trayectoria
trayectoria = deque(maxlen=20)

# ---------- Detección y análisis de movimiento ----------
def detectar_movimiento(frame):
    global trayectoria
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(frame_rgb)
    gesto_detectado = None

    if resultados.multi_hand_landmarks:
        for mano in resultados.multi_hand_landmarks:
            # Punto 0 es la base de la palma
            punto_palma = mano.landmark[0]
            h, w, _ = frame.shape
            x = int(punto_palma.x * w)
            y = int(punto_palma.y * h)
            trayectoria.append((x, y))
            mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

        # Lógica simple: si la mano se mueve horizontalmente izquierda-derecha
        if len(trayectoria) >= 10:
            x_dif = trayectoria[-1][0] - trayectoria[0][0]
            if abs(x_dif) > 100:
                gesto_detectado = "Saludo (hola)"

    return frame, gesto_detectado

# ---------- Mostrar resultado en pantalla ----------
def mostrar_resultado(frame, gesto):
    if gesto:
        cv2.putText(frame, f"Gesto: {gesto}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

        frame = cv2.flip(frame, 1)  # espejo horizontal para mayor naturalidad
        frame, gesto = detectar_movimiento(frame)
        frame = mostrar_resultado(frame, gesto)

        cv2.imshow("Lenguaje de Señas - Demo", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('s'):
            guardar_captura(frame)

    cap.release()
    cv2.destroyAllWindows()

# ---------- Ejecutar ----------
if __name__ == "__main__":
    capturar_video()
