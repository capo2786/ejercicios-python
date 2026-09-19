#!/usr/bin/env python3
"""
generar_video_prueba.py — crea un video sintetico de un canal con el agua subiendo.

Sirve para probar flood_monitor.py sin necesitar una camara real. La superficie
sube de forma conocida, asi puedes comparar lo que MIDE el script contra lo que
REALMENTE pasa: es la unica forma honesta de evaluar un detector.
"""
import cv2
import numpy as np

W, H, FPS, SEG = 640, 360, 25, 40
NIVEL_INICIAL_PX, NIVEL_FINAL_PX = 300, 150      # y del agua (menor y = mas alto)

out = cv2.VideoWriter("rio_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
rng = np.random.default_rng(7)
xx = np.arange(W)

for i in range(FPS * SEG):
    t = i / FPS
    u = t / SEG
    f = np.zeros((H, W, 3), np.uint8)
    f[:, :] = (110, 125, 120)                                  # concreto del canal
    cv2.rectangle(f, (0, 0), (W, 110), (150, 160, 165), -1)    # cielo
    cv2.rectangle(f, (0, 108), (W, 116), (95, 105, 100), -1)   # baranda

    y = int(NIVEL_INICIAL_PX - (NIVEL_INICIAL_PX - NIVEL_FINAL_PX) * u)

    # agua: base + oleaje (ondas que se mueven) + un poco de ruido fino
    agua = np.zeros((H - y, W, 3), np.float32)
    agua[:, :] = (150, 105, 40)
    filas = np.arange(H - y).reshape(-1, 1)
    onda = (12 * np.sin(xx / 14.0 + t * 5.0 + filas / 9.0)
            + 7 * np.sin(xx / 31.0 - t * 3.2))
    agua += onda[:, :, None]
    agua += rng.normal(0, 4, agua.shape)
    f[y:, :] = np.clip(agua, 0, 255).astype(np.uint8)

    cv2.line(f, (0, y), (W, y), (185, 145, 85), 2)             # brillo del borde
    out.write(f)

out.release()
print(f"rio_test.mp4 listo: {FPS*SEG} cuadros, el agua sube de "
      f"y={NIVEL_INICIAL_PX}px a y={NIVEL_FINAL_PX}px en {SEG}s")
