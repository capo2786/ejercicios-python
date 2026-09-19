#!/usr/bin/env python3
"""
flood_monitor.py — Deteccion de riesgo de inundacion sobre un VIDEO (un solo script)
====================================================================================

Lee un archivo de video de una camara fija apuntando a un rio/canal y, cuadro a
cuadro:

  1. Segmenta el agua  = movimiento persistente (la lamina de agua nunca esta
     quieta) + color, restringido a una region de interes (ROI) que tu defines.
  2. Extrae la linea de agua (borde superior del agua) columna por columna.
  3. Convierte pixeles -> metros con una calibracion de 2 puntos de altura conocida.
  4. Suaviza la serie, calcula tendencia (m/h) y caudal con curva de gasto.
  5. Clasifica el riesgo: NORMAL / VIGILANCIA / ALERTA / DESBORDE CRITICO.
  6. Dibuja el HUD encima y escribe el CSV + el video anotado.

IMPORTANTE: el reloj del analisis es el TIEMPO DEL VIDEO (no el del CPU), asi que
la tendencia en m/h sale igual procese a 5 FPS o a 300 FPS.

Instalacion:
    pip install opencv-python numpy

Uso:
    # 1) Calibrar una vez por camara (clics sobre un cuadro del video)
    python flood_monitor.py rio.mp4 --calibrate

    # 2) Analizar el video y exportar resultados
    python flood_monitor.py rio.mp4 --station 02 --record salida.mp4

    # Sin ventana (servidor) y solo un tramo del video
    python flood_monitor.py rio.mp4 --headless --start 30 --end 180

    # Ver a velocidad real / mas rapido
    python flood_monitor.py rio.mp4 --speed 1     (1 = tiempo real, 0 = a tope)

Teclas: q salir | espacio pausa | s captura PNG | r reinicia historial
"""

# Permite usar anotaciones de tipo modernas (ej. "float | None") en Python 3.8+
from __future__ import annotations

import argparse                 # leer las opciones de la linea de comandos
import csv                      # escribir el archivo de resultados niveles.csv
import json                     # leer/guardar la calibracion (station.json)
import os                       # comprobar si existen archivos
import sys                      # salir con mensaje de error, escribir en consola
import time                     # reloj real, para reproducir a velocidad controlada
import urllib.request           # enviar alertas por HTTP (webhook), sin librerias extra
from collections import deque   # lista con tamano maximo: al llenarse descarta lo mas viejo
from datetime import datetime, timedelta   # fechas y formato hh:mm:ss

import cv2                      # OpenCV: leer video, procesar imagenes, dibujar
import numpy as np              # calculo numerico con arreglos (las imagenes son arreglos)

# --------------------------------------------------------------------------- #
# Paleta (BGR)
# --------------------------------------------------------------------------- #
# OpenCV usa el orden Azul-Verde-Rojo (BGR), no RGB. (230, 220, 60) es cian.
CYAN = (230, 220, 60)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
DARK = (35, 28, 20)
# Un color por nivel de riesgo: verde -> amarillo -> naranja -> rojo
RISK_COLORS = {
    "NORMAL": (120, 220, 0),
    "VIGILANCIA": (0, 220, 255),
    "ALERTA": (0, 140, 255),
    "DESBORDE CRITICO": (60, 60, 255),
}
# Orden de gravedad; sirve para "subir un escalon" de riesgo con un indice
RISK_ORDER = ["NORMAL", "VIGILANCIA", "ALERTA", "DESBORDE CRITICO"]

# Configuracion por defecto. station.json puede sobrescribir cualquiera de estas claves.
DEFAULT_CONFIG = {
    "roi": None,                  # [[x,y], ...] poligono del cauce
    "ref_points": None,           # [[x1,y1,h1_m], [x2,y2,h2_m]]
    # alturas (en metros) a partir de las cuales cambia el nivel de riesgo
    "thresholds": {"vigilancia": 1.8, "alerta": 2.6, "critico": 3.2},
    "rating": {"c": 2.1, "h0": 0.15, "exp": 1.5},      # Q = c*(h-h0)^exp
    # cuanto pesa el movimiento y el color al decidir si un pixel es agua,
    # y "bin" es el umbral final: puntaje > 0.38 -> es agua
    "weights": {"motion": 0.62, "color": 0.38, "bin": 0.38},
    # rango de color del agua en HSV (Tono, Saturacion, Valor): minimo y maximo
    "hsv_low": [70, 10, 25],
    "hsv_high": [135, 190, 235],
    "smooth_alpha": 0.18,         # EMA del nivel
    "trend_window_s": 25.0,       # ventana (en segundos de video) para la tendencia
    "rise_rate_alert": 0.35,      # m/h a partir del cual se escala el riesgo
}


# --------------------------------------------------------------------------- #
# 1. Segmentacion del agua
# --------------------------------------------------------------------------- #
# "Segmentar" = decidir, pixel por pixel, si pertenece al agua o no.
# El resultado es una MASCARA: imagen en blanco (255 = agua) y negro (0 = no agua).
class WaterSegmenter:
    """Mascara de agua = movimiento persistente + color, dentro del ROI."""

    def __init__(self, cfg: dict) -> None:
        # Sustractor de fondo MOG2: aprende como se ve la escena "normal" y marca
        # los pixeles que cambian. Lo quieto (orillas, puentes) pasa a ser fondo;
        # el agua, que se mueve siempre, queda marcada como primer plano.
        #   history=400      -> recuerda ~400 cuadros para modelar el fondo
        #   varThreshold=20  -> que tan distinto debe ser un pixel para contar como cambio
        #   detectShadows    -> apagado: no nos interesa distinguir sombras
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=20, detectShadows=False
        )
        # Acumulador de movimiento: promedio de cuanto se ha movido cada pixel
        # en los ultimos cuadros. Arranca vacio (None) y se crea con el primer cuadro.
        self.motion_acc: np.ndarray | None = None
        self.w = cfg["weights"]
        # Limites de color HSV como arreglos de 8 bits (lo que pide cv2.inRange)
        self.lo = np.array(cfg["hsv_low"], dtype=np.uint8)
        self.hi = np.array(cfg["hsv_high"], dtype=np.uint8)
        # "Pincel" eliptico de 7x7 para la morfologia (cerrar huecos, borrar puntitos)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def update(self, frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        # energia de movimiento acumulada: el agua "titila" en todos los cuadros
        # Se desenfoca antes para que el ruido del sensor no parezca movimiento.
        fg = self.bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        # fg > 0 -> 1.0 donde hubo cambio, 0.0 donde no
        motion = (fg > 0).astype(np.float32)
        if self.motion_acc is None:
            self.motion_acc = motion
        else:
            # Promedio movil: 90 % de lo que ya sabiamos + 10 % del cuadro nuevo.
            # Un auto que pasa una vez casi no deja huella; el agua, que se mueve
            # en TODOS los cuadros, acumula un valor alto. Esa es la clave.
            self.motion_acc = 0.90 * self.motion_acc + 0.10 * motion
        # Desenfoque grande para convertir puntitos sueltos en zonas continuas
        motion_s = cv2.GaussianBlur(self.motion_acc, (21, 21), 0)
        # Normaliza a 0..1: con 35 % de actividad ya se considera "movimiento pleno"
        motion_s = np.clip(motion_s / 0.35, 0.0, 1.0)

        # color tipico del agua (azul-verdoso / gris, saturacion baja-media)
        # HSV separa el tono del brillo, asi el color resiste mejor los cambios de luz
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # inRange da 255 si el pixel esta dentro del rango de color; se pasa a 0..1
        color = cv2.inRange(hsv, self.lo, self.hi).astype(np.float32) / 255.0
        color = cv2.GaussianBlur(color, (15, 15), 0)

        # Puntaje combinado por pixel: 62 % movimiento + 38 % color.
        # El movimiento pesa mas porque el color solo engana (cielo, sombras azules).
        score = self.w["motion"] * motion_s + self.w["color"] * color
        # Umbral: puntaje alto -> 255 (agua), bajo -> 0
        mask = (score > self.w["bin"]).astype(np.uint8) * 255
        # Solo nos interesa lo que esta dentro del ROI (el cauce); lo demas se borra
        mask = cv2.bitwise_and(mask, roi_mask)

        # Morfologia matematica:
        #   CLOSE (dilatar y luego erosionar) -> rellena huecos dentro del agua
        #   OPEN  (erosionar y luego dilatar) -> borra manchitas sueltas de ruido
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        # El rio es UNA sola mancha grande; se descarta todo lo demas
        return self._largest_blob(mask)

    @staticmethod
    def _largest_blob(mask: np.ndarray) -> np.ndarray:
        # Etiqueta cada mancha blanca conectada (vecindad de 8 pixeles).
        # n = numero de manchas (incluye el fondo, que es la etiqueta 0)
        # stats = por cada mancha: x, y, ancho, alto, AREA
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask                 # no hay ninguna mancha, solo fondo
        # stats[1:] salta el fondo; argmax elige la de mayor area; +1 corrige el salto
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # Nueva mascara con solo esa mancha
        return np.where(labels == idx, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 2. Linea de agua (borde superior de la lamina)
# --------------------------------------------------------------------------- #
def water_line(mask: np.ndarray, roi_box: tuple, min_run: int = 6, step: int = 4):
    """Para cada columna busca la primera racha continua de agua.

    Devuelve (puntos, y_mediana, MAD). La racha minima evita que un reflejo
    suelto o una gota en el lente se tome como superficie del agua.
    """
    # Rectangulo que encierra el ROI: solo se recorre esa zona
    x0, y0, x1, y1 = roi_box
    pts: list[tuple[int, int]] = []

    # Recorre columnas de izquierda a derecha, de 4 en 4 pixeles (mas rapido)
    for x in range(x0, x1, step):
        # Columna vertical de la mascara: True donde hay agua (de arriba hacia abajo)
        col = mask[y0:y1, x] > 0
        if not col.any():
            continue                    # en esta columna no hay agua
        # posiciones donde arranca cada racha y su longitud (vectorizado)
        # Truco: se rodea la columna con ceros y se calcula la diferencia entre
        # vecinos. +1 = empieza agua, -1 = termina agua.
        d = np.diff(np.concatenate(([0], col.view(np.uint8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        runs = ends - starts            # largo de cada racha de agua
        # Rachas de al menos 6 pixeles seguidos (lo mas corto es ruido)
        ok = np.flatnonzero(runs >= min_run)
        if ok.size:
            # La PRIMERA racha valida (la mas alta en la imagen) es la superficie.
            # Se guarda su coordenada y en la imagen completa (por eso + y0).
            pts.append((x, y0 + int(starts[ok[0]])))

    if not pts:
        return [], None, None           # no se encontro superficie en ninguna columna

    # Estadistica robusta para descartar columnas raras:
    # MEDIANA en vez de promedio (un valor extremo no la mueve) y
    # MAD = desviacion absoluta mediana (que tan dispersos estan los puntos).
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    med = float(np.median(ys))
    mad = float(np.median(np.abs(ys - med)))
    # Tolerancia: 3 veces la dispersion, pero nunca menos de 8 pixeles
    tol = max(3.0 * mad, 8.0)
    # Se quedan solo los puntos cercanos a la mediana (fuera reflejos, orillas, etc.)
    keep = [p for p in pts if abs(p[1] - med) <= tol]
    if keep:
        # Se recalcula la mediana con los puntos limpios: ese es el nivel en pixeles
        med = float(np.median([p[1] for p in keep]))
    # MAD sirve despues como medida de confianza: linea plana = MAD chico = confiable
    return keep, med, mad


# --------------------------------------------------------------------------- #
# 3. Calibracion pixel -> metro
# --------------------------------------------------------------------------- #
# La camara mide en pixeles, pero necesitamos metros. Si conocemos la altura
# real de 2 puntos de la imagen (ej. marcas de una regleta), trazamos una recta
# entre ellos y convertimos cualquier "y" en pixeles a metros.
class Calibration:
    """Escala lineal a partir de 2 puntos de altura conocida en la imagen."""

    def __init__(self, ref_points) -> None:
        # Cada punto: (x en pixeles, y en pixeles, altura real en metros)
        (x1, y1, h1), (x2, y2, h2) = ref_points
        # Si ambos puntos estan a la misma altura en la imagen no hay escala posible
        if abs(float(y2) - float(y1)) < 1e-6:
            raise ValueError("Los 2 puntos de referencia no pueden estar a la misma altura en pixeles.")
        self.y1, self.h1 = float(y1), float(h1)
        # Pendiente de la recta: cuantos metros vale 1 pixel.
        # Suele ser NEGATIVA, porque en las imagenes la y crece hacia ABAJO
        # y el agua sube hacia ARRIBA.
        self.m_per_px = (float(h2) - float(h1)) / (float(y2) - float(y1))

    def to_meters(self, y_px: float) -> float:
        # Ecuacion de la recta: altura = h1 + (y - y1) * metros_por_pixel
        return self.h1 + (float(y_px) - self.y1) * self.m_per_px


# --------------------------------------------------------------------------- #
# 4. Estado hidrologico
# --------------------------------------------------------------------------- #
# Guarda la "memoria" del rio: nivel suavizado, historial, pico maximo,
# confianza y riesgo actual.
class RiverState:
    def __init__(self, cfg: dict) -> None:
        self.th = cfg["thresholds"]
        self.rt = cfg["rating"]
        self.alpha = cfg["smooth_alpha"]
        self.window = cfg["trend_window_s"]
        self.rise_alert = cfg["rise_rate_alert"]
        self.level: float | None = None           # nivel actual suavizado (m)
        self.hist: deque = deque(maxlen=4000)     # (t_video_s, nivel_m)
        self.conf_hist: deque = deque(maxlen=30)  # confianza de los ultimos 30 cuadros
        self.risk = "NORMAL"
        self.peak: float | None = None            # nivel maximo visto

    def update(self, level_raw, mad_px, coverage, t_video):
        # Si en este cuadro no se encontro la linea de agua, la confianza es 0
        if level_raw is None:
            self.conf_hist.append(0.0)
            return
        # EMA (media movil exponencial): 82 % del nivel anterior + 18 % del nuevo.
        # Asi las olas y el ruido no hacen "saltar" el nivel de un cuadro a otro.
        self.level = (level_raw if self.level is None
                      else (1 - self.alpha) * self.level + self.alpha * level_raw)
        self.hist.append((t_video, self.level))
        self.peak = self.level if self.peak is None else max(self.peak, self.level)

        # Confianza del cuadro, combinando dos senales (cada una de 0 a 1):
        c_line = float(np.clip(1.0 - (mad_px / 25.0), 0.0, 1.0))   # linea consistente
        c_cov = float(np.clip(coverage / 0.25, 0.0, 1.0))          # ROI bien cubierto
        self.conf_hist.append(0.65 * c_line + 0.35 * c_cov)
        self.risk = self._classify()

    def reset_history(self) -> None:
        """Descarta las muestras del arranque, conservando el nivel suavizado."""
        # Se usa al terminar el warmup: los primeros segundos son poco fiables
        # porque el sustractor de fondo todavia esta aprendiendo la escena.
        self.hist.clear()
        self.conf_hist.clear()
        self.peak = self.level
        self.risk = "NORMAL"

    @property
    def confidence(self) -> float:
        # Promedio de la confianza de los ultimos 30 cuadros
        return float(np.mean(self.conf_hist)) if self.conf_hist else 0.0

    @property
    def trend(self) -> float:
        """m/h — pendiente por minimos cuadrados sobre segundos DE VIDEO."""
        # Pocas muestras = tendencia no confiable -> 0
        if len(self.hist) < 8:
            return 0.0
        # Solo se usan las muestras de los ultimos 25 segundos de video
        t_now = self.hist[-1][0]
        pts = [(t, h) for t, h in self.hist if t_now - t <= self.window]
        if len(pts) < 8:
            return 0.0
        t = np.array([p[0] for p in pts], dtype=np.float64)
        h = np.array([p[1] for p in pts], dtype=np.float64)
        if t.max() - t.min() < 1e-6:
            return 0.0                  # todas las muestras en el mismo instante
        # Ajusta una recta nivel = a*t + b por minimos cuadrados.
        # "a" (la pendiente) es la velocidad de subida en metros por SEGUNDO.
        slope = float(np.polyfit(t - t[0], h, 1)[0])
        # x 3600 -> metros por HORA, la unidad que usan los hidrologos
        return slope * 3600.0

    @property
    def flow(self) -> float:
        # Caudal estimado (m3/s) con la "curva de gasto": Q = c * (h - h0)^exp
        # h0 es el nivel al que el rio deja de correr. Los coeficientes se obtienen
        # en campo; aqui son valores de ejemplo.
        if self.level is None:
            return 0.0
        head = max(self.level - self.rt["h0"], 0.0)
        return float(self.rt["c"] * head ** self.rt["exp"])

    def _classify(self) -> str:
        h, tr = self.level, self.trend
        # 1) Riesgo segun la ALTURA, comparando contra los umbrales
        if h >= self.th["critico"]:
            r = "DESBORDE CRITICO"
        elif h >= self.th["alerta"]:
            r = "ALERTA"
        elif h >= self.th["vigilancia"]:
            r = "VIGILANCIA"
        else:
            r = "NORMAL"
        # 2) Si ademas sube RAPIDO, se sube un escalon: un rio a 2 m que crece
        #    0.5 m/h es mas peligroso que uno a 2 m que esta quieto.
        if tr > self.rise_alert and r != "DESBORDE CRITICO":   # sube rapido -> escala
            r = RISK_ORDER[min(RISK_ORDER.index(r) + 1, 3)]
        return r


# --------------------------------------------------------------------------- #
# 5. HUD
# --------------------------------------------------------------------------- #
# HUD = los paneles con informacion que se dibujan encima del video.
def panel(img, x, y, w, h, title, rows, accent=CYAN):
    # Recorta el panel para que no se salga de la imagen
    H, W = img.shape[:2]
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        return
    # Fondo semitransparente: mezcla 62 % color oscuro + 38 % imagen original.
    # "sub" es una vista del recorte, asi que se modifica la imagen directamente.
    sub = img[y:y + h, x:x + w]
    cv2.addWeighted(np.full(sub.shape, DARK, np.uint8), 0.62, sub, 0.38, 0, sub)
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 1)       # borde
    cv2.rectangle(img, (x, y), (x + w, y + 22), accent, -1)     # barra de titulo (relleno)
    cv2.putText(img, title, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DARK, 1, cv2.LINE_AA)
    # Cada fila: etiqueta a la izquierda (color de acento) y valor a la derecha (blanco)
    for i, (k, v) in enumerate(rows):
        yy = y + 42 + i * 20
        if yy > y + h - 4:
            break                       # no cabe mas texto en el panel
        cv2.putText(img, f"{k}:", (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, accent, 1, cv2.LINE_AA)
        cv2.putText(img, str(v), (x + 122, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)


def sparkline(img, x, y, w, h, hist, color=CYAN):
    # Mini grafico del nivel en el tiempo, dentro de un rectangulo
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    if len(hist) < 3:
        return
    # Ultimos "w" valores: uno por cada pixel de ancho
    vals = np.array([v for _, v in hist][-w:], dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    # Rango minimo de 8 cm para que el ruido no parezca una ola enorme
    rng = max(hi - lo, 0.08)
    # Convierte cada valor a coordenadas de pixel (y invertida: mas alto = mas arriba)
    xs = np.linspace(x, x + w, len(vals)).astype(np.int32)
    ys = (y + h - (vals - lo) / rng * h).astype(np.int32)
    cv2.polylines(img, [np.stack([xs, ys], 1)], False, color, 1, cv2.LINE_AA)


def draw_hud(frame, mask, line_pts, roi_poly, state, station, t_video,
             progress, warming=False):
    # Se dibuja sobre una copia para no alterar el cuadro original
    out = frame.copy()
    # Pinta el agua detectada con un tinte azul al 30 %
    tint = np.zeros_like(out)
    tint[mask > 0] = (140, 90, 20)
    cv2.addWeighted(tint, 0.30, out, 0.70, 0, out)

    # Contorno del ROI (gris) y linea de agua detectada (cian)
    if roi_poly is not None:
        cv2.polylines(out, [roi_poly], True, GREY, 1)
    if len(line_pts) > 1:
        cv2.polylines(out, [np.array(line_pts, np.int32)], False, CYAN, 2, cv2.LINE_AA)

    # Durante el calentamiento se usa cian neutro; despues, el color del riesgo
    color = CYAN if warming else RISK_COLORS.get(state.risk, CYAN)
    lvl = "--" if state.level is None else f"{state.level:.2f} m"
    tr = state.trend
    # Menos de 5 cm/h en cualquier sentido se considera "estable"
    arrow = "SUBIENDO" if tr > 0.05 else ("BAJANDO" if tr < -0.05 else "ESTABLE")

    # Panel principal (arriba a la izquierda)
    panel(out, 14, 14, 306, 166, f"ESTACION {station}  |  ANALISIS DE VIDEO", [
        ("Riesgo", "CALIBRANDO..." if warming else state.risk),
        ("Confianza", f"{state.confidence * 100:.1f} %"),
        ("Nivel", lvl),
        ("Tendencia", f"{arrow} ({tr:+.2f} m/h)"),
        ("Caudal est.", f"{state.flow:.2f} m3/s"),
        ("Nivel max.", "--" if state.peak is None else f"{state.peak:.2f} m"),
    ], accent=color)

    # Panel de historico (arriba a la derecha), solo si hay espacio suficiente
    H, W = out.shape[:2]
    pw = min(262, max(W - 300, 120))
    px = W - pw - 14
    if px > 330:
        panel(out, px, 14, pw, 116, "HISTORICO DE NIVEL", [
            ("Muestras", len(state.hist)),
            ("T. video", str(timedelta(seconds=int(t_video)))),
        ], accent=CYAN)
        sparkline(out, px + 10, 84, pw - 20, 36, list(state.hist), color)

    # barra de avance del video
    if progress is not None:
        cv2.rectangle(out, (0, H - 5), (int(W * progress), H - 1), color, -1)
    # Fecha y hora actuales abajo a la izquierda
    cv2.putText(out, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    # Marco grueso de color alrededor de toda la imagen si hay ALERTA o peor
    if not warming and state.risk in ("ALERTA", "DESBORDE CRITICO"):
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), color, 6)
    return out


# --------------------------------------------------------------------------- #
# 6. Calibracion interactiva sobre un cuadro del video
# --------------------------------------------------------------------------- #
# Se usa con --calibrate: el usuario marca con el mouse el ROI y 2 puntos de
# altura conocida. El resultado se guarda en station.json.
def run_calibration(frame, path):
    clicks: list[tuple[int, int]] = []

    # Funcion que OpenCV llama en cada evento del mouse sobre la ventana
    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))           # clic izquierdo: agrega punto
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()                    # clic derecho: deshace el ultimo

    cv2.namedWindow("calibracion", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibracion", on_mouse)
    print("\n[1] Clic en los vertices del ROI (el cauce). ENTER para cerrarlo.")
    print("[2] Luego 2 clics en puntos de altura conocida (regleta, pilar, borde).")
    print("    Clic derecho deshace. ESC cancela.\n")

    roi = None
    refs: list[tuple[int, int]] = []
    # Bucle de dibujo: redibuja los clics cada 20 ms hasta que el usuario termine
    while True:
        vis = frame.copy()
        # Circulo numerado en cada clic
        for i, p in enumerate(clicks):
            cv2.circle(vis, p, 5, CYAN, -1)
            cv2.putText(vis, str(i + 1), (p[0] + 8, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
        # Fase 1: poligono abierto mientras se marcan los vertices del ROI
        if roi is None and len(clicks) > 1:
            cv2.polylines(vis, [np.array(clicks, np.int32)], False, CYAN, 1)
        # Fase 2: el ROI ya cerrado se muestra en gris
        if roi is not None:
            cv2.polylines(vis, [np.array(roi, np.int32)], True, GREY, 1)
        msg = "ROI: clic vertices + ENTER" if roi is None else "REFERENCIAS: 2 clics + ENTER"
        cv2.putText(vis, msg, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.imshow("calibracion", vis)

        # Lee el teclado. 27 = ESC; 13 y 10 = ENTER (segun el sistema operativo)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit("Calibracion cancelada.")
        if k in (13, 10):
            if roi is None:
                # Termina la fase 1: un poligono necesita al menos 3 vertices
                if len(clicks) < 3:
                    print("Necesitas al menos 3 vertices.")
                    continue
                roi = [list(p) for p in clicks]
                clicks.clear()              # se reutiliza la lista para la fase 2
            else:
                # Termina la fase 2: exactamente 2 puntos de referencia
                if len(clicks) != 2:
                    print("Necesitas exactamente 2 puntos de referencia.")
                    continue
                refs = list(clicks)
                break
    cv2.destroyAllWindows()

    # Las alturas reales se escriben en la consola (ej. marcas de la regleta)
    h1 = float(input("Altura real del punto 1 (m): ").strip())
    h2 = float(input("Altura real del punto 2 (m): ").strip())
    # Copia profunda de la configuracion por defecto (truco: pasar a JSON y volver)
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["roi"] = roi
    cfg["ref_points"] = [[refs[0][0], refs[0][1], h1], [refs[1][0], refs[1][1], h2]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {path}. Ahora corre sin --calibrate.")
    return cfg


def load_config(path, frame_shape):
    # Si existe station.json: se parte de los valores por defecto y se
    # reemplazan solo las claves que trae el archivo
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg.update(user)
        print(f"[config] {path}")
        return cfg
    # Sin calibracion: ROI = imagen completa y una escala inventada
    # (abajo = 0 m, a 35 % de la altura = 3 m). Sirve para probar, no para medir.
    h, w = frame_shape[:2]
    print("[aviso] Sin calibracion: ROI = cuadro completo y escala supuesta de 3 m.\n"
          "        Los metros son APROXIMADOS. Corre --calibrate para datos reales.")
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["roi"] = [[0, 0], [w, 0], [w, h], [0, h]]
    cfg["ref_points"] = [[w // 2, int(h * 0.95), 0.0], [w // 2, int(h * 0.35), 3.0]]
    return cfg


# --------------------------------------------------------------------------- #
# 7. Alertas
# --------------------------------------------------------------------------- #
# Envia un POST con JSON a una URL (webhook): asi se puede conectar con
# Telegram, Slack, un servidor propio, etc.
def send_alert(webhook, payload):
    if not webhook:
        return                          # no se configuro --webhook: no se envia nada
    try:
        req = urllib.request.Request(
            webhook, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:      # noqa: BLE001
        # Si falla la red, se avisa pero el monitoreo NO se detiene
        print(f"[alerta] fallo el webhook: {exc}")


# --------------------------------------------------------------------------- #
# 8. Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
    # ---- opciones de la linea de comandos (python flood_monitor.py --help) ---- #
    ap = argparse.ArgumentParser(description="Deteccion de riesgo de inundacion en video")
    ap.add_argument("video", help="ruta del archivo de video (.mp4, .avi, .mov...)")
    ap.add_argument("--config", default="station.json", help="JSON de calibracion")
    ap.add_argument("--calibrate", action="store_true", help="calibrar con clics y salir")
    ap.add_argument("--station", default="01")
    ap.add_argument("--csv", default="niveles.csv")
    ap.add_argument("--record", default=None, help="mp4 de salida con el HUD")
    ap.add_argument("--webhook", default=None, help="URL para POSTear alertas")
    ap.add_argument("--headless", action="store_true", help="sin ventana")
    ap.add_argument("--start", type=float, default=0.0, help="segundo inicial")
    ap.add_argument("--end", type=float, default=None, help="segundo final")
    ap.add_argument("--stride", type=int, default=1, help="procesa 1 de cada N cuadros")
    ap.add_argument("--log-every", type=float, default=2.0, help="segundos de video entre filas del CSV")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="0 = lo mas rapido posible; 1 = tiempo real; 2 = doble")
    ap.add_argument("--warmup", type=float, default=3.0,
                    help="segundos iniciales que se miden pero NO generan alertas "
                         "(el modelo de fondo todavia esta aprendiendo la escena)")
    args = ap.parse_args()

    # ---- abrir el video ---- #
    if not os.path.exists(args.video):
        sys.exit(f"No existe el archivo: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"OpenCV no pudo abrir el video: {args.video}")

    # Datos del video: cuadros por segundo, total de cuadros y duracion
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n_frames / fps_in if n_frames else 0.0
    # --start: salta directamente a ese segundo del video
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    # Se lee un primer cuadro para conocer el tamano de la imagen
    ok, frame = cap.read()
    if not ok:
        sys.exit("No pude leer el primer cuadro.")
    H, W = frame.shape[:2]
    print(f"[video] {args.video}  {W}x{H}  {fps_in:.2f} FPS  "
          f"{n_frames} cuadros  {timedelta(seconds=int(dur))}")

    # Modo calibracion: se marca el ROI y las referencias, se guarda y se termina
    if args.calibrate:
        run_calibration(frame, args.config)
        cap.release()
        return

    cfg = load_config(args.config, frame.shape)

    # ---- preparar el ROI ---- #
    # Poligono del cauce -> mascara (255 dentro, 0 fuera)
    roi_poly = np.array(cfg["roi"], np.int32)
    roi_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly], 255)
    # Rectangulo que encierra al poligono: limita la busqueda de la linea de agua
    bx, by, bw, bh = cv2.boundingRect(roi_poly)
    roi_box = (bx, by, min(bx + bw, W), min(by + bh, H))
    # Area del ROI en pixeles (para calcular que fraccion cubre el agua)
    roi_area = float(max(cv2.countNonZero(roi_mask), 1))

    # Las tres piezas del sistema: segmentar, calibrar y llevar el estado del rio
    seg = WaterSegmenter(cfg)
    cal = Calibration(cfg["ref_points"])
    state = RiverState(cfg)

    # ---- video de salida (opcional, con --record) ---- #
    writer = None
    if args.record:
        # "mp4v" = codec MPEG-4. Si se salta cuadros (stride), se ajustan los FPS
        # para que el video resultante dure lo mismo que el original.
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in / max(args.stride, 1), (W, H))
        if not writer.isOpened():
            print("[aviso] no pude abrir el VideoWriter; sigo sin grabar.")
            writer = None

    # ---- CSV de resultados ---- #
    # Se abre en modo "a" (append): cada corrida AGREGA filas al final.
    # El encabezado solo se escribe si el archivo es nuevo.
    new_csv = not os.path.exists(args.csv)
    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if new_csv:
        csv_w.writerow(["t_video_s", "timestamp", "estacion", "nivel_m",
                        "caudal_m3s", "tendencia_m_h", "riesgo", "confianza"])

    # ---- variables del bucle principal ---- #
    vis = frame.copy()                  # ultimo cuadro dibujado (se muestra si esta en pausa)
    idx = 0                             # cuadros leidos
    processed = 0                       # cuadros realmente analizados
    last_log = -1e9                     # ultimo segundo escrito en el CSV
    last_risk = "NORMAL"                # riesgo anterior, para detectar cambios
    warmed = False                      # ya termino el calentamiento?
    paused = False
    events: list[tuple[float, str, float]] = []   # cambios de riesgo (para el resumen)
    t0_wall = time.time()               # hora real de inicio (para --speed)
    t_video = args.start

    print("Procesando...  (q = salir)")
    try:
        while True:
            if not paused:
                if idx > 0:                       # el primer cuadro ya esta leido
                    ok, frame = cap.read()
                    if not ok:
                        print("\nFin del video.")
                        break
                idx += 1

                # tiempo del VIDEO: la referencia temporal de todo el analisis
                # (si el video no informa su posicion, se calcula con los FPS)
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_video = pos / 1000.0 if pos and pos > 0 else args.start + idx / fps_in
                if args.end is not None and t_video > args.end:
                    print("\nLlegue al segundo final pedido.")
                    break
                # --stride N: analiza solo 1 de cada N cuadros (mas rapido)
                if args.stride > 1 and (idx - 1) % args.stride:
                    continue
                processed += 1

                # ======= EL PIPELINE, en 5 lineas ======= #
                # 1) mascara de agua
                mask = seg.update(frame, roi_mask)
                # 2) linea de agua: puntos, altura mediana (px) y dispersion
                pts, y_med, mad = water_line(mask, roi_box)
                # 3) fraccion del ROI cubierta por agua (0..1)
                coverage = cv2.countNonZero(mask) / roi_area
                # 4) pixeles -> metros
                level = cal.to_meters(y_med) if y_med is not None else None
                # 5) actualizar nivel suavizado, tendencia, confianza y riesgo
                #    (sin linea de agua se pasa una dispersion enorme = confianza 0)
                state.update(level, mad if mad is not None else 99.0, coverage, t_video)

                # arranque: se mide pero no se alerta (el fondo aun se esta aprendiendo)
                warming = (t_video - args.start) < args.warmup

                # Dibuja el HUD y, si corresponde, lo graba en el video de salida
                progress = (t_video / dur) if dur else None
                vis = draw_hud(frame, mask, pts, roi_poly, state,
                               args.station, t_video, progress, warming)
                if writer:
                    writer.write(vis)

                # ---- deteccion de cambios de riesgo ---- #
                if warming:
                    last_risk = state.risk      # arrastra sin emitir eventos falsos
                    warmed = False
                elif not warmed:                # primer cuadro util: borra el arranque
                    # Sin esto, la tendencia se calcula con los datos locos del
                    # arranque y puede dar valores absurdos (ej. -4382 m/h).
                    state.reset_history()
                    last_risk = state.risk
                    warmed = True
                elif state.risk != last_risk:
                    # El riesgo cambio: se registra el evento
                    events.append((t_video, state.risk, state.level or 0.0))
                    # Si es ALERTA o peor: se imprime y se envia el webhook
                    if RISK_ORDER.index(state.risk) >= RISK_ORDER.index("ALERTA"):
                        print(f"\n[t={timedelta(seconds=int(t_video))}] *** {state.risk} *** "
                              f"nivel={state.level:.2f} m  tendencia={state.trend:+.2f} m/h")
                        send_alert(args.webhook, {
                            "estacion": args.station, "riesgo": state.risk,
                            "t_video_s": round(t_video, 2),
                            "nivel_m": round(state.level or 0.0, 2),
                            "tendencia_m_h": round(state.trend, 2),
                            "ts": datetime.now().isoformat()})
                    last_risk = state.risk

                # ---- escribir una fila en el CSV cada --log-every segundos de video ---- #
                if (state.level is not None and not warming
                        and t_video - last_log >= args.log_every):
                    last_log = t_video
                    csv_w.writerow([round(t_video, 2),
                                    datetime.now().isoformat(timespec="seconds"),
                                    args.station, round(state.level, 3),
                                    round(state.flow, 3), round(state.trend, 3),
                                    state.risk, round(state.confidence, 3)])
                    csv_f.flush()               # guarda en disco ya (por si se corta)

                # Sin ventana: muestra el progreso en la consola cada 25 cuadros.
                # "\r" vuelve al inicio de la linea, asi se sobrescribe en el mismo lugar.
                if args.headless and processed % 25 == 0:
                    pct = f"{progress * 100:5.1f}%" if progress else f"{processed} cuadros"
                    sys.stdout.write(
                        f"\r  {pct}  t={timedelta(seconds=int(t_video))}  "
                        f"nivel={state.level if state.level is None else round(state.level, 2)} m  "
                        f"{state.risk:<18}")
                    sys.stdout.flush()

                if args.speed > 0:                # reproducir a velocidad controlada
                    # Si vamos adelantados respecto al tiempo real, se espera un poco
                    target = (t_video - args.start) / args.speed
                    lag = target - (time.time() - t0_wall)
                    if lag > 0:
                        time.sleep(min(lag, 0.25))

            # ---- ventana y teclado ---- #
            if not args.headless:
                try:
                    cv2.imshow("Monitor de inundacion", vis)
                except cv2.error:
                    # OpenCV instalado sin soporte de ventanas (servidor, Colab)
                    print("[aviso] este OpenCV no tiene ventanas; sigo en modo headless.")
                    args.headless = True
                    continue
                # waitKey refresca la ventana y lee una tecla (espera 1 ms)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break                           # salir
                if k == ord(" "):
                    paused = not paused             # pausa / continua
                if k == ord("s"):
                    name = f"captura_{int(t_video)}s.png"
                    cv2.imwrite(name, vis)          # guarda captura PNG
                    print("Guardado", name)
                if k == ord("r"):
                    # reinicia el historial de nivel
                    state.hist.clear()
                    state.level = None
                    state.peak = None
    except KeyboardInterrupt:
        # Ctrl+C: se sale limpio (el bloque finally cierra todo igual)
        print("\nInterrumpido.")
    finally:
        # Se ejecuta SIEMPRE, aunque haya error: libera el video y cierra archivos
        cap.release()
        if writer:
            writer.release()
        csv_f.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass          # build de OpenCV sin GUI (servidor)

    # ------------------------- resumen -------------------------
    # Estadisticas finales impresas en la consola
    print("\n" + "=" * 62)
    print(f"Cuadros procesados : {processed}")
    print(f"Tiempo de video    : {timedelta(seconds=int(t_video))}")
    if state.hist:
        niveles = [h for _, h in state.hist]
        print(f"Nivel min/med/max  : {min(niveles):.2f} / "
              f"{float(np.mean(niveles)):.2f} / {max(niveles):.2f} m")
        print(f"Riesgo final       : {state.risk}  (confianza {state.confidence*100:.1f} %)")
    if events:
        print("Cambios de riesgo  :")
        for t, r, h in events:
            print(f"   {str(timedelta(seconds=int(t))):>8}  ->  {r:<18} ({h:.2f} m)")
    print(f"CSV                : {args.csv}")
    if args.record:
        print(f"Video anotado      : {args.record}")
    print("=" * 62)


# Solo se ejecuta main() si el archivo se corre directamente
# (no cuando se importa desde otro script)
if __name__ == "__main__":
    main()
