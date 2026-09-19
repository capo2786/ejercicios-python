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

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Paleta (BGR)
# --------------------------------------------------------------------------- #
CYAN = (230, 220, 60)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
DARK = (35, 28, 20)
RISK_COLORS = {
    "NORMAL": (120, 220, 0),
    "VIGILANCIA": (0, 220, 255),
    "ALERTA": (0, 140, 255),
    "DESBORDE CRITICO": (60, 60, 255),
}
RISK_ORDER = ["NORMAL", "VIGILANCIA", "ALERTA", "DESBORDE CRITICO"]

DEFAULT_CONFIG = {
    "roi": None,                  # [[x,y], ...] poligono del cauce
    "ref_points": None,           # [[x1,y1,h1_m], [x2,y2,h2_m]]
    "thresholds": {"vigilancia": 1.8, "alerta": 2.6, "critico": 3.2},
    "rating": {"c": 2.1, "h0": 0.15, "exp": 1.5},      # Q = c*(h-h0)^exp
    "weights": {"motion": 0.62, "color": 0.38, "bin": 0.38},
    "hsv_low": [70, 10, 25],
    "hsv_high": [135, 190, 235],
    "smooth_alpha": 0.18,         # EMA del nivel
    "trend_window_s": 25.0,       # ventana (en segundos de video) para la tendencia
    "rise_rate_alert": 0.35,      # m/h a partir del cual se escala el riesgo
}


# --------------------------------------------------------------------------- #
# 1. Segmentacion del agua
# --------------------------------------------------------------------------- #
class WaterSegmenter:
    """Mascara de agua = movimiento persistente + color, dentro del ROI."""

    def __init__(self, cfg: dict) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=20, detectShadows=False
        )
        self.motion_acc: np.ndarray | None = None
        self.w = cfg["weights"]
        self.lo = np.array(cfg["hsv_low"], dtype=np.uint8)
        self.hi = np.array(cfg["hsv_high"], dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def update(self, frame: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        # energia de movimiento acumulada: el agua "titila" en todos los cuadros
        fg = self.bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        motion = (fg > 0).astype(np.float32)
        if self.motion_acc is None:
            self.motion_acc = motion
        else:
            self.motion_acc = 0.90 * self.motion_acc + 0.10 * motion
        motion_s = cv2.GaussianBlur(self.motion_acc, (21, 21), 0)
        motion_s = np.clip(motion_s / 0.35, 0.0, 1.0)

        # color tipico del agua (azul-verdoso / gris, saturacion baja-media)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = cv2.inRange(hsv, self.lo, self.hi).astype(np.float32) / 255.0
        color = cv2.GaussianBlur(color, (15, 15), 0)

        score = self.w["motion"] * motion_s + self.w["color"] * color
        mask = (score > self.w["bin"]).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, roi_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        return self._largest_blob(mask)

    @staticmethod
    def _largest_blob(mask: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return np.where(labels == idx, 255, 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 2. Linea de agua (borde superior de la lamina)
# --------------------------------------------------------------------------- #
def water_line(mask: np.ndarray, roi_box: tuple, min_run: int = 6, step: int = 4):
    """Para cada columna busca la primera racha continua de agua.

    Devuelve (puntos, y_mediana, MAD). La racha minima evita que un reflejo
    suelto o una gota en el lente se tome como superficie del agua.
    """
    x0, y0, x1, y1 = roi_box
    pts: list[tuple[int, int]] = []

    for x in range(x0, x1, step):
        col = mask[y0:y1, x] > 0
        if not col.any():
            continue
        # posiciones donde arranca cada racha y su longitud (vectorizado)
        d = np.diff(np.concatenate(([0], col.view(np.uint8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        runs = ends - starts
        ok = np.flatnonzero(runs >= min_run)
        if ok.size:
            pts.append((x, y0 + int(starts[ok[0]])))

    if not pts:
        return [], None, None

    ys = np.array([p[1] for p in pts], dtype=np.float32)
    med = float(np.median(ys))
    mad = float(np.median(np.abs(ys - med)))
    tol = max(3.0 * mad, 8.0)
    keep = [p for p in pts if abs(p[1] - med) <= tol]
    if keep:
        med = float(np.median([p[1] for p in keep]))
    return keep, med, mad


# --------------------------------------------------------------------------- #
# 3. Calibracion pixel -> metro
# --------------------------------------------------------------------------- #
class Calibration:
    """Escala lineal a partir de 2 puntos de altura conocida en la imagen."""

    def __init__(self, ref_points) -> None:
        (x1, y1, h1), (x2, y2, h2) = ref_points
        if abs(float(y2) - float(y1)) < 1e-6:
            raise ValueError("Los 2 puntos de referencia no pueden estar a la misma altura en pixeles.")
        self.y1, self.h1 = float(y1), float(h1)
        self.m_per_px = (float(h2) - float(h1)) / (float(y2) - float(y1))

    def to_meters(self, y_px: float) -> float:
        return self.h1 + (float(y_px) - self.y1) * self.m_per_px


# --------------------------------------------------------------------------- #
# 4. Estado hidrologico
# --------------------------------------------------------------------------- #
class RiverState:
    def __init__(self, cfg: dict) -> None:
        self.th = cfg["thresholds"]
        self.rt = cfg["rating"]
        self.alpha = cfg["smooth_alpha"]
        self.window = cfg["trend_window_s"]
        self.rise_alert = cfg["rise_rate_alert"]
        self.level: float | None = None
        self.hist: deque = deque(maxlen=4000)     # (t_video_s, nivel_m)
        self.conf_hist: deque = deque(maxlen=30)
        self.risk = "NORMAL"
        self.peak: float | None = None

    def update(self, level_raw, mad_px, coverage, t_video):
        if level_raw is None:
            self.conf_hist.append(0.0)
            return
        self.level = (level_raw if self.level is None
                      else (1 - self.alpha) * self.level + self.alpha * level_raw)
        self.hist.append((t_video, self.level))
        self.peak = self.level if self.peak is None else max(self.peak, self.level)

        c_line = float(np.clip(1.0 - (mad_px / 25.0), 0.0, 1.0))   # linea consistente
        c_cov = float(np.clip(coverage / 0.25, 0.0, 1.0))          # ROI bien cubierto
        self.conf_hist.append(0.65 * c_line + 0.35 * c_cov)
        self.risk = self._classify()

    def reset_history(self) -> None:
        """Descarta las muestras del arranque, conservando el nivel suavizado."""
        self.hist.clear()
        self.conf_hist.clear()
        self.peak = self.level
        self.risk = "NORMAL"

    @property
    def confidence(self) -> float:
        return float(np.mean(self.conf_hist)) if self.conf_hist else 0.0

    @property
    def trend(self) -> float:
        """m/h — pendiente por minimos cuadrados sobre segundos DE VIDEO."""
        if len(self.hist) < 8:
            return 0.0
        t_now = self.hist[-1][0]
        pts = [(t, h) for t, h in self.hist if t_now - t <= self.window]
        if len(pts) < 8:
            return 0.0
        t = np.array([p[0] for p in pts], dtype=np.float64)
        h = np.array([p[1] for p in pts], dtype=np.float64)
        if t.max() - t.min() < 1e-6:
            return 0.0
        slope = float(np.polyfit(t - t[0], h, 1)[0])
        return slope * 3600.0

    @property
    def flow(self) -> float:
        if self.level is None:
            return 0.0
        head = max(self.level - self.rt["h0"], 0.0)
        return float(self.rt["c"] * head ** self.rt["exp"])

    def _classify(self) -> str:
        h, tr = self.level, self.trend
        if h >= self.th["critico"]:
            r = "DESBORDE CRITICO"
        elif h >= self.th["alerta"]:
            r = "ALERTA"
        elif h >= self.th["vigilancia"]:
            r = "VIGILANCIA"
        else:
            r = "NORMAL"
        if tr > self.rise_alert and r != "DESBORDE CRITICO":   # sube rapido -> escala
            r = RISK_ORDER[min(RISK_ORDER.index(r) + 1, 3)]
        return r


# --------------------------------------------------------------------------- #
# 5. HUD
# --------------------------------------------------------------------------- #
def panel(img, x, y, w, h, title, rows, accent=CYAN):
    H, W = img.shape[:2]
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        return
    sub = img[y:y + h, x:x + w]
    cv2.addWeighted(np.full(sub.shape, DARK, np.uint8), 0.62, sub, 0.38, 0, sub)
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 1)
    cv2.rectangle(img, (x, y), (x + w, y + 22), accent, -1)
    cv2.putText(img, title, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DARK, 1, cv2.LINE_AA)
    for i, (k, v) in enumerate(rows):
        yy = y + 42 + i * 20
        if yy > y + h - 4:
            break
        cv2.putText(img, f"{k}:", (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, accent, 1, cv2.LINE_AA)
        cv2.putText(img, str(v), (x + 122, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1, cv2.LINE_AA)


def sparkline(img, x, y, w, h, hist, color=CYAN):
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    if len(hist) < 3:
        return
    vals = np.array([v for _, v in hist][-w:], dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    rng = max(hi - lo, 0.08)
    xs = np.linspace(x, x + w, len(vals)).astype(np.int32)
    ys = (y + h - (vals - lo) / rng * h).astype(np.int32)
    cv2.polylines(img, [np.stack([xs, ys], 1)], False, color, 1, cv2.LINE_AA)


def draw_hud(frame, mask, line_pts, roi_poly, state, station, t_video,
             progress, warming=False):
    out = frame.copy()
    tint = np.zeros_like(out)
    tint[mask > 0] = (140, 90, 20)
    cv2.addWeighted(tint, 0.30, out, 0.70, 0, out)

    if roi_poly is not None:
        cv2.polylines(out, [roi_poly], True, GREY, 1)
    if len(line_pts) > 1:
        cv2.polylines(out, [np.array(line_pts, np.int32)], False, CYAN, 2, cv2.LINE_AA)

    color = CYAN if warming else RISK_COLORS.get(state.risk, CYAN)
    lvl = "--" if state.level is None else f"{state.level:.2f} m"
    tr = state.trend
    arrow = "SUBIENDO" if tr > 0.05 else ("BAJANDO" if tr < -0.05 else "ESTABLE")

    panel(out, 14, 14, 306, 166, f"ESTACION {station}  |  ANALISIS DE VIDEO", [
        ("Riesgo", "CALIBRANDO..." if warming else state.risk),
        ("Confianza", f"{state.confidence * 100:.1f} %"),
        ("Nivel", lvl),
        ("Tendencia", f"{arrow} ({tr:+.2f} m/h)"),
        ("Caudal est.", f"{state.flow:.2f} m3/s"),
        ("Nivel max.", "--" if state.peak is None else f"{state.peak:.2f} m"),
    ], accent=color)

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
    cv2.putText(out, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if not warming and state.risk in ("ALERTA", "DESBORDE CRITICO"):
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), color, 6)
    return out


# --------------------------------------------------------------------------- #
# 6. Calibracion interactiva sobre un cuadro del video
# --------------------------------------------------------------------------- #
def run_calibration(frame, path):
    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow("calibracion", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibracion", on_mouse)
    print("\n[1] Clic en los vertices del ROI (el cauce). ENTER para cerrarlo.")
    print("[2] Luego 2 clics en puntos de altura conocida (regleta, pilar, borde).")
    print("    Clic derecho deshace. ESC cancela.\n")

    roi = None
    refs: list[tuple[int, int]] = []
    while True:
        vis = frame.copy()
        for i, p in enumerate(clicks):
            cv2.circle(vis, p, 5, CYAN, -1)
            cv2.putText(vis, str(i + 1), (p[0] + 8, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
        if roi is None and len(clicks) > 1:
            cv2.polylines(vis, [np.array(clicks, np.int32)], False, CYAN, 1)
        if roi is not None:
            cv2.polylines(vis, [np.array(roi, np.int32)], True, GREY, 1)
        msg = "ROI: clic vertices + ENTER" if roi is None else "REFERENCIAS: 2 clics + ENTER"
        cv2.putText(vis, msg, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.imshow("calibracion", vis)

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit("Calibracion cancelada.")
        if k in (13, 10):
            if roi is None:
                if len(clicks) < 3:
                    print("Necesitas al menos 3 vertices.")
                    continue
                roi = [list(p) for p in clicks]
                clicks.clear()
            else:
                if len(clicks) != 2:
                    print("Necesitas exactamente 2 puntos de referencia.")
                    continue
                refs = list(clicks)
                break
    cv2.destroyAllWindows()

    h1 = float(input("Altura real del punto 1 (m): ").strip())
    h2 = float(input("Altura real del punto 2 (m): ").strip())
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    cfg["roi"] = roi
    cfg["ref_points"] = [[refs[0][0], refs[0][1], h1], [refs[1][0], refs[1][1], h2]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {path}. Ahora corre sin --calibrate.")
    return cfg


def load_config(path, frame_shape):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg.update(user)
        print(f"[config] {path}")
        return cfg
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
def send_alert(webhook, payload):
    if not webhook:
        return
    try:
        req = urllib.request.Request(
            webhook, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:      # noqa: BLE001
        print(f"[alerta] fallo el webhook: {exc}")


# --------------------------------------------------------------------------- #
# 8. Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
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

    if not os.path.exists(args.video):
        sys.exit(f"No existe el archivo: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"OpenCV no pudo abrir el video: {args.video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n_frames / fps_in if n_frames else 0.0
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    ok, frame = cap.read()
    if not ok:
        sys.exit("No pude leer el primer cuadro.")
    H, W = frame.shape[:2]
    print(f"[video] {args.video}  {W}x{H}  {fps_in:.2f} FPS  "
          f"{n_frames} cuadros  {timedelta(seconds=int(dur))}")

    if args.calibrate:
        run_calibration(frame, args.config)
        cap.release()
        return

    cfg = load_config(args.config, frame.shape)

    roi_poly = np.array(cfg["roi"], np.int32)
    roi_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly], 255)
    bx, by, bw, bh = cv2.boundingRect(roi_poly)
    roi_box = (bx, by, min(bx + bw, W), min(by + bh, H))
    roi_area = float(max(cv2.countNonZero(roi_mask), 1))

    seg = WaterSegmenter(cfg)
    cal = Calibration(cfg["ref_points"])
    state = RiverState(cfg)

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in / max(args.stride, 1), (W, H))
        if not writer.isOpened():
            print("[aviso] no pude abrir el VideoWriter; sigo sin grabar.")
            writer = None

    new_csv = not os.path.exists(args.csv)
    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if new_csv:
        csv_w.writerow(["t_video_s", "timestamp", "estacion", "nivel_m",
                        "caudal_m3s", "tendencia_m_h", "riesgo", "confianza"])

    vis = frame.copy()
    idx = 0
    processed = 0
    last_log = -1e9
    last_risk = "NORMAL"
    warmed = False
    paused = False
    events: list[tuple[float, str, float]] = []
    t0_wall = time.time()
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
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_video = pos / 1000.0 if pos and pos > 0 else args.start + idx / fps_in
                if args.end is not None and t_video > args.end:
                    print("\nLlegue al segundo final pedido.")
                    break
                if args.stride > 1 and (idx - 1) % args.stride:
                    continue
                processed += 1

                mask = seg.update(frame, roi_mask)
                pts, y_med, mad = water_line(mask, roi_box)
                coverage = cv2.countNonZero(mask) / roi_area
                level = cal.to_meters(y_med) if y_med is not None else None
                state.update(level, mad if mad is not None else 99.0, coverage, t_video)

                # arranque: se mide pero no se alerta (el fondo aun se esta aprendiendo)
                warming = (t_video - args.start) < args.warmup

                progress = (t_video / dur) if dur else None
                vis = draw_hud(frame, mask, pts, roi_poly, state,
                               args.station, t_video, progress, warming)
                if writer:
                    writer.write(vis)

                if warming:
                    last_risk = state.risk      # arrastra sin emitir eventos falsos
                    warmed = False
                elif not warmed:                # primer cuadro util: borra el arranque
                    state.reset_history()
                    last_risk = state.risk
                    warmed = True
                elif state.risk != last_risk:
                    events.append((t_video, state.risk, state.level or 0.0))
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

                if (state.level is not None and not warming
                        and t_video - last_log >= args.log_every):
                    last_log = t_video
                    csv_w.writerow([round(t_video, 2),
                                    datetime.now().isoformat(timespec="seconds"),
                                    args.station, round(state.level, 3),
                                    round(state.flow, 3), round(state.trend, 3),
                                    state.risk, round(state.confidence, 3)])
                    csv_f.flush()

                if args.headless and processed % 25 == 0:
                    pct = f"{progress * 100:5.1f}%" if progress else f"{processed} cuadros"
                    sys.stdout.write(
                        f"\r  {pct}  t={timedelta(seconds=int(t_video))}  "
                        f"nivel={state.level if state.level is None else round(state.level, 2)} m  "
                        f"{state.risk:<18}")
                    sys.stdout.flush()

                if args.speed > 0:                # reproducir a velocidad controlada
                    target = (t_video - args.start) / args.speed
                    lag = target - (time.time() - t0_wall)
                    if lag > 0:
                        time.sleep(min(lag, 0.25))

            if not args.headless:
                try:
                    cv2.imshow("Monitor de inundacion", vis)
                except cv2.error:
                    print("[aviso] este OpenCV no tiene ventanas; sigo en modo headless.")
                    args.headless = True
                    continue
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    paused = not paused
                if k == ord("s"):
                    name = f"captura_{int(t_video)}s.png"
                    cv2.imwrite(name, vis)
                    print("Guardado", name)
                if k == ord("r"):
                    state.hist.clear()
                    state.level = None
                    state.peak = None
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
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


if __name__ == "__main__":
    main()
