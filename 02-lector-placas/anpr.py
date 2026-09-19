#!/usr/bin/env python3
"""
anpr.py — Lectura automatica de placas vehiculares (ANPR/ALPR) en UN SOLO SCRIPT
=================================================================================

Lee un video (o webcam/RTSP) y, cuadro a cuadro:

  1. DETECTA placas candidatas con dos estrategias combinadas:
       a) cascade Haar de placas (viene incluido en OpenCV)
       b) morfologia blackhat + Sobel + cierre -> contornos con forma de placa
  2. SIGUE cada placa entre cuadros con un rastreador por centroide (le asigna ID).
  3. CORRIGE la perspectiva del recorte y lo binariza (CLAHE + Otsu + deskew).
  4. Hace OCR del recorte (Tesseract por defecto; EasyOCR si lo tienes).
  5. VOTA POR CARACTER entre todas las lecturas del mismo vehiculo.
  6. VALIDA el formato (Ecuador por defecto) y exporta CSV + recortes + video.

Por que la votacion importa: el OCR de un solo cuadro se equivoca seguido
(confunde 0/O, 1/I, 8/B, 5/S). Como el mismo auto aparece en 20-40 cuadros, se
toma el caracter mas votado en cada posicion. Eso sube la exactitud muchisimo
mas que cualquier ajuste de preprocesado sobre un cuadro suelto.

Instalacion:
    pip install opencv-python numpy pytesseract
    sudo apt install tesseract-ocr          # Debian/Ubuntu
    # opcional, mejor precision pero pesado:  pip install easyocr

Uso:
    python anpr.py patio.mp4
    python anpr.py patio.mp4 --record salida.mp4 --crops recortes/
    python anpr.py patio.mp4 --ocr easyocr --formato ecuador
    python anpr.py 0 --headless                      # webcam
    python anpr.py patio.mp4 --roi                   # dibujar zona de lectura

Teclas: q salir | espacio pausa | s captura PNG
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Paleta (BGR)
# --------------------------------------------------------------------------- #
CYAN = (230, 220, 60)
GREEN = (120, 220, 0)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
DARK = (35, 28, 20)

# Formatos de placa. Se valida sobre el texto SIN guiones ni espacios.
FORMATOS = {
    # Ecuador: 3 letras + 3 o 4 numeros (ABC1234 / ABC123)
    "ecuador": r"^[A-Z]{3}[0-9]{3,4}$",
    # Colombia / Peru: ABC123 o ABC12D
    "andino": r"^[A-Z]{3}[0-9]{2,4}[A-Z]?$",
    # Mercosur (Argentina/Brasil): AB123CD
    "mercosur": r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$",
    # Generico: 5 a 8 alfanumericos con al menos una letra y un numero
    "libre": r"^(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{5,8}$",
}

# Confusiones tipicas del OCR, aplicadas segun la POSICION esperada del caracter
A_LETRA = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
A_NUMERO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "G": "6", "B": "8", "A": "4"}


# --------------------------------------------------------------------------- #
# 1. Deteccion de placas
# --------------------------------------------------------------------------- #
class PlateDetector:
    """Dos detectores complementarios; se fusionan sus cajas por IoU."""

    def __init__(self, min_ratio=1.8, max_ratio=6.0, min_area=900, use_cascade=True):
        self.min_ratio, self.max_ratio, self.min_area = min_ratio, max_ratio, min_area
        self.cascade = None
        if use_cascade:
            path = os.path.join(cv2.data.haarcascades,
                                "haarcascade_russian_plate_number.xml")
            if os.path.exists(path):
                c = cv2.CascadeClassifier(path)
                if not c.empty():
                    self.cascade = c
        self.rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        self.sq_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # -- a) cascade Haar ----------------------------------------------------- #
    def _by_cascade(self, gray):
        if self.cascade is None:
            return []
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.08,
                                              minNeighbors=5, minSize=(60, 20))
        return [tuple(map(int, r)) for r in found]

    # -- b) morfologia ------------------------------------------------------- #
    def _by_morphology(self, gray):
        # las placas son texto oscuro sobre fondo claro => blackhat lo resalta
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rect_k)
        grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad = np.absolute(grad)
        mn, mx = grad.min(), grad.max()
        if mx - mn < 1e-6:
            return []
        grad = (255 * (grad - mn) / (mx - mn)).astype(np.uint8)

        grad = cv2.GaussianBlur(grad, (5, 5), 0)
        grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, self.rect_k)
        _, th = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, self.rect_k, iterations=2)
        th = cv2.erode(th, self.sq_k, iterations=1)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or w * h < self.min_area:
                continue
            if self.min_ratio <= w / h <= self.max_ratio and h >= 14:
                out.append((x, y, w, h))
        return out

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
        boxes = self._by_cascade(gray) + self._by_morphology(gray)
        return self._merge(boxes)

    @staticmethod
    def _merge(boxes, iou_thr=0.35):
        """Funde cajas solapadas (no-max suppression simple por area)."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        keep = []
        for b in boxes:
            if all(PlateDetector._iou(b, k) < iou_thr for k in keep):
                keep.append(b)
        return keep

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# 2. Preprocesado del recorte
# --------------------------------------------------------------------------- #
def deskew(binary):
    """Endereza la placa usando el angulo del rectangulo minimo del texto."""
    pts = cv2.findNonZero(255 - binary)
    if pts is None or len(pts) < 20:
        return binary
    angle = cv2.minAreaRect(pts)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.6 or abs(angle) > 20:      # ruido o giro imposible
        return binary
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preparar_recorte(crop, alto_objetivo=64):
    """Recorte BGR -> imagen binaria lista para OCR (texto negro, fondo blanco)."""
    if crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if h < 8 or w < 20:
        return None
    escala = max(alto_objetivo / h, 1.0)
    crop = cv2.resize(crop, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 60, 60)
    gray = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)

    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if np.mean(th) < 127:                 # asegura fondo claro
        th = cv2.bitwise_not(th)
    th = deskew(th)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


# --------------------------------------------------------------------------- #
# 3. Motores de OCR
# --------------------------------------------------------------------------- #
class OCR:
    """Envoltura sobre Tesseract o EasyOCR. Devuelve (texto, confianza 0-1)."""

    WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, motor="auto"):
        self.motor = None
        if motor in ("auto", "easyocr"):
            try:
                import easyocr                                    # noqa: PLC0415
                self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                self.motor = "easyocr"
            except Exception:                                     # noqa: BLE001
                if motor == "easyocr":
                    sys.exit("EasyOCR no esta instalado: pip install easyocr")
        if self.motor is None and motor in ("auto", "tesseract"):
            try:
                import pytesseract                                # noqa: PLC0415
                pytesseract.get_tesseract_version()
                self.pt = pytesseract
                self.motor = "tesseract"
            except Exception as exc:                              # noqa: BLE001
                sys.exit(f"No hay OCR disponible ({exc}).\n"
                         "  pip install pytesseract && sudo apt install tesseract-ocr")
        if self.motor is None:
            sys.exit("No se pudo inicializar ningun motor de OCR.")
        print(f"[ocr] motor: {self.motor}")

    def leer(self, binaria):
        if binaria is None:
            return "", 0.0
        if self.motor == "tesseract":
            return self._tesseract(binaria)
        return self._easyocr(binaria)

    def _tesseract(self, img):
        cfg = (f"--oem 3 --psm 7 -c tessedit_char_whitelist={self.WHITELIST}")
        try:
            data = self.pt.image_to_data(img, config=cfg,
                                         output_type=self.pt.Output.DICT)
        except Exception:                                         # noqa: BLE001
            return "", 0.0
        partes, confs = [], []
        for txt, cf in zip(data["text"], data["conf"]):
            txt = re.sub(r"[^A-Z0-9]", "", txt.upper())
            try:
                cf = float(cf)
            except (TypeError, ValueError):
                cf = -1.0
            if txt and cf >= 0:
                partes.append(txt)
                confs.append(cf)
        if not partes:
            return "", 0.0
        return "".join(partes), float(np.mean(confs)) / 100.0

    def _easyocr(self, img):
        try:
            res = self.reader.readtext(img, allowlist=self.WHITELIST, detail=1)
        except Exception:                                         # noqa: BLE001
            return "", 0.0
        if not res:
            return "", 0.0
        texto = "".join(re.sub(r"[^A-Z0-9]", "", r[1].upper()) for r in res)
        conf = float(np.mean([r[2] for r in res]))
        return texto, conf


# --------------------------------------------------------------------------- #
# 4. Rastreo + votacion por caracter
# --------------------------------------------------------------------------- #
class Track:
    """Una placa seguida a lo largo del tiempo, con sus lecturas acumuladas."""

    _next_id = 1

    def __init__(self, box, t):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.t_inicio = self.t_visto = t
        self.perdido = 0
        self.lecturas: list[tuple[str, float]] = []
        self.votos: dict[int, Counter] = defaultdict(Counter)
        self.mejor_crop = None
        self.mejor_conf = 0.0
        self.reportado = False

    @property
    def centro(self):
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    def agregar(self, texto, conf, crop):
        if not texto:
            return
        self.lecturas.append((texto, conf))
        for i, ch in enumerate(texto):
            self.votos[i][ch] += conf            # el voto pesa por confianza
        if conf > self.mejor_conf:
            self.mejor_conf, self.mejor_crop = conf, crop

    def consenso(self, formato_re=None):
        """Texto mas votado por posicion + correccion de confusiones por posicion."""
        if not self.lecturas:
            return "", 0.0
        largo = Counter(len(t) for t, _ in self.lecturas).most_common(1)[0][0]
        chars, pesos = [], []
        for i in range(largo):
            if not self.votos[i]:
                continue
            ch, peso = self.votos[i].most_common(1)[0]
            total = sum(self.votos[i].values())
            chars.append(ch)
            pesos.append(peso / total if total else 0.0)
        texto = "".join(chars)
        if formato_re is not None:
            texto = corregir_por_formato(texto, formato_re)
        conf = float(np.mean(pesos)) if pesos else 0.0
        # mas lecturas coincidentes => mas confianza (tope en 12 lecturas)
        conf *= min(1.0, 0.55 + 0.45 * min(len(self.lecturas), 12) / 12)
        return texto, conf


def corregir_por_formato(texto, formato_re):
    """Si el texto casi cumple el formato, arregla confusiones O/0, I/1, S/5..."""
    if not texto or formato_re.match(texto):
        return texto
    # patron esperado: cuantas letras iniciales y cuantos digitos despues
    m = re.match(r"\^\[A-Z\]\{(\d+)\}", formato_re.pattern)
    n_letras = int(m.group(1)) if m else 3
    arreglado = []
    for i, ch in enumerate(texto):
        if i < n_letras:
            arreglado.append(A_LETRA.get(ch, ch))
        else:
            arreglado.append(A_NUMERO.get(ch, ch))
    cand = "".join(arreglado)
    return cand if formato_re.match(cand) else texto


class Tracker:
    """Asociacion por cercania de centroides (suficiente para placas en video)."""

    def __init__(self, dist_max=90, max_perdido=12):
        self.dist_max, self.max_perdido = dist_max, max_perdido
        self.tracks: list[Track] = []

    def actualizar(self, boxes, t):
        for tr in self.tracks:
            tr.perdido += 1
        for box in boxes:
            cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
            mejor, mejor_d = None, self.dist_max
            for tr in self.tracks:
                tx, ty = tr.centro
                d = float(np.hypot(cx - tx, cy - ty))
                if d < mejor_d:
                    mejor, mejor_d = tr, d
            if mejor is None:
                self.tracks.append(Track(box, t))
            else:
                mejor.box, mejor.perdido, mejor.t_visto = box, 0, t
        muertos = [tr for tr in self.tracks if tr.perdido > self.max_perdido]
        self.tracks = [tr for tr in self.tracks if tr.perdido <= self.max_perdido]
        return muertos

    def activos(self):
        return [tr for tr in self.tracks if tr.perdido == 0]


# --------------------------------------------------------------------------- #
# 5. HUD
# --------------------------------------------------------------------------- #
def panel(img, x, y, w, h, titulo, filas, accent=CYAN):
    H, W = img.shape[:2]
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        return
    sub = img[y:y + h, x:x + w]
    cv2.addWeighted(np.full(sub.shape, DARK, np.uint8), 0.65, sub, 0.35, 0, sub)
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 1)
    cv2.rectangle(img, (x, y), (x + w, y + 22), accent, -1)
    cv2.putText(img, titulo, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DARK, 1, cv2.LINE_AA)
    for i, fila in enumerate(filas):
        yy = y + 42 + i * 20
        if yy > y + h - 4:
            break
        cv2.putText(img, str(fila), (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, WHITE, 1, cv2.LINE_AA)


def dibujar(frame, tracks, roi_poly, confirmadas, t_video, progress, fmt_re, umbral):
    out = frame.copy()
    if roi_poly is not None:
        cv2.polylines(out, [roi_poly], True, GREY, 1)

    for tr in tracks:
        x, y, w, h = tr.box
        texto, conf = tr.consenso(fmt_re)
        valido = bool(texto and fmt_re.match(texto))
        color = GREEN if (valido and conf >= umbral) else (AMBER if texto else RED)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        etiqueta = f"#{tr.id} {texto or '...'}"
        if texto:
            etiqueta += f" {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ytxt = max(y - 8, th + 6)
        cv2.rectangle(out, (x, ytxt - th - 6), (x + tw + 10, ytxt + 4), color, -1)
        cv2.putText(out, etiqueta, (x + 5, ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, DARK, 2, cv2.LINE_AA)
        cv2.putText(out, f"{len(tr.lecturas)} lecturas", (x, y + h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    H, W = out.shape[:2]
    ultimas = list(confirmadas)[-6:][::-1]
    panel(out, 14, 14, 270, 44 + 20 * max(len(ultimas), 1),
          f"PLACAS CONFIRMADAS: {len(confirmadas)}",
          ultimas or ["(ninguna aun)"], accent=CYAN)
    panel(out, W - 210, 14, 196, 84, "SESION", [
        f"En pantalla: {len(tracks)}",
        f"T. video: {timedelta(seconds=int(t_video))}",
    ], accent=CYAN)
    if progress is not None:
        cv2.rectangle(out, (0, H - 5), (int(W * progress), H - 1), CYAN, -1)
    return out


# --------------------------------------------------------------------------- #
# 6. ROI interactivo
# --------------------------------------------------------------------------- #
def pedir_roi(frame):
    clicks = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and clicks:
            clicks.pop()

    cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("roi", on_mouse)
    print("\nClic en los vertices de la zona de lectura. ENTER acepta, ESC omite.\n")
    while True:
        vis = frame.copy()
        for p in clicks:
            cv2.circle(vis, p, 5, CYAN, -1)
        if len(clicks) > 1:
            cv2.polylines(vis, [np.array(clicks, np.int32)], False, CYAN, 1)
        cv2.putText(vis, "ROI: clic + ENTER (ESC = todo el cuadro)", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.imshow("roi", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if k in (13, 10) and len(clicks) >= 3:
            cv2.destroyAllWindows()
            return np.array(clicks, np.int32)


# --------------------------------------------------------------------------- #
# 7. Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Lectura automatica de placas en video")
    ap.add_argument("video", help="archivo de video, indice de webcam (0) o URL RTSP")
    ap.add_argument("--ocr", default="auto", choices=["auto", "tesseract", "easyocr"])
    ap.add_argument("--formato", default="ecuador", choices=list(FORMATOS) + ["ninguno"])
    ap.add_argument("--csv", default="placas.csv")
    ap.add_argument("--crops", default=None, help="carpeta donde guardar los recortes")
    ap.add_argument("--record", default=None, help="mp4 de salida con el HUD")
    ap.add_argument("--roi", action="store_true", help="dibujar la zona de lectura")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="procesa 1 de cada N cuadros")
    ap.add_argument("--ocr-cada", type=int, default=3,
                    help="corre el OCR cada N cuadros por placa (ahorra CPU)")
    ap.add_argument("--min-lecturas", type=int, default=3,
                    help="lecturas coincidentes necesarias para confirmar una placa")
    ap.add_argument("--min-conf", type=float, default=0.55, help="confianza minima 0-1")
    ap.add_argument("--speed", type=float, default=0.0, help="0 = a tope, 1 = tiempo real")
    args = ap.parse_args()

    fmt_re = re.compile(FORMATOS.get(args.formato, FORMATOS["libre"])) \
        if args.formato != "ninguno" else re.compile(r"^[A-Z0-9]{4,9}$")

    src = int(args.video) if args.video.isdigit() else args.video
    if isinstance(src, str) and not os.path.exists(src) and "://" not in src:
        sys.exit(f"No existe el archivo: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"OpenCV no pudo abrir: {args.video}")

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
    print(f"[formato] {args.formato}: {fmt_re.pattern}")

    roi_poly = pedir_roi(frame) if (args.roi and not args.headless) else None
    roi_mask = None
    if roi_poly is not None:
        roi_mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(roi_mask, [roi_poly], 255)

    det = PlateDetector()
    ocr = OCR(args.ocr)
    tracker = Tracker()
    if args.crops:
        os.makedirs(args.crops, exist_ok=True)

    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in / max(args.stride, 1), (W, H))
        if not writer.isOpened():
            print("[aviso] no pude abrir el VideoWriter; sigo sin grabar.")
            writer = None

    nuevo = not os.path.exists(args.csv)
    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if nuevo:
        csv_w.writerow(["placa", "confianza", "valida_formato", "n_lecturas",
                        "t_primer_visto_s", "t_ultimo_visto_s", "track_id",
                        "timestamp", "recorte"])

    confirmadas: dict[str, dict] = {}
    vis = frame.copy()
    idx = processed = 0
    paused = False
    t0_wall = time.time()
    t_video = args.start

    def cerrar_track(tr):
        """Confirma (o descarta) una placa cuando el vehiculo sale de escena."""
        texto, conf = tr.consenso(fmt_re)
        if not texto or len(tr.lecturas) < args.min_lecturas or conf < args.min_conf:
            return
        valida = bool(fmt_re.match(texto))
        if args.formato != "ninguno" and not valida:
            return
        prev = confirmadas.get(texto)
        if prev and conf <= prev["conf"]:
            prev["t_fin"] = max(prev["t_fin"], tr.t_visto)
            return
        ruta = ""
        if args.crops and tr.mejor_crop is not None:
            ruta = os.path.join(args.crops, f"{texto}_{tr.id}.png")
            cv2.imwrite(ruta, tr.mejor_crop)
        confirmadas[texto] = {"conf": conf, "n": len(tr.lecturas),
                              "t_ini": tr.t_inicio, "t_fin": tr.t_visto,
                              "id": tr.id, "crop": ruta, "valida": valida}
        csv_w.writerow([texto, round(conf, 3), valida, len(tr.lecturas),
                        round(tr.t_inicio, 2), round(tr.t_visto, 2), tr.id,
                        datetime.now().isoformat(timespec="seconds"), ruta])
        csv_f.flush()
        print(f"\n  [{timedelta(seconds=int(tr.t_visto))}] PLACA: {texto}  "
              f"conf={conf*100:.0f}%  ({len(tr.lecturas)} lecturas)")

    print("Procesando...  (q = salir)")
    try:
        while True:
            if not paused:
                if idx > 0:
                    ok, frame = cap.read()
                    if not ok:
                        print("\nFin del video.")
                        break
                idx += 1
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_video = pos / 1000.0 if pos and pos > 0 else args.start + idx / fps_in
                if args.end is not None and t_video > args.end:
                    print("\nLlegue al segundo final pedido.")
                    break
                if args.stride > 1 and (idx - 1) % args.stride:
                    continue
                processed += 1

                trabajo = frame if roi_mask is None else \
                    cv2.bitwise_and(frame, frame, mask=roi_mask)
                boxes = det.detect(trabajo)
                for tr in tracker.actualizar(boxes, t_video):
                    cerrar_track(tr)

                # OCR solo cada N cuadros por placa: el cuello de botella es este
                for tr in tracker.activos():
                    if processed % max(args.ocr_cada, 1):
                        continue
                    x, y, w, h = tr.box
                    pad = int(0.06 * h)
                    crop = frame[max(y - pad, 0):min(y + h + pad, H),
                                 max(x - pad, 0):min(x + w + pad, W)]
                    binaria = preparar_recorte(crop)
                    texto, conf = ocr.leer(binaria)
                    if texto and 4 <= len(texto) <= 9:
                        tr.agregar(texto, conf, crop.copy())

                progress = (t_video / dur) if dur else None
                vis = dibujar(frame, tracker.activos(), roi_poly, confirmadas,
                              t_video, progress, fmt_re, args.min_conf)
                if writer:
                    writer.write(vis)

                if args.headless and processed % 25 == 0:
                    pct = f"{progress*100:5.1f}%" if progress else f"{processed} cuadros"
                    sys.stdout.write(f"\r  {pct}  t={timedelta(seconds=int(t_video))}  "
                                     f"en pantalla={len(tracker.activos())}  "
                                     f"confirmadas={len(confirmadas)}   ")
                    sys.stdout.flush()

                if args.speed > 0:
                    objetivo = (t_video - args.start) / args.speed
                    lag = objetivo - (time.time() - t0_wall)
                    if lag > 0:
                        time.sleep(min(lag, 0.25))

            if not args.headless:
                try:
                    cv2.imshow("ANPR", vis)
                except cv2.error:
                    print("[aviso] este OpenCV no tiene ventanas; sigo headless.")
                    args.headless = True
                    continue
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    paused = not paused
                if k == ord("s"):
                    nombre = f"captura_{int(t_video)}s.png"
                    cv2.imwrite(nombre, vis)
                    print("Guardado", nombre)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        for tr in tracker.tracks:          # cierra lo que quedo en escena
            cerrar_track(tr)
        cap.release()
        if writer:
            writer.release()
        csv_f.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ----------------------------- resumen ----------------------------- #
    print("\n" + "=" * 66)
    print(f"Cuadros procesados : {processed}")
    print(f"Tiempo de video    : {timedelta(seconds=int(t_video))}")
    print(f"Placas confirmadas : {len(confirmadas)}")
    if confirmadas:
        print(f"\n  {'PLACA':<10} {'CONF':>6} {'LECT':>5}  {'PRIMER VISTO':>13}")
        print("  " + "-" * 40)
        for txt, d in sorted(confirmadas.items(), key=lambda kv: kv[1]["t_ini"]):
            print(f"  {txt:<10} {d['conf']*100:5.0f}% {d['n']:5d}  "
                  f"{str(timedelta(seconds=int(d['t_ini']))):>13}")
    print(f"\nCSV                : {args.csv}")
    if args.crops:
        print(f"Recortes           : {args.crops}/")
    if args.record:
        print(f"Video anotado      : {args.record}")
    print("=" * 66)


if __name__ == "__main__":
    main()
