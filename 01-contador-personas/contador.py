#!/usr/bin/env python3
"""
contador.py — Conteo de personas en video (entradas / salidas / aforo)
=======================================================================

Lee un video (o webcam/RTSP) y, cuadro a cuadro:

  1. DETECTA personas. Tres motores, en orden de preferencia:
       yolo   -> ultralytics YOLO, si lo tienes instalado (el mas preciso)
       dnn    -> MobileNet-SSD via cv2.dnn, si le pasas los archivos del modelo
       hog    -> HOG + SVM, incluido en OpenCV, sin descargar nada (respaldo)
  2. SIGUE a cada persona con un rastreador por centroide (le asigna un ID).
  3. CUENTA los cruces de una LINEA que tu defines: quien la cruza en un sentido
     suma ENTRADA, en el otro SALIDA. El aforo es entradas - salidas.
  4. Registra en CSV cada cruce y una serie temporal de ocupacion.

Por que se cuenta por CRUCE DE LINEA y no "cuantas cajas hay": contar cajas por
cuadro da un numero que tiembla (una persona tapada por otra desaparece y
"vuelve a entrar"). El cruce de linea con ID persistente cuenta cada persona
UNA vez, que es lo que de verdad te sirve para aforo.

Instalacion:
    pip install opencv-python numpy
    # opcional (mucho mejor deteccion):  pip install ultralytics

Uso:
    python contador.py entrada.mp4 --linea              # dibujar la linea con el mouse
    python contador.py entrada.mp4 --motor yolo
    python contador.py entrada.mp4 --headless --record salida.mp4
    python contador.py 0                                 # webcam

Teclas: q salir | espacio pausa | s captura PNG | r reinicia contadores
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
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
COLORES = [(230, 220, 60), (120, 220, 0), (0, 190, 255), (200, 120, 255),
           (255, 180, 90), (120, 255, 220), (180, 180, 255), (90, 230, 160)]


# --------------------------------------------------------------------------- #
# 1. Detectores de personas
# --------------------------------------------------------------------------- #
class Detector:
    """Devuelve una lista de cajas (x, y, w, h) de personas."""

    def __init__(self, motor="auto", modelo=None, proto=None, conf_min=0.45):
        self.conf_min = conf_min
        self.motor = None

        if motor in ("auto", "yolo"):
            try:
                from ultralytics import YOLO                      # noqa: PLC0415
                self.yolo = YOLO(modelo or "yolov8n.pt")
                self.motor = "yolo"
            except Exception as exc:                              # noqa: BLE001
                if motor == "yolo":
                    sys.exit(f"YOLO no disponible ({exc}).  pip install ultralytics")

        if self.motor is None and motor in ("auto", "dnn"):
            if modelo and proto and os.path.exists(modelo) and os.path.exists(proto):
                self.net = cv2.dnn.readNetFromCaffe(proto, modelo)
                self.motor = "dnn"
            elif motor == "dnn":
                sys.exit("Para --motor dnn pasa --modelo y --proto de MobileNet-SSD.")

        if self.motor is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.motor = "hog"

        print(f"[detector] motor: {self.motor}")
        if self.motor == "hog":
            print("           (HOG es el respaldo: rapido pero impreciso con gente "
                  "muy junta.\n            Para produccion instala:  pip install ultralytics)")

    def detectar(self, frame):
        if self.motor == "yolo":
            return self._yolo(frame)
        if self.motor == "dnn":
            return self._dnn(frame)
        return self._hog(frame)

    def _yolo(self, frame):
        res = self.yolo(frame, classes=[0], conf=self.conf_min, verbose=False)
        cajas = []
        for r in res:
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b[:4])
                cajas.append((x1, y1, x2 - x1, y2 - y1))
        return cajas

    def _dnn(self, frame):
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        self.net.setInput(blob)
        det = self.net.forward()
        cajas = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < self.conf_min or int(det[0, 0, i, 1]) != 15:   # 15 = person
                continue
            x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
            cajas.append((x1, y1, x2 - x1, y2 - y1))
        return cajas

    def _hog(self, frame):
        escala = 640 / max(frame.shape[1], 1)
        chico = cv2.resize(frame, None, fx=escala, fy=escala) if escala < 1 else frame
        rects, pesos = self.hog.detectMultiScale(chico, winStride=(8, 8),
                                                 padding=(8, 8), scale=1.05)
        inv = 1 / escala if escala < 1 else 1.0
        cajas = [tuple((np.array(r) * inv).astype(int))
                 for r, p in zip(rects, pesos) if p >= 0.3]
        return nms(cajas, 0.4)


def nms(cajas, thr=0.4):
    """Supresion de no-maximos por area (evita contar 3 veces a la misma persona)."""
    if not cajas:
        return []
    cajas = sorted(cajas, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for b in cajas:
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# 2. Rastreo por centroide
# --------------------------------------------------------------------------- #
class Persona:
    _next_id = 1

    def __init__(self, caja, t):
        self.id = Persona._next_id
        Persona._next_id += 1
        self.caja = caja
        self.t_inicio = self.t_visto = t
        self.perdido = 0
        self.rastro: deque = deque(maxlen=48)
        self.rastro.append(self.centro)
        self.contada = False          # ya cruzo la linea una vez
        self.color = COLORES[self.id % len(COLORES)]

    @property
    def centro(self):
        x, y, w, h = self.caja
        return x + w / 2, y + h / 2

    def mover(self, caja, t):
        self.caja, self.perdido, self.t_visto = caja, 0, t
        self.rastro.append(self.centro)


class Rastreador:
    """Asocia detecciones con personas ya vistas por cercania del centroide."""

    def __init__(self, dist_max=110, max_perdido=25):
        self.dist_max, self.max_perdido = dist_max, max_perdido
        self.personas: list[Persona] = []

    def actualizar(self, cajas, t):
        for p in self.personas:
            p.perdido += 1

        libres = list(self.personas)
        for caja in cajas:
            cx, cy = caja[0] + caja[2] / 2, caja[1] + caja[3] / 2
            mejor, mejor_d = None, self.dist_max
            for p in libres:
                px, py = p.centro
                d = float(np.hypot(cx - px, cy - py))
                if d < mejor_d:
                    mejor, mejor_d = p, d
            if mejor is None:
                self.personas.append(Persona(caja, t))
            else:
                mejor.mover(caja, t)
                libres.remove(mejor)

        salieron = [p for p in self.personas if p.perdido > self.max_perdido]
        self.personas = [p for p in self.personas if p.perdido <= self.max_perdido]
        return salieron

    def visibles(self):
        return [p for p in self.personas if p.perdido == 0]


# --------------------------------------------------------------------------- #
# 3. Linea de conteo
# --------------------------------------------------------------------------- #
class LineaConteo:
    """Cuenta cruces con signo: de lado negativo a positivo = ENTRADA."""

    def __init__(self, p1, p2, invertir=False):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.invertir = invertir
        self.entradas = 0
        self.salidas = 0
        self.eventos: list[tuple[float, str, int]] = []

    def lado(self, punto):
        """Signo del producto cruz: de que lado de la linea cae el punto."""
        v = self.p2 - self.p1
        w = np.array(punto, dtype=np.float64) - self.p1
        c = v[0] * w[1] - v[1] * w[0]
        return 0 if abs(c) < 1e-9 else (1 if c > 0 else -1)

    def revisar(self, persona, t, min_rastro=4):
        """Compara el lado actual contra el de hace unos cuadros."""
        if persona.contada or len(persona.rastro) < min_rastro:
            return None
        antes = self.lado(persona.rastro[-min_rastro])
        ahora = self.lado(persona.rastro[-1])
        if antes == 0 or ahora == 0 or antes == ahora:
            return None
        if not self._cerca(persona.rastro[-1]):
            return None            # cruzo la recta infinita, pero fuera del segmento
        entra = (antes < 0 and ahora > 0) != self.invertir
        persona.contada = True
        if entra:
            self.entradas += 1
        else:
            self.salidas += 1
        tipo = "ENTRADA" if entra else "SALIDA"
        self.eventos.append((t, tipo, persona.id))
        return tipo

    def _cerca(self, punto, margen=1.35):
        """El cruce vale solo si ocurre sobre el segmento dibujado, no fuera."""
        v = self.p2 - self.p1
        largo2 = float(v @ v)
        if largo2 < 1e-9:
            return False
        w = np.array(punto, dtype=np.float64) - self.p1
        s = float(v @ w) / largo2          # proyeccion normalizada 0..1
        holgura = (margen - 1.0) / 2
        return -holgura <= s <= 1 + holgura

    @property
    def aforo(self):
        return self.entradas - self.salidas


def pedir_linea(frame, ruta_cfg):
    """Dos clics definen la linea de conteo. Se guarda para la proxima vez."""
    pts = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow("linea", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("linea", on_mouse)
    print("\n2 clics para la linea de conteo (puerta, pasillo, torniquete).")
    print("Clic derecho deshace. ENTER acepta. ESC cancela.\n")
    while True:
        vis = frame.copy()
        for p in pts:
            cv2.circle(vis, p, 6, CYAN, -1)
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], CYAN, 3)
        cv2.putText(vis, "LINEA DE CONTEO: 2 clics + ENTER", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.imshow("linea", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if k in (13, 10) and len(pts) == 2:
            cv2.destroyAllWindows()
            with open(ruta_cfg, "w", encoding="utf-8") as f:
                json.dump({"linea": [list(pts[0]), list(pts[1])]}, f, indent=2)
            print(f"Linea guardada en {ruta_cfg}")
            return pts


# --------------------------------------------------------------------------- #
# 4. HUD
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
    cv2.putText(img, titulo, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, DARK, 1, cv2.LINE_AA)
    for i, fila in enumerate(filas):
        yy = y + 44 + i * 26
        if yy > y + h - 4:
            break
        cv2.putText(img, str(fila), (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, WHITE, 1, cv2.LINE_AA)


def dibujar(frame, personas, linea, t_video, progress, serie, aforo_max):
    out = frame.copy()
    H, W = out.shape[:2]

    if linea is not None:
        cv2.line(out, tuple(map(int, linea.p1)), tuple(map(int, linea.p2)),
                 CYAN, 3, cv2.LINE_AA)
        # flecha que indica cual sentido cuenta como ENTRADA
        medio = ((linea.p1 + linea.p2) / 2).astype(int)
        v = linea.p2 - linea.p1
        n = np.array([-v[1], v[0]], dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-9) * (-40 if linea.invertir else 40)
        cv2.arrowedLine(out, tuple(medio), tuple((medio + n).astype(int)),
                        GREEN, 3, tipLength=0.35)
        cv2.putText(out, "IN", tuple((medio + n * 1.35).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)

    for p in personas:
        x, y, w, h = p.caja
        cv2.rectangle(out, (x, y), (x + w, y + h), p.color, 2)
        cx, cy = map(int, p.centro)
        cv2.circle(out, (cx, cy), 4, p.color, -1)
        if len(p.rastro) > 1:
            pts = np.array(p.rastro, np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, p.color, 2, cv2.LINE_AA)
        etq = f"#{p.id}" + (" OK" if p.contada else "")
        cv2.putText(out, etq, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, p.color, 2, cv2.LINE_AA)

    e = linea.entradas if linea else 0
    s = linea.salidas if linea else 0
    a = linea.aforo if linea else len(personas)
    panel(out, 14, 14, 232, 140, "CONTEO DE PERSONAS", [
        f"Entradas : {e}",
        f"Salidas  : {s}",
        f"Aforo    : {a}",
    ], accent=GREEN if a <= aforo_max else RED)

    panel(out, W - 210, 14, 196, 100, "SESION", [
        f"Visibles: {len(personas)}",
        f"T: {timedelta(seconds=int(t_video))}",
    ], accent=CYAN)

    # curva de ocupacion
    if len(serie) > 2:
        x0, y0, wg, hg = 14, H - 84, 232, 56
        cv2.rectangle(out, (x0, y0), (x0 + wg, y0 + hg), (90, 90, 90), 1)
        vals = np.array([v for _, v in serie][-wg:], dtype=np.float32)
        top = max(float(vals.max()), 1.0)
        xs = np.linspace(x0, x0 + wg, len(vals)).astype(np.int32)
        ys = (y0 + hg - vals / top * (hg - 4)).astype(np.int32)
        cv2.polylines(out, [np.stack([xs, ys], 1)], False, CYAN, 1, cv2.LINE_AA)
        cv2.putText(out, f"ocupacion (max {int(top)})", (x0 + 4, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)

    if aforo_max < 9999 and a > aforo_max:
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), RED, 6)
        cv2.putText(out, "AFORO EXCEDIDO", (W // 2 - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3, cv2.LINE_AA)

    if progress is not None:
        cv2.rectangle(out, (0, H - 5), (int(W * progress), H - 1), CYAN, -1)
    return out


# --------------------------------------------------------------------------- #
# 5. Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Conteo de personas en video")
    ap.add_argument("video", help="archivo de video, indice de webcam (0) o URL RTSP")
    ap.add_argument("--motor", default="auto", choices=["auto", "yolo", "dnn", "hog"])
    ap.add_argument("--modelo", default=None, help="pesos YOLO (.pt) o caffemodel")
    ap.add_argument("--proto", default=None, help="prototxt de MobileNet-SSD")
    ap.add_argument("--conf", type=float, default=0.45, help="confianza minima")
    ap.add_argument("--config", default="linea.json")
    ap.add_argument("--linea", action="store_true", help="dibujar la linea con el mouse")
    ap.add_argument("--invertir", action="store_true", help="invertir sentido entrada/salida")
    ap.add_argument("--aforo-max", type=int, default=9999, help="alerta al superarlo")
    ap.add_argument("--csv", default="conteo.csv")
    ap.add_argument("--serie-csv", default=None, help="CSV de ocupacion en el tiempo")
    ap.add_argument("--record", default=None, help="mp4 de salida con el HUD")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="procesa 1 de cada N cuadros")
    ap.add_argument("--speed", type=float, default=0.0, help="0 = a tope, 1 = tiempo real")
    args = ap.parse_args()

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

    # ---- linea de conteo: mouse, archivo guardado, o media pantalla ---- #
    pts = None
    if args.linea and not args.headless:
        pts = pedir_linea(frame, args.config)
    if pts is None and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            pts = [tuple(p) for p in json.load(f)["linea"]]
        print(f"[config] linea desde {args.config}")
    if pts is None:
        pts = [(0, H // 2), (W, H // 2)]
        print("[aviso] sin linea definida: uso una horizontal a media pantalla.\n"
              "        Corre con --linea para dibujar la tuya.")
    linea = LineaConteo(pts[0], pts[1], invertir=args.invertir)

    det = Detector(args.motor, args.modelo, args.proto, args.conf)
    rastreador = Rastreador()

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
        csv_w.writerow(["t_video_s", "timestamp", "evento", "persona_id",
                        "entradas", "salidas", "aforo"])

    serie_f = serie_w = None
    if args.serie_csv:
        nuevo_s = not os.path.exists(args.serie_csv)
        serie_f = open(args.serie_csv, "a", newline="", encoding="utf-8")
        serie_w = csv.writer(serie_f)
        if nuevo_s:
            serie_w.writerow(["t_video_s", "timestamp", "visibles", "aforo"])

    serie: deque = deque(maxlen=2000)
    vis = frame.copy()
    idx = processed = 0
    ultimo_log = -1e9
    paused = False
    t0_wall = time.time()
    t_video = args.start

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

                cajas = det.detectar(frame)
                rastreador.actualizar(cajas, t_video)

                for p in rastreador.visibles():
                    tipo = linea.revisar(p, t_video)
                    if tipo:
                        csv_w.writerow([round(t_video, 2),
                                        datetime.now().isoformat(timespec="seconds"),
                                        tipo, p.id, linea.entradas, linea.salidas,
                                        linea.aforo])
                        csv_f.flush()
                        print(f"\n  [{timedelta(seconds=int(t_video))}] {tipo}"
                              f"  persona #{p.id}  ->  aforo={linea.aforo}")

                visibles = rastreador.visibles()
                serie.append((t_video, len(visibles)))
                if serie_w and t_video - ultimo_log >= 1.0:
                    ultimo_log = t_video
                    serie_w.writerow([round(t_video, 2),
                                      datetime.now().isoformat(timespec="seconds"),
                                      len(visibles), linea.aforo])
                    serie_f.flush()

                progress = (t_video / dur) if dur else None
                vis = dibujar(frame, visibles, linea, t_video, progress,
                              serie, args.aforo_max)
                if writer:
                    writer.write(vis)

                if args.headless and processed % 25 == 0:
                    pct = f"{progress*100:5.1f}%" if progress else f"{processed} cuadros"
                    sys.stdout.write(f"\r  {pct}  t={timedelta(seconds=int(t_video))}  "
                                     f"visibles={len(visibles)}  "
                                     f"in={linea.entradas} out={linea.salidas} "
                                     f"aforo={linea.aforo}   ")
                    sys.stdout.flush()

                if args.speed > 0:
                    objetivo = (t_video - args.start) / args.speed
                    lag = objetivo - (time.time() - t0_wall)
                    if lag > 0:
                        time.sleep(min(lag, 0.25))

            if not args.headless:
                try:
                    cv2.imshow("Conteo de personas", vis)
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
                if k == ord("r"):
                    linea.entradas = linea.salidas = 0
                    linea.eventos.clear()
                    print("Contadores reiniciados.")
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        cap.release()
        if writer:
            writer.release()
        csv_f.close()
        if serie_f:
            serie_f.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ----------------------------- resumen ----------------------------- #
    picos = [v for _, v in serie] or [0]
    print("\n" + "=" * 62)
    print(f"Cuadros procesados : {processed}")
    print(f"Tiempo de video    : {timedelta(seconds=int(t_video))}")
    print(f"Entradas           : {linea.entradas}")
    print(f"Salidas            : {linea.salidas}")
    print(f"Aforo final        : {linea.aforo}")
    print(f"Pico simultaneo    : {max(picos)} personas en pantalla")
    if linea.eventos:
        print("\nCronologia:")
        for t, tipo, pid in linea.eventos:
            print(f"   {str(timedelta(seconds=int(t))):>8}  {tipo:<8} persona #{pid}")
    print(f"\nCSV de eventos     : {args.csv}")
    if args.serie_csv:
        print(f"CSV de ocupacion   : {args.serie_csv}")
    if args.record:
        print(f"Video anotado      : {args.record}")
    print("=" * 62)


if __name__ == "__main__":
    main()
