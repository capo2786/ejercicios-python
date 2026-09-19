#!/bin/bash
# Doble clic para instalar todo lo necesario. Se corre UNA sola vez.
cd "$(dirname "$0")"
echo "============================================================"
echo "  Instalando dependencias para los 3 proyectos de vision"
echo "============================================================"
echo

PY=$(command -v python3 || command -v python)
echo "Usando Python: $PY"
$PY --version
echo

echo "[1/3] Librerias de Python (OpenCV, NumPy, YOLO, OCR)..."
$PY -m pip install --upgrade pip
$PY -m pip install opencv-python numpy ultralytics pytesseract

echo
echo "[2/3] Tesseract (motor de OCR para las placas)..."
if command -v tesseract >/dev/null 2>&1; then
    echo "      Ya esta instalado: $(tesseract --version 2>&1 | head -1)"
elif command -v brew >/dev/null 2>&1; then
    brew install tesseract
else
    echo "      AVISO: no encontre Homebrew."
    echo "      Instalalo desde https://brew.sh y vuelve a correr este archivo."
    echo "      (Solo afecta al proyecto 02-lector-placas; los otros dos funcionan igual.)"
fi

echo
echo "[3/3] Verificando..."
$PY - <<'PYCHECK'
mods = [("cv2","OpenCV"), ("numpy","NumPy"), ("ultralytics","YOLO"), ("pytesseract","OCR")]
for m, nombre in mods:
    try:
        __import__(m)
        print(f"   OK   {nombre}")
    except ImportError:
        print(f"   FALTA {nombre}  <-- revisa el error de arriba")
PYCHECK

echo
echo "============================================================"
echo "  Listo. Ahora entra a cualquier carpeta y haz doble clic"
echo "  en su archivo EJECUTAR.command"
echo "============================================================"
echo
read -p "Presiona ENTER para cerrar..."
