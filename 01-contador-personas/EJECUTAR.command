#!/bin/bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
echo "=== CONTADOR DE PERSONAS ==="
echo "Se va a abrir una ventana con el video."
echo "Teclas:  q = salir   espacio = pausa   s = guardar captura"
echo
$PY contador.py vtest.avi --motor yolo --speed 1 --end 40 \
     --record salida_anotada.mp4 --serie-csv ocupacion.csv
echo
read -p "Presiona ENTER para cerrar..."
