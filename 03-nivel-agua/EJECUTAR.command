#!/bin/bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
echo "=== MONITOR DE NIVEL DE AGUA ==="
echo "Se va a abrir una ventana con el video."
echo "Teclas:  q = salir   espacio = pausa   s = guardar captura"
echo
$PY flood_monitor.py rio_test.mp4 --station 02 --speed 1 \
     --record salida_anotada.mp4
echo
read -p "Presiona ENTER para cerrar..."
