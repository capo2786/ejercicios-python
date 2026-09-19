#!/bin/bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
echo "=== LECTOR DE PLACAS ==="
echo "Se va a abrir una ventana con el video."
echo "Teclas:  q = salir   espacio = pausa   s = guardar captura"
echo
$PY anpr.py autos.mp4 --formato ecuador --speed 1 \
     --crops recortes --record salida_anotada.mp4
echo
read -p "Presiona ENTER para cerrar..."
