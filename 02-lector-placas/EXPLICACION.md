# Lector de placas (ANPR) — cómo funciona

## Qué hace

Lee un video, encuentra las placas de los vehículos, las lee, y confirma cada
una solo cuando está segura. Sobre el video de prueba:

```
  PLACA        CONF  LECT   PRIMER VISTO
  ----------------------------------------
  PBA1234       97%    45        0:00:01
  GSC7789       93%    46        0:00:09
  ABG4521       94%    46        0:00:17
```

Tres de tres, sin falsos positivos. Fíjate en la columna `LECT`: cada placa se
leyó unas **45 veces**. Ahí está el truco completo.

---

## El problema real del OCR

Si tomas un cuadro suelto y le pasas un motor de OCR, obtienes algo así:

```
cuadro 12  ->  PBA1Z34
cuadro 13  ->  PBA1234
cuadro 14  ->  P8A1234
cuadro 15  ->  PBA1234
cuadro 16  ->  PBAI234
```

Ninguna lectura individual es confiable. El OCR confunde sistemáticamente
**0/O, 1/I, 5/S, 8/B, 2/Z** — son pares que en muchas tipografías se distinguen
por dos o tres píxeles, y con desenfoque de movimiento se vuelven idénticos.

La tentación del principiante es pelear con el preprocesamiento: más contraste,
otro umbral, otro filtro. Se puede mejorar algo, pero es una guerra perdida
contra la física de la imagen.

**La solución no está en el cuadro; está en el tiempo.** El auto no aparece una
vez: aparece en 45 cuadros. Si juntas las 45 lecturas y votas posición por
posición, los errores —que son aleatorios— se cancelan, y la respuesta correcta
—que es consistente— gana:

```
posición:      0    1    2    3    4    5    6
lecturas:      P    B    A    1    2    3    4
               P    B    A    1    Z    3    4
               P    8    A    1    2    3    4
               P    B    A    I    2    3    4
             ────────────────────────────────────
  votado:      P    B    A    1    2    3    4     ✓
```

Esto se llama **fusión temporal** y es la idea central del script.

---

## Las cuatro piezas

```
  DETECTAR          PREPARAR           LEER            VOTAR
┌───────────┐   ┌──────────────┐   ┌─────────┐   ┌────────────┐
│ ¿dónde    │──▶│ enderezar,   │──▶│  OCR    │──▶│ consenso   │
│ hay placa?│   │ binarizar    │   │         │   │ entre      │
│           │   │              │   │         │   │ cuadros    │
└───────────┘   └──────────────┘   └─────────┘   └────────────┘
 PlateDetector   preparar_recorte      OCR         Track.consenso
```

---

## Pieza 1: detectar — clase `PlateDetector`

Usa **dos métodos a la vez** y fusiona sus resultados, porque cada uno falla en
casos distintos.

**a) Cascade de Haar** (`_by_cascade`). Es un clasificador entrenado que viene
incluido con OpenCV — no hay que descargar nada. Rápido, pero entrenado con
placas europeas, así que pierde algunas.

**b) Morfología matemática** (`_by_morphology`). Acá no hay aprendizaje
automático, solo procesamiento de imagen clásico, y vale la pena seguir el
razonamiento paso a paso:

```python
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.rect_k)
```
*Blackhat* resalta regiones **oscuras sobre fondo claro** — exactamente lo que
son los caracteres negros de una placa blanca. El kernel es rectangular y ancho
(25×7) porque buscamos texto horizontal, no manchas cualesquiera.

```python
grad = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
```
Sobel en dirección X mide **cambios horizontales bruscos**. El texto tiene
muchísimos bordes verticales seguidos (los trazos de las letras); el asfalto o
el cielo, casi ninguno.

```python
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, self.rect_k)
```
El *cierre* con el mismo kernel ancho une los caracteres sueltos en **un solo
bloque**. Deja de haber siete letras y pasa a haber una mancha con forma de
placa.

Después se buscan contornos y se filtran por relación de aspecto: una placa es
entre 1.8 y 6 veces más ancha que alta. Una ventana, una llanta o una sombra no
pasan ese filtro.

Al final, `_merge()` fusiona las cajas que los dos métodos propusieron para la
misma placa, usando **IoU** (*intersection over union*): el área compartida
dividida para el área total. Si dos cajas comparten más del 35%, son la misma.

---

## Pieza 2: preparar el recorte — `preparar_recorte()`

El OCR no lee fotos; lee imágenes binarias limpias. Esta función hace la
traducción, y cada paso tiene una razón:

| paso | por qué |
|---|---|
| `cv2.resize` a 64 px de alto | Tesseract necesita caracteres de cierto tamaño mínimo; agrandar ayuda aunque no añada información |
| `bilateralFilter` | quita ruido **conservando los bordes** — un desenfoque normal derretiría las letras |
| `CLAHE` | ecualización de histograma **local**: rescata la placa cuando media está en sombra y media al sol |
| `threshold OTSU` | elige el umbral de blanco/negro automáticamente según la imagen, no un número fijo |
| `deskew` | endereza la placa si el auto viene en ángulo |
| `copyMakeBorder` | Tesseract lee peor lo que toca el borde de la imagen; se le da margen |

El `deskew()` merece una mirada: toma todos los píxeles de texto, calcula el
**rectángulo rotado mínimo** que los contiene (`cv2.minAreaRect`) y usa su
ángulo para rotar todo de vuelta. Pero solo si el ángulo está entre 0.6° y 20°:
menos que eso es ruido y no vale rotar, más que eso significa que el rectángulo
agarró basura y rotar empeoraría todo. **Un buen algoritmo sabe cuándo no
actuar.**

---

## Pieza 3: leer — clase `OCR`

Una envoltura delgada sobre dos motores posibles: **Tesseract** (predeterminado,
no descarga nada) o **EasyOCR** (más preciso, más pesado). El resto del script
no sabe cuál está usando.

Dos parámetros de Tesseract hacen casi toda la diferencia:

```python
--psm 7                        # "la imagen es UNA sola línea de texto"
-c tessedit_char_whitelist=ABC...012...   # solo letras y números
```

La lista blanca es especialmente importante: sin ella, Tesseract puede devolver
guiones, puntos o letras acentuadas que no existen en ninguna placa. Restringir
el alfabeto **reduce el espacio de error**.

---

## Pieza 4: votar — `Track.agregar()` y `Track.consenso()`

Cada placa rastreada acumula sus lecturas. Pero el voto no es simple:

```python
for i, ch in enumerate(texto):
    self.votos[i][ch] += conf      # el voto PESA por confianza
```

Una lectura con 90% de confianza vale más que una con 30%. Es un voto
ponderado, no una simple mayoría.

Y encima viene la corrección final, `corregir_por_formato()`, que usa algo que
el OCR no sabe: **la estructura de la placa**. En Ecuador es tres letras y
luego números. Entonces:

- un `0` en la posición 1 tiene que ser una `O`
- una `S` en la posición 5 tiene que ser un `5`

```python
if i < n_letras:  arreglado.append(A_LETRA.get(ch, ch))
else:             arreglado.append(A_NUMERO.get(ch, ch))
```

Con la salvaguarda de que la corrección solo se acepta si el resultado **sí
valida** contra el formato. Si no, se devuelve el original sin tocar: es mejor
entregar una lectura dudosa que una inventada con confianza falsa.

> Esta es una lección que se repite en todo sistema de IA aplicada: **el
> conocimiento del dominio vale más que el modelo**. Saber que una placa
> ecuatoriana es `LLL-NNNN` corrige errores que ningún preprocesamiento
> arreglaría.

Formatos incluidos: `ecuador`, `andino` (Colombia/Perú), `mercosur`
(Argentina/Brasil), `libre`.

---

## El umbral de confirmación

Una placa solo se reporta si cumple **tres** condiciones:

1. al menos 3 lecturas (`--min-lecturas`)
2. confianza del consenso ≥ 0.55 (`--min-conf`)
3. valida contra el formato

Es un sistema deliberadamente **conservador**: prefiere no reportar antes que
reportar mal. En un control de acceso, una placa inventada es mucho peor que una
placa no leída — la primera le abre la puerta a quien no debe.

---

## Para probar

```bash
# Ver los recortes que el sistema usó para decidir
open recortes/

# Generar otro video de prueba con placas distintas
python generar_video_prueba.py

# Con EasyOCR en vez de Tesseract (si lo instalaste)
python anpr.py autos.mp4 --ocr easyocr

# Exigiendo más evidencia antes de confirmar
python anpr.py autos.mp4 --min-lecturas 10 --min-conf 0.8
```

## Retos

1. **Demuestra que la votación sirve.** Pon `--min-lecturas 1` y compara los
   resultados contra la corrida normal. ¿Aparecen placas erróneas?

2. **Rompe el formato.** Edita `generar_video_prueba.py` y pon una placa que
   no cumpla el formato ecuatoriano, por ejemplo `AB-12345`. ¿Qué hace el
   script? ¿Es el comportamiento correcto?

3. **Agrega tu formato.** Añade al diccionario `FORMATOS` el patrón de placas
   de motos ecuatorianas o el de vehículos oficiales.

4. **Compara detectores.** Comenta la llamada a `_by_cascade` y corre solo con
   morfología. Después al revés. ¿Cuál encuentra más placas? ¿Cuál se equivoca
   más?

5. **El caso difícil.** Graba con tu celular un auto en movimiento de noche y
   pásalo por el script. Probablemente falle. Diagnostica **dónde**: ¿no
   detecta la placa, o la detecta y no la lee? Guarda los recortes con
   `--crops` para verlo.

6. **Discusión.** Este sistema identifica vehículos individuales. ¿Qué
   obligaciones legales y éticas aparecen al desplegarlo? ¿Cuánto tiempo
   deberían guardarse los registros? ¿Quién debería poder consultarlos?
