# Monitor de nivel de agua — cómo funciona

## Qué hace

Mira un río o canal con una cámara fija, mide **a qué altura está el agua en
metros**, calcula si está subiendo y clasifica el riesgo de desborde. Sobre el
video de prueba:

```
Nivel min/med/max  : 1.11 / 2.50 / 3.89 m
Riesgo final       : DESBORDE CRITICO  (confianza 81.7 %)
Cambios de riesgo  :
    0:00:03  ->  VIGILANCIA         (1.13 m)
    0:00:12  ->  ALERTA             (1.80 m)
    0:00:22  ->  DESBORDE CRITICO   (2.60 m)
```

Este es el más interesante de los tres proyectos, porque **no hay una red
neuronal en ninguna parte**. Es física y geometría. Y funciona.

---

## El problema: ¿qué es "agua" en una imagen?

Detectar personas o placas es fácil de plantear: tienen forma. El agua no. Es
transparente, refleja el cielo, cambia de color con la hora del día y con la
turbiedad de la crecida. Un detector basado solo en color fallaría en cuanto
pase una nube.

Pero el agua tiene una propiedad que el concreto, la vegetación y el asfalto no
tienen: **nunca está quieta**. Esa es la señal que usa el script.

---

## Pieza 1: segmentar el agua — clase `WaterSegmenter`

Combina dos evidencias independientes y las suma con pesos.

**Evidencia A — movimiento acumulado.**

```python
fg = self.bg.apply(cv2.GaussianBlur(frame, (5,5), 0))
motion = (fg > 0).astype(np.float32)
self.motion_acc = 0.90 * self.motion_acc + 0.10 * motion
```

`createBackgroundSubtractorMOG2` aprende cómo se ve cada píxel "normalmente" y
marca los que cambian. Pero un cuadro suelto de movimiento es ruidoso: una hoja
que vuela también se marca.

Por eso se **acumula con un promedio exponencial**: el 90% del valor anterior
más el 10% del nuevo. Un píxel que se mueve constantemente (agua) acumula un
valor alto. Un píxel que se movió una vez (la hoja) se desvanece en pocos
cuadros. Este filtro es la diferencia entre detectar agua y detectar cualquier
cosa que pase.

> Ese promedio exponencial — `nuevo = α·dato + (1−α)·anterior` — aparece tres
> veces en este script y es una de las herramientas más útiles que existen para
> series temporales. Suaviza sin necesidad de guardar historial.

**Evidencia B — color.**

```python
color = cv2.inRange(hsv, self.lo, self.hi)
```

Se trabaja en **HSV**, no en RGB, a propósito: en HSV el *matiz* (qué color es)
está separado del *valor* (qué tan iluminado está). Eso hace que el rango
sobreviva a que pase una nube, cosa que en RGB no pasaría.

**La fusión:**

```python
score = 0.62 * motion_s + 0.38 * color
mask  = (score > 0.38)
```

Se confía más en el movimiento (62%) que en el color (38%), porque el
movimiento es la señal más robusta. Los pesos están en el archivo de
configuración; cambiarlos y ver qué pasa es el mejor ejercicio de este
proyecto.

Al final, `_largest_blob()` se queda solo con la región conectada más grande.
El río es **una** masa continua de agua; los reflejos sueltos en el pavimento
mojado, no.

---

## Pieza 2: encontrar la superficie — `water_line()`

Ya sabemos qué píxeles son agua. Ahora, ¿dónde está exactamente el borde
superior?

Se recorre **columna por columna** y en cada una se busca el primer píxel de
agua desde arriba. Pero con una condición crucial: tiene que haber una **racha
continua** de al menos 6 píxeles de agua.

```python
d = np.diff(np.concatenate(([0], col.view(np.uint8), [0])))
starts = np.flatnonzero(d ==  1)
ends   = np.flatnonzero(d == -1)
runs   = ends - starts
ok     = np.flatnonzero(runs >= min_run)
```

Sin la racha mínima, un reflejo del sol o una gota en el lente —un píxel suelto
marcado como agua— se tomaría como la superficie, y el nivel saltaría medio
metro en un cuadro.

> El truco del `np.diff` sobre el arreglo con ceros en los extremos es un
> patrón que vale la pena aprenderse: encuentra **todos** los inicios y finales
> de rachas en un arreglo booleano sin un solo bucle de Python. En un video de
> 25 cuadros por segundo, la diferencia entre vectorizar y no hacerlo decide si
> el sistema corre en tiempo real.

Después se calcula la **mediana** de todas las alturas encontradas, y se
descarta lo que se aleje más de 3 MAD (*desviación absoluta mediana*). Se usa
mediana y no promedio porque la mediana **no se deja arrastrar por valores
extremos**: si diez columnas dieron la superficie correcta y una dio un
disparate, el promedio se mueve, la mediana no.

---

## Pieza 3: píxeles a metros — clase `Calibration`

Una medición en píxeles no le sirve a nadie. La conversión se hace con dos
puntos de altura conocida — la base y el tope de un pilar, dos marcas de una
regleta:

```python
self.m_per_px = (h2 - h1) / (y2 - y1)

def to_meters(self, y_px):
    return self.h1 + (y_px - self.y1) * self.m_per_px
```

Regla de tres, nada más. Pero es el paso que convierte un experimento en un
instrumento de medición.

**Un detalle honesto:** en el video de prueba el agua sube de 0.8 m a 3.8 m
reales, y el script reporta de 1.11 m a 3.89 m. Hay un sesgo de unos 20 cm. No
es un error de programación: la línea brillante del borde del agua hace que la
superficie detectada quede unos píxeles más arriba que la real. **Todo
instrumento tiene sesgo.** Lo profesional no es fingir que no existe, sino
medirlo contra una referencia conocida y corregirlo.

---

## Pieza 4: interpretar — clase `RiverState`

Acá el nivel se convierte en información accionable.

**Suavizado.** El mismo promedio exponencial de antes, ahora sobre el nivel.
Filtra el temblor cuadro a cuadro.

**Tendencia.** Ajusta una recta por mínimos cuadrados sobre la ventana reciente
y devuelve la pendiente en **metros por hora**:

```python
slope = np.polyfit(t - t[0], h, 1)[0]
return slope * 3600.0
```

**Acá hay una decisión de diseño que vale oro.** El tiempo `t` no es el reloj
del computador: es el **tiempo del video**, leído con
`cap.get(cv2.CAP_PROP_POS_MSEC)`. Si usaras el reloj real, la tendencia saldría
distinta según qué tan rápido procese tu máquina — y un video de archivo
procesado a 300 cuadros por segundo reportaría crecidas diez veces más rápidas
de lo que fueron. Es el error clásico al pasar de cámara en vivo a video
grabado.

**Caudal.** Una curva de gasto tipo vertedero, `Q = c·(h−h₀)^1.5`. Es la
relación estándar en hidráulica entre altura de lámina y caudal. Los
coeficientes del archivo de configuración son genéricos: para caudal confiable
se necesitan **aforos reales** de ese punto del río.

**Riesgo.** Umbrales por nivel, más una regla que escala el riesgo si el agua
**sube rápido** aunque todavía esté baja:

```python
if tr > self.rise_alert and r != "DESBORDE CRITICO":
    r = RISK_ORDER[min(RISK_ORDER.index(r) + 1, 3)]
```

Es la diferencia entre un termómetro y un sistema de alerta temprana. Un río a
2 metros subiendo 50 cm por hora es más peligroso que uno a 2.5 metros estable.

**Confianza.** Y acá una nota de honestidad intelectual. El video viral que
inspiró esto mostraba *"Confidence: 95.4%"*. Ese número, en un sistema con red
neuronal, sale del clasificador. Acá no hay clasificador, así que la confianza
se construye con lo que sí se puede medir:

```python
c_line = 1.0 - (mad_px / 25.0)        # ¿qué tan consistente salió la línea?
c_cov  = coverage / 0.25              # ¿qué tan bien cubierto quedó el ROI?
conf   = 0.65 * c_line + 0.35 * c_cov
```

No es una probabilidad. Es un indicador de calidad de la medición, y está
documentado como tal. **Inventar un número de confianza que suene bien es fácil;
reportar uno que signifique algo es el trabajo real.**

---

## El período de calentamiento

```python
warming = (t_video - args.start) < args.warmup
```

Los primeros 3 segundos se miden pero **no generan alertas ni entran al
historial**. El modelo de fondo todavía está aprendiendo la escena y produce
lecturas disparatadas.

Esto no estaba en la primera versión del script. Apareció porque al probarlo, el
primer cuadro reportó un pico falso de 4.40 m y una tendencia de −4382 m/h. Se
corrigió porque **se probó**. Vale la pena decirlo en voz alta: el código que
nunca se ejecutó contra datos reales no está terminado, está empezado.

---

## Para probar

```bash
# Calibrar con tu propio video (clics del mouse)
python flood_monitor.py rio_test.mp4 --calibrate

# Regenerar el video de prueba
python generar_video_prueba.py

# Solo un tramo, procesando 1 de cada 3 cuadros
python flood_monitor.py rio_test.mp4 --start 10 --end 25 --stride 3
```

## Retos

1. **Rompe la segmentación.** En `station.json` cambia los pesos: pon
   `{"motion": 0.0, "color": 1.0, "bin": 0.38}` para usar solo color.
   ¿Qué tan mal se pone? Ahora al revés, solo movimiento. ¿Cuál importa más?

2. **Mide el sesgo.** El video sube de 0.8 m a 3.8 m reales (míralo en
   `generar_video_prueba.py`). El script reporta 1.11 a 3.89. Calcula el error
   en cada punto. ¿Es constante o crece? ¿Cómo lo corregirías en el código?

3. **Quita el calentamiento.** Corre con `--warmup 0` y observa la primera fila
   del CSV. ¿Entiendes por qué el filtro tiene que existir?

4. **La mediana contra el promedio.** En `water_line()`, cambia
   `np.median` por `np.mean` en las dos líneas donde aparece. Corre y compara
   la estabilidad del nivel. ¿Por qué la mediana aguanta mejor?

5. **Una crecida real.** Busca en YouTube un video de crecida de río con
   cámara fija, bájalo con `yt-dlp`, calíbralo y córrelo. Este es el reto de
   verdad: los datos del mundo real nunca se parecen al video de prueba.

6. **Diseño.** Si tuvieras que instalar esto en una quebrada de Quito, ¿qué
   pasa de noche? ¿Y si la cámara se mueve con el viento? Propón una solución
   para cada caso.
