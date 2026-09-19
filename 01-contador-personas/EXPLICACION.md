# Contador de personas — cómo funciona

## Qué hace

Lee un video, encuentra a cada persona, le pone un número que la sigue mientras
camina, y cuenta cuando cruza una línea que tú dibujas. El resultado es
**entradas, salidas y aforo actual**.

Corriendo sobre `vtest.avi` (40 segundos de peatones reales):

```
Entradas           : 6
Salidas            : 1
Aforo final        : 5
Pico simultaneo    : 10 personas en pantalla
```

---

## La decisión de diseño que importa

Hay dos formas de "contar personas en un video", y una está mal.

**La forma ingenua:** en cada cuadro, detectar cuántas personas hay y reportar
ese número. Falla feo. Si dos personas se cruzan, una tapa a la otra por medio
segundo y el conteo baja; cuando reaparece, el conteo sube. En un pasillo con
gente, ese número tiembla tanto que no sirve para nada. Peor: no distingue a
alguien que entró de alguien que solo pasó frente a la cámara.

**La forma correcta:** seguir a cada persona con una identidad, y contar el
momento en que **cruza una línea**. Así cada persona se cuenta exactamente una
vez, y sabes en qué sentido iba.

Todo el diseño del script sale de esa decisión. Se necesitan tres piezas:

```
   DETECTAR            RASTREAR              CONTAR
 ┌───────────┐      ┌─────────────┐      ┌──────────────┐
 │  ¿dónde   │─────▶│  ¿cuál es   │─────▶│  ¿cruzó la   │
 │  hay      │      │  cuál entre │      │  línea? ¿en  │
 │  gente?   │      │  cuadros?   │      │  qué sentido?│
 └───────────┘      └─────────────┘      └──────────────┘
    YOLO             Rastreador            LineaConteo
```

---

## Pieza 1: detectar — clase `Detector`

Recibe un cuadro, devuelve una lista de cajas `(x, y, ancho, alto)` donde hay
personas. Hay tres motores posibles y el script elige el mejor disponible:

| motor | qué es | cuándo usarlo |
|---|---|---|
| `yolo` | red neuronal moderna | **el bueno**, el que usamos |
| `dnn` | MobileNet-SSD | si ya tienes el modelo descargado |
| `hog` | HOG + SVM, de 2005 | respaldo, no requiere descargar nada |

Vale la pena mirar `_hog()` aunque no lo usemos: es el método **clásico** de
detección de peatones (histograma de gradientes orientados + máquina de vectores
de soporte). Viene incluido en OpenCV. Funciona, pero se equivoca mucho con
gente junta o parcialmente tapada. Compararlo contra YOLO en el mismo video es
un buen ejercicio para ver de dónde a dónde llegó la visión por computador en
veinte años:

```bash
python contador.py vtest.avi --motor hog
python contador.py vtest.avi --motor yolo
```

Después de detectar viene `nms()` — *supresión de no-máximos*. Los detectores
suelen proponer varias cajas encimadas para la misma persona; `nms` se queda con
la más grande y descarta las que se le solapan más de un 40%. Sin ese paso,
contarías a la misma persona tres veces.

---

## Pieza 2: rastrear — clases `Persona` y `Rastreador`

El detector no tiene memoria: en cada cuadro devuelve cajas sueltas, sin saber
que la caja de arriba a la izquierda es la misma persona que estaba ahí hace un
instante. Esa memoria la pone el rastreador.

El método es el más simple que funciona: **asociación por centroide**. Para cada
caja nueva, se busca la persona ya conocida cuyo centro esté más cerca. Si hay
una a menos de 110 píxeles, es ella; si no, es alguien que acaba de entrar en
escena.

```python
d = float(np.hypot(cx - px, cy - py))    # distancia entre centros
if d < mejor_d:
    mejor, mejor_d = p, d
```

Dos detalles que lo hacen aguantar el mundo real:

**Tolerancia a desapariciones.** Si una persona no se detecta en un cuadro
(alguien la tapó, se movió raro), no se borra: se le suma uno a `perdido` y se
mantiene hasta 25 cuadros. Sin esto, cada oclusión breve crearía una "persona
nueva" y el conteo se dispararía.

**Rastro.** Cada persona guarda sus últimas 48 posiciones en un `deque`. Ese
rastro es lo que se dibuja como estela de color en el video, y —más importante—
es lo que permite detectar el cruce.

> `deque(maxlen=48)` es una lista que se limita sola: al meter el elemento 49,
> bota el primero automáticamente. Evita que la memoria crezca sin control en un
> video de horas. Es un detalle chico con consecuencias grandes si el sistema va
> a correr 24/7.

---

## Pieza 3: contar — clase `LineaConteo`

Acá está la parte matemáticamente linda del script.

**¿De qué lado de una línea está un punto?** Se responde con el signo del
producto cruz. Si la línea va de `p1` a `p2`, y el punto es `q`:

```python
v = self.p2 - self.p1          # vector de la línea
w = punto   - self.p1          # vector hacia el punto
c = v[0]*w[1] - v[1]*w[0]      # producto cruz en 2D
return 1 if c > 0 else -1      # el signo dice el lado
```

Una sola multiplicación cruzada y ya sabes el lado. No hace falta calcular
ángulos ni pendientes, y funciona con la línea en cualquier orientación —
incluso vertical, donde la fórmula de la pendiente explota por división entre
cero.

**¿Cruzó?** Se compara el lado de hace unos cuadros contra el lado actual. Si
cambiaron de signo, cruzó:

```python
antes = self.lado(persona.rastro[-min_rastro])   # hace 4 cuadros
ahora = self.lado(persona.rastro[-1])            # ahora
if antes == ahora: return None                   # no cruzó
```

Se comparan cuadros separados, no consecutivos, a propósito. Si compararas
cuadro contra cuadro, alguien parado justo sobre la línea oscilaría de lado por
el ruido de la detección y generaría decenas de cruces falsos.

**¿Cruzó por donde debía?** Hay una trampa geométrica: el producto cruz responde
sobre la **recta infinita**, no sobre el segmento que dibujaste. Alguien
caminando a diez metros de la puerta, pero alineado con ella, también "cambia de
lado". Por eso `_cerca()` proyecta el punto sobre el segmento y exige que caiga
dentro de él:

```python
s = float(v @ w) / largo2      # 0 = inicio del segmento, 1 = final
return -holgura <= s <= 1 + holgura
```

**¿En qué sentido?** De lado negativo a positivo es entrada; al revés, salida.
Cuál es cuál depende de cómo dibujaste la línea, así que `--invertir` lo voltea.
En el video, la flecha verde marcada `IN` muestra cuál sentido está contando
como entrada.

Y `persona.contada = True` asegura que cada persona se cuente **una sola vez**,
aunque se quede parada sobre la línea meciéndose.

---

## El resultado

Dos CSV, que es como esto se conecta con análisis de datos de verdad:

- `conteo.csv` — una fila por cruce: momento, sentido, ID, aforo resultante.
- `ocupacion.csv` — serie temporal: cuánta gente había en cada segundo.

Con el segundo sacas horas pico, tiempo promedio de permanencia y curvas de
afluencia. El video anotado es lo bonito; **el CSV es lo que sirve**.

---

## Para probar

```bash
# 1. Con tu propia línea de conteo (2 clics del mouse)
python contador.py vtest.avi --linea

# 2. Comparar el detector clásico contra el moderno
python contador.py vtest.avi --motor hog
python contador.py vtest.avi --motor yolo

# 3. Con alerta de aforo máximo
python contador.py vtest.avi --aforo-max 4

# 4. Con tu propio video
python contador.py mi_video.mp4 --linea
```

## Retos

1. **Rompe el rastreador.** Baja `dist_max` de 110 a 20 en la clase
   `Rastreador` y observa qué pasa con los IDs. ¿Por qué se multiplican?
   ¿Qué valor sería razonable para una cámara más lejana?

2. **Rompe el contador.** Pon `min_rastro=1` en `LineaConteo.revisar()`.
   ¿Cuántos cruces falsos aparecen? ¿Por qué?

3. **Mide el error.** Cuenta a mano cuánta gente cruza realmente en los
   primeros 40 segundos de `vtest.avi` y compáralo con lo que reportó el
   script. ¿Se equivocó por exceso o por defecto? ¿Dónde?

4. **Dos líneas.** Modifica el script para soportar dos líneas paralelas y
   contar solo a quien cruce ambas en orden. Esto elimina los falsos de gente
   que se asoma y se devuelve. Es como funcionan los contadores comerciales.

5. **¿Cuánto tarda?** Mide cuántos cuadros por segundo procesa con `--motor
   hog` contra `--motor yolo`. Si tuvieras que procesar 10 cámaras en un solo
   computador, ¿cuál elegirías y qué sacrificarías?
