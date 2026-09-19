# Visión por computador sobre video — tres sistemas que funcionan

Tres proyectos completos. Cada uno es **un solo archivo de Python**, cada uno
resuelve un problema real, y los tres vienen con su video, su salida ya
generada y la explicación de cómo funcionan por dentro.

No son ejercicios de juguete. Son las versiones simplificadas de sistemas que
hoy están desplegados en peajes, centros comerciales y estaciones
hidrológicas.

| carpeta | qué hace | lo que vas a aprender |
|---|---|---|
| `01-contador-personas` | cuenta entradas, salidas y aforo | detección con redes neuronales, rastreo, geometría de cruces |
| `02-lector-placas` | lee placas vehiculares | morfología matemática, OCR, **fusión temporal** |
| `03-nivel-agua` | mide nivel de río y riesgo de desborde | segmentación sin IA, calibración, series de tiempo |

---

## Empezar en dos pasos

### 1. Instalar (una sola vez)

Doble clic en **`INSTALAR.command`**.

Si macOS dice *"no se puede abrir porque proviene de un desarrollador no
identificado"*: clic derecho sobre el archivo → **Abrir** → **Abrir**. Solo la
primera vez.

Tarda unos minutos. Está instalando OpenCV, YOLO y un motor de OCR.

### 2. Ejecutar

Entra a cualquier carpeta y doble clic en **`EJECUTAR.command`**.

Se abre una ventana con el video y las detecciones encima, en vivo.

```
Teclas:   q = salir      espacio = pausa      s = guardar captura
```

La primera vez que corras el contador de personas, YOLO descarga su modelo
(6 MB) automáticamente. Es normal que se demore un poco más.

---

## Si prefieres la terminal

```bash
cd 01-contador-personas
python contador.py vtest.avi --motor yolo --speed 1
```

```bash
cd 02-lector-placas
python anpr.py autos.mp4 --formato ecuador --speed 1
```

```bash
cd 03-nivel-agua
python flood_monitor.py rio_test.mp4 --station 02 --speed 1
```
Todos los scripts comparten las mismas opciones:

| opción | para qué |
|---|---|
| `--speed 1` | reproducir a velocidad real (sin esto va a toda máquina) |
| `--headless` | sin ventana, para servidores o Colab |
| `--record salida.mp4` | grabar el video con las detecciones |
| `--start 10 --end 40` | procesar solo un tramo |
| `--stride 3` | procesar 1 de cada 3 cuadros (más rápido) |

Y todos aceptan `--help`.

---

## Qué hay en cada carpeta

```
01-contador-personas/
├── EJECUTAR.command        <- doble clic
├── contador.py             <- el código (un solo archivo)
├── EXPLICACION.md          <- cómo funciona, paso a paso
├── vtest.avi               <- video de entrada
├── linea.json              <- dónde está la línea de conteo
├── salida_anotada.mp4      <- resultado ya generado
├── conteo.csv              <- cada entrada y salida
└── ocupacion.csv           <- cuánta gente había en cada segundo
```

**Abre siempre el `EXPLICACION.md`.** El video llamativo es la carnada; lo que
importa está ahí.

---

## Tres ideas que se repiten

Si te llevas solo tres cosas de estos proyectos, que sean estas.

**1. Un cuadro no alcanza; el tiempo sí.**

El lector de placas acierta porque lee la misma placa 45 veces y vota. El
contador de personas funciona porque sigue a cada persona en vez de contar
cajas sueltas. El monitor de agua detecta el río porque el agua **se mueve
siempre**, no por su color en un instante.

En video, la dimensión temporal es información gratuita. Casi nadie la
aprovecha bien, y es donde está la mayor ganancia.

**2. El conocimiento del dominio le gana al modelo.**

El lector de placas corrige `0`→`O` porque **sabe** que las placas ecuatorianas
empiezan con tres letras. Ningún ajuste de preprocesamiento habría logrado eso.
El monitor de agua no usa ninguna red neuronal, y mide bien, porque entiende
qué distingue al agua de todo lo demás.

Antes de buscar un modelo más grande, pregúntate qué sabes tú del problema que
el modelo no sabe.

**3. Si no lo probaste, no está terminado.**

El período de calentamiento del monitor de agua existe porque en la primera
prueba el sistema reportó una tendencia de **−4382 metros por hora**. La
supresión de no-máximos del contador existe porque sin ella la misma persona se
contaba tres veces.

Ninguno de esos errores se ve leyendo el código. Se ven al ejecutarlo.

---

## Sobre los datos y las personas

El contador de personas y el lector de placas procesan información que
identifica a individuos. Si alguno de estos sistemas sale de esta carpeta hacia
un despliegue real, aparecen obligaciones concretas: aviso a quienes son
grabados, plazo de retención definido, control de quién consulta los registros,
y una razón legítima para recolectar.

Vale la pena discutirlo en clase antes que después. La pregunta técnica es *"¿se
puede hacer?"*. La pregunta profesional es *"¿debería hacerse, y bajo qué
condiciones?"*.

---

## De dónde sacar más videos

- `vtest.avi` y otras muestras de OpenCV:
  `https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/`
- Personas: [MOT Challenge](https://motchallenge.net/data/MOT20/)
- Placas: [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset),
  [UFPR-ALPR](https://github.com/raysonlaroca/ufpr-alpr-dataset)
- Ríos: cámaras en vivo de Japón, grabadas con `yt-dlp`

Pero el mejor video es el que grabes tú, con tu celular, en el lugar donde iría
la cámara. Los datasets públicos sirven para comparar; tu propio video sirve
para que funcione.
