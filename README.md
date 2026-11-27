# Caleta-Scan — clasificación de especies de pescado: CNN contra transformer

Proyecto Aplicado del ramo **Aprendizaje Automático II** (PUCV, prof. Christoffer Löffler).
Grupo **G3 — Caleta-Scan**: Patricio Toledo, Andrés Saavedra, Sebastián Casas y Bastián Molina.

Clasificar especies de pescado en imágenes de faena, como apoyo al monitoreo de
Sernapesca en la Caleta Portales. El requisito central del curso es **entrenar y
comparar ResNet contra ViT**, con interpretabilidad (Grad-CAM / attention rollout).
Para separar el efecto de la arquitectura del de tamaño y antigüedad, se comparan
cinco modelos en tres pares con preentrenamiento parejo.

> **Nota sobre las especies.** Reineta, merluza y congrio son el caso de uso del
> enunciado, no el espacio de etiquetas: ninguna aparece en los datasets públicos
> disponibles. Se entrena sobre las especies que sí existen (9 en Kaggle; en FishNet,
> 20 clases, de las cuales 15 son especies). Con 45 fotos de iNaturalist se prueba qué
> hacen los modelos frente a especies chilenas.

---

## Estado (entrega final)

| | |
|---|---|
| Datos y splits | construidos y verificados (0 fugas de grupo en Kaggle; test de FishNet purgado por secuencia) |
| Par original · ResNet-50 vs ViT-B/16 | FishNet limpio: ViT +0,042 de macro-F1, margen [+0,007, +0,077] |
| Mismo tamaño · ResNet-50 vs DeiT-S | DeiT-S 0,354; diferencia +0,003 [-0,036, +0,040]: no se distinguen |
| Par moderno · ConvNeXt-T vs Swin-T | 0,423 vs 0,424; diferencia -0,001 [-0,029, +0,027]: no se distinguen |
| Moderno vs antiguo | ConvNeXt-T − ResNet-50 +0,069 [+0,035, +0,105]: lo que mejora el resultado es un modelo moderno, no la familia |
| Qué significan | con un buen modelo, el límite es la foto: la ventaja moderna crece con el tamaño del pez (+6 a +20 puntos de acierto), la cámara cambia el acierto de 36 % a 83 %; ver `scripts/investigacion.py` |
| Robustez | con los atunes agrupados y en cámaras no vistas, la ventaja del ViT deja de distinguirse |
| Kaggle | ResNet y ViT no se distinguen; sin ver el pez, las dos redes aciertan ~97 % |
| Interpretabilidad | Kaggle (máscaras) y FishNet (cajas, pez en posición al azar y control barajado) |
| «No sé» | umbral de energía elegido en validación; el ViT rechaza mejor peces ajenos y especies chilenas |
| Velocidad | ViT-B/16 2,7× más lento en CPU (119 vs 43 ms por pez) sin cambiar qué es factible |
| Demo | video de 2 minutos en `docs/Caleta-Scan-demo.mp4`; en vivo, `python -m src.demo` compara los cinco modelos sobre la misma caja, con «no sé» |
| Informe técnico | `docs/informe.html` y `docs/informe.pdf` (6 páginas) |
| Data card | media página en `docs/data-card-resumen.pdf`; versión extendida en `docs/data-card.pdf` |
| Tabla de métricas | `docs/tabla-metricas.md` (también `.pdf` y `runs/tabla_metricas.csv`) |

---

## Entregables del Final

| Entregable | Dónde |
|---|---|
| Repositorio de código | este repositorio: [`README.md`](README.md) y [`requirements.txt`](requirements.txt) |
| Data card (media página) | [`docs/data-card-resumen.pdf`](docs/data-card-resumen.pdf): origen, licencias, número de imágenes, división train/val/test y sesgos conocidos. Versión extendida: [`docs/data-card.pdf`](docs/data-card.pdf) |
| Modelos comparados y tabla de métricas | [`docs/tabla-metricas.md`](docs/tabla-metricas.md) · [`.pdf`](docs/tabla-metricas.pdf) · [`runs/tabla_metricas.csv`](runs/tabla_metricas.csv); configuración de cada modelo en `configs/e4`, `e5`, `e10`, `e11` y `e12`, y sus métricas por semilla en `runs/<exp>/seed<n>/metrics.json` |
| Demo funcional | **Video de 2 minutos:** [`docs/Caleta-Scan-demo.mp4`](docs/Caleta-Scan-demo.mp4), la demo en acción sobre fotos que los modelos no vieron. **En vivo:** [`src/demo.py`](src/demo.py) + [`src/demo.html`](src/demo.html): `python -m scripts.bajar_pesos` y luego `python -m src.demo`, abrir http://127.0.0.1:8000 (`--difuminar` para mostrarla en público). Clasifica la misma caja con los cinco modelos |
| Presentación interactiva (ExpoData) | https://claude.ai/artifact/KhJuDhZtCWiNUYYNfiQxQU: los resultados para público general, con juegos y controles. Complementa la demo: no corre los modelos, sus cifras vienen de `runs/` |
| Informe técnico (6 páginas) | [`docs/informe.pdf`](docs/informe.pdf) · [`docs/informe.html`](docs/informe.html) |
| Láminas de la presentación final | [`docs/Caleta-Scan-Final-G3.pdf`](docs/Caleta-Scan-Final-G3.pdf) · [`.pptx`](docs/Caleta-Scan-Final-G3.pptx) |

**Los cinco modelos, prueba limpia de FishNet** (macro-F1 sobre 15 especies, promedio de 3 semillas,
margen 95 % por bootstrap de lances):

| Modelo | Familia | Parámetros | Macro-F1 | Margen 95 % | Cámaras nuevas |
|---|---|---|---|---|---|
| Swin-T | transformer | 27,5 M | **0,424** | 0,369 – 0,457 | 0,338 |
| ConvNeXt-T | CNN | 27,8 M | **0,423** | 0,372 – 0,457 | 0,336 |
| ViT-B/16 | transformer | 85,8 M | 0,393 | 0,349 – 0,424 | 0,280 |
| DeiT-S | transformer | 21,7 M | 0,354 | 0,311 – 0,383 | 0,245 |
| ResNet-50 | CNN | 23,5 M | 0,352 | 0,306 – 0,385 | 0,253 |

La tabla completa, con las comparaciones pareadas, está en `docs/tabla-metricas.md`.

**Pesos.** Los puntos de control (`runs/<exp>/seed<n>/checkpoint.pt`, de 83 MB a 328 MB
cada uno) no se versionan por tamaño. Los de la semilla 0 de los cinco modelos, junto con
los logits que la demo usa para fijar τ y el umbral del «no sé», están en el
[release v1.0](https://github.com/Nirvesta04/CaletaScan/releases/tag/v1.0) (742 MB):
`python -m scripts.bajar_pesos` los baja, verifica y deja en `runs/`. Sin los datos crudos
la demo no muestra fotos de ejemplo, pero clasifica las que se suban. Para regenerar todo
desde cero, el paso 4 de «Cómo reproducir».

---

## Cómo reproducir

### 1. Entorno

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Debe imprimir `2.6.0+cu124 True`. Si imprime `False` o una versión sin `+cu124`, pip
instaló la variante CPU-only: reinstalar torch desde el índice indicado en
`requirements.txt`. Es un fallo silencioso, no lanza ningún error.

### 2. Datos

Los datos crudos (7,6 GB) **no están versionados**. Se obtienen así:

- **Kaggle — A Large Scale Fish Dataset:** descargar desde Kaggle y extraer en
  `data/raw/`, dejando `Fish_Dataset/` y `NA_Fish_Dataset/`.
- **FishNet Open Images:** `python scripts/download_fishnet_subset.py`. Descarga
  selectiva por *range requests* sobre el ZIP remoto (4,49 GB de los 36,4 GB totales).
  Es reanudable y valida cada JPEG.
- **Fotos chilenas:** las URL de observación están en `data/chile_creditos.csv`; las
  imágenes van en `data/raw/chile/<especie>/`.

### 3. Splits

```bash
python scripts/build_splits.py
```

Idempotente. Regenera los CSV de `data/splits/`, que **sí están versionados** porque
son el contrato del experimento.

### 4. Entrenamiento

```bash
python -m src.train --config configs/e4_resnet50_fishnet.yaml   # uno por YAML de configs/
python -m scripts.entrenar_pares                                # E10, E11 y E12 en orden
python -m scripts.entrenar_pares --parar                        # se detiene al cerrar la época
```

`train.py` guarda un punto de control por época y salta las semillas completas: una
corrida se puede cortar y relanzar con el mismo comando sin perder trabajo.

### 5. Análisis, figuras y entregables

Todo esto trabaja sobre los logits guardados en `runs/` y no usa la GPU, salvo la
interpretabilidad, la inferencia OOD y la prueba chilena (minutos).

```bash
python -m scripts.analisis_final            # pares, por especie, atunes, cámaras, épocas
python -m scripts.tabla_metricas            # docs/tabla-metricas.md y runs/tabla_metricas.csv
python -m src.evaluate --run e5_vit_b_16_fishnet --compare e4_resnet50_fishnet --limpio
python -m src.interpret                     # pointing game en Kaggle
python -m scripts.interpret_fishnet         # interpretabilidad en FishNet
python -m src.calibrate --inferir && python -m src.calibrate    # calibración y OOD
python -m scripts.chile_ood                 # especies chilenas
python -m src.speed                         # latencias (máquina libre)
python scripts/auditoria.py                 # fuga, duplicados y sobreajuste
python -m tests.test_stats && python -m tests.test_calibrate
python -m src.demo                          # demo en http://127.0.0.1:8000
python -m scripts.build_informe             # docs/informe.html y .pdf
python -m scripts.figuras_mazo && python -m scripts.graficos_mazo && python -m scripts.build_pptx_final   # mazo en entrega/
```

El mazo se escribe en `entrega/`, que no se versiona; la versión presentada está copiada en
`docs/Caleta-Scan-Final-G3.pptx` y `.pdf` (ver «Privacidad»).

---

## Estructura

```
configs/        un YAML por corrida (E1…E12): el YAML es el experimento
data/
  raw/          datos crudos (ignorado por git, 7,6 GB)
  splits/       manifiestos de split (versionados)
  chile_creditos.csv  autor y licencia de las 45 fotos chilenas
docs/
  informe.html / .pdf  informe técnico final (4 a 6 páginas)
  data-card-resumen.pdf  data card de media página
  Caleta-Scan-demo.mp4  video de la demo (2 min)
  data-card.html / .pdf  qué contienen los datos, de dónde salieron, qué curamos
                      nosotros y qué no se puede concluir con ellos
  tabla-metricas.md / .pdf  los cinco modelos y sus comparaciones pareadas
  plan-modelos.html   protocolo de modelado del Avance, con su discusión completa
  Caleta-Scan-Final-G3.pdf / .pptx  la presentación final
  contexto-para-ia.md archivo de contexto para estudiar el proyecto con un asistente
  figs/               figuras sin fotos (matrices de confusión, diferencias)
  Caleta-Scan-Avance-G3.pptx  la presentación del Avance
runs/<exp>/seed<n>/   config.yaml y metrics.json versionados; logits y pesos no
runs/*.json           resultados agregados de cada análisis
scripts/
  download_fishnet_subset.py, build_splits.py
  bajar_pesos.py      baja del release los pesos que usa la demo
  entrenar_pares.py   entrena E10-E12, se puede detener y retomar
  analisis_final.py   tablas de la entrega final
  tabla_metricas.py   tabla de métricas de los cinco modelos
  interpret_fishnet.py, chile_ood.py
  auditoria.py        fuga, duplicados por píxeles y sobreajuste
  build_informe.py, figuras_mazo.py, build_pptx_final.py, build_pptx.py
src/
  data.py       datasets, transforms y loaders
  models.py     fábrica de modelos con pesos fijados y grupos de parámetros
  train.py      entrenamiento reanudable, dirigido por --config
  evaluate.py   métricas, ajuste de logits, bootstrap de clúster y comparación pareada
  interpret.py  Grad-CAM, attention rollout, relevancia de Chefer y pointing game
  calibrate.py  temperatura, ECE/NLL/Brier y OOD cruzado Kaggle↔FishNet
  speed.py      latencia GPU/CPU, MACs medidos y tamaño de los modelos
  demo.py       demo local (sin dependencias nuevas) + demo.html, su interfaz
tests/          validación de la maquinaria estadística contra sklearn
BITACORA.md     documento de traspaso: estado, decisiones cerradas y hallazgos
```

---

## Fuentes de datos y licencias

| Fuente | Contenido usado | Licencia | Cita |
|---|---|---|---|
| Kaggle — *A Large Scale Fish Dataset* | 9 especies, 1000 img + 1000 máscaras GT c/u, 590×445 | **CC BY 4.0** | Ulucan, Karakaya & Turkan, ASYU 2020 |
| FishNet Open Images v1.0.0 | subconjunto de 20.002 img, 99.019 recortes | **CC BY-SA 4.0** | Kay & Merrifield, 2021; The Nature Conservancy |
| iNaturalist | 45 fotos de prueba (reineta, merluza, congrio) | **CC0, CC BY o CC BY-NC** según la foto | autores en `data/chile_creditos.csv` |
| OzFish | no descargado | CC BY 3.0 AU | AIMS/UWA/Curtin, 2019 |

**CC BY-SA es vírica:** un derivado combinado que se redistribuya arrastraría ShareAlike.

**Privacidad.** FishNet es monitoreo electrónico a bordo y el 54 % de su entrenamiento
son tripulantes. No se redistribuyen los datos crudos ni las figuras de trabajo con fotos
(`docs/figs_privadas/`). Las únicas fotos publicadas son las de la presentación final, el video
de la demo y la presentación de la ExpoData: pocas, con las caras difuminadas, sin los fotogramas que traen el aviso de
confidencialidad del proveedor y con la atribución CC BY-SA 4.0 de FishNet.

---

## Decisiones metodológicas

Están razonadas en `BITACORA.md` §4 y en `docs/plan-modelos.html`. Las que más
condicionan el código:

1. **El split de Kaggle es por foto original, no por imagen.** Las 1000 imágenes por
   clase son rotaciones y espejados de 30–50 fotos. Un split aleatorio deja rotaciones
   de la misma foto en train y test y la accuracy sube a ~100% sin generalizar.
2. **El split oficial de FishNet filtra por secuencia.** No está agrupado por `seq_id`:
   en nuestro subconjunto, el 45,8 % de las cajas de test (39,1 % de las imágenes)
   comparte secuencia (= fotogramas del mismo lance) con train. Se reporta sobre el
   test purgado.
3. **Los pesos preentrenados van fijados explícitamente**, nunca `DEFAULT`, y cada par
   tiene preentrenamiento parejo: ResNet-50 `IMAGENET1K_V2` (80,86) vs. ViT-B/16
   `IMAGENET1K_V1` (81,07); DeiT-S (79,86); ConvNeXt-T (82,52) vs. Swin-T (81,47).
4. **Los intervalos de confianza se calculan por bootstrap de clúster**, sobre la foto
   original en Kaggle y sobre la secuencia en FishNet. El test de Kaggle tiene n = 62
   unidades independientes, no 1336.
5. **El umbral del «no sé» se elige en validación**, nunca en el test.
