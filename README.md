# Caleta-Scan — clasificación de especies de pescado con ResNet vs. ViT

Proyecto Aplicado del ramo **Aprendizaje Automático II** (PUCV, prof. Christoffer Löffler).
Grupo **G3 — Caleta-Scan**.

Clasificar especies de pescado en imágenes de faena, como apoyo al monitoreo de
Sernapesca en la Caleta Portales. El requisito central del curso es **entrenar y
comparar ResNet contra ViT**, con interpretabilidad (Grad-CAM / attention rollout).

> **Nota sobre las especies.** Reineta, merluza y congrio son el caso de uso del
> enunciado, no el espacio de etiquetas: ninguna aparece en los datasets públicos
> disponibles. Se entrena sobre las especies que sí existen en los datos (9 en Kaggle,
> 17 taxones modelados en FishNet) y la brecha de dominio se discute explícitamente
> en el informe, donde es una fila propia de la rúbrica.

---

## Estado

| | |
|---|---|
| Datos y splits | construidos y verificados (0 fugas de grupo en Kaggle; test de FishNet purgado por secuencia) |
| E1–E2 · Kaggle | hechos; ResNet y ViT no se distinguen (el test tiene 62 fotos independientes) |
| E4–E6 · FishNet | hechos; ViT > ResNet en macro-F1, Δ +0,042 con IC que no cruza cero |
| E7 · controles de fondo | hechos; sin ver el pez, las dos redes aciertan ~97 % en Kaggle |
| Interpretabilidad | Grad-CAM, attention rollout y Chefer, con pointing game contra máscaras GT |
| Calibración y OOD | hechos; el ViT rechaza peces ajenos mucho mejor (AUROC +0,21) |
| Velocidad vs. precisión | hecha; ViT 2,7× más lento en CPU (119 vs 43 ms) sin cambiar qué es factible: se elige el ViT |
| Demo | funcional: `python -m src.demo` levanta la comparación en vivo con rechazo de «no sé» |
| Data card | `docs/data-card.html`: fuentes, composición, curación, fugas medidas y usos apropiados |
| Pendiente | informe; E9 opcional |
| Próxima entrega | Avance, 28-sep-2026 (presentación de 15 min) |

Todos los resultados, con sus intervalos y su discusión, están en `docs/plan-modelos.html`.

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

### 3. Splits

```bash
python scripts/build_splits.py
```

Idempotente. Regenera los tres CSV de `data/splits/`, que **sí están versionados**
porque son el contrato del experimento.

### 4. Experimentos y análisis

```bash
python -m src.train --config configs/e4_resnet50_fishnet.yaml   # uno por YAML de configs/
python -m src.evaluate --run e5_vit_b_16_fishnet --compare e4_resnet50_fishnet --limpio
python -m src.interpret                                         # pointing game
python -m src.calibrate --inferir && python -m src.calibrate    # calibración y OOD
python -m src.speed                                             # latencias (máquina libre)
python -m src.demo                                              # demo en http://127.0.0.1:8000
python -m tests.test_stats && python -m tests.test_calibrate    # validación estadística
```

`train.py` reanuda por época lo que quedó a medias y salta las semillas completas, así
que una corrida se puede cortar y relanzar con el mismo comando. Todo lo posterior al
entrenamiento trabaja sobre los logits guardados en `runs/` y no necesita GPU, salvo la
inferencia OOD cruzada (5 min).

---

## Estructura

```
configs/        un YAML por corrida (E1…E7): el YAML es el experimento
data/
  raw/          datos crudos (ignorado por git, 7,6 GB)
  splits/       manifiestos de split (versionados)
docs/
  plan-modelos.html   protocolo de modelado y resultados: qué se entrena, con qué
                      receta, contra qué métrica y con qué prueba estadística
  data-card.html      qué contienen los datos, de dónde salieron, qué curamos
                      nosotros y qué no se puede concluir con ellos
runs/<exp>/seed<n>/   config.yaml y metrics.json versionados; logits y pesos no
scripts/
  download_fishnet_subset.py
  build_splits.py
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
| FishNet Open Images v1.0.0 | subconjunto de 20.002 img, 99.019 recortes | **CC BY-SA 4.0** | The Nature Conservancy, 2022 |
| OzFish | no descargado; rol previsto: sondeo fuera de distribución | **CC BY 3.0 AU** | AIMS/UWA/Curtin, 2019 |

Las tres licencias son distintas y deben declararse por separado.
**CC BY-SA es vírica:** un derivado combinado que se redistribuya arrastraría ShareAlike.

---

## Decisiones metodológicas

Están razonadas en `BITACORA.md` §4 y en `docs/plan-modelos.html`. Las cuatro que más
condicionan el código:

1. **El split de Kaggle es por foto original, no por imagen.** Las 1000 imágenes por
   clase son rotaciones y espejados de 30–50 fotos. Un split aleatorio deja rotaciones
   de la misma foto en train y test y la accuracy sube a ~100% sin generalizar.
2. **El split oficial de FishNet filtra por secuencia.** No está agrupado por `seq_id`:
   el 44,5 % de las cajas de test comparte secuencia (= fotogramas del mismo lance) con
   train. Se reportan las métricas sobre el test oficial **y** sobre el test purgado.
3. **Los pesos preentrenados van fijados explícitamente**, nunca `DEFAULT`.
   `ResNet50_Weights.DEFAULT` apunta a V1 (76,13 % en ImageNet) mientras ViT-B/16 está
   en 81,07 %: comparar esos dos mide la receta de preentrenamiento, no la arquitectura.
   El par correcto es ResNet-50 `IMAGENET1K_V2` (80,86) vs. ViT-B/16 `IMAGENET1K_V1`.
4. **Los intervalos de confianza se calculan por bootstrap de clúster**, sobre la foto
   original en Kaggle y sobre la secuencia en FishNet. El test de Kaggle tiene n = 62
   unidades independientes, no 1336.

---

## Equipo

Grupo G3 — Caleta-Scan · Aprendizaje Automático II · PUCV · 2026-2
