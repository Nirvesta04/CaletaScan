# Bitácora — Caleta-Scan (G3)

Estado al **27 de agosto de 2026**. Documento de traspaso: contiene todo lo necesario
para retomar el proyecto en una sesión nueva sin releer el historial.

**Fase actual: datos listos, modelos sin empezar.** El siguiente paso es implementar
los baselines de ResNet y ViT.

---

## 1. El proyecto

Ramo **Aprendizaje Automático II** (PUCV, prof. Christoffer Löffler). Grupo **G3 — Caleta-Scan**,
uno de siete proyectos de visión por computador en la Región de Valparaíso.

**Objetivo según el enunciado:** contar y clasificar especies (reineta, merluza, congrio)
en la Caleta Portales, como apoyo al monitoreo de Sernapesca.

**Requisito central:** entrenar y **comparar ResNet vs. ViT**. El enunciado sugiere que
ResNet clasifique por patrón de escamas/aleta y que ViT separe mejor peces amontonados
y superpuestos gracias a la atención global.

### Fechas y ponderación

| Instancia | Fecha | % nota final |
|---|---|---|
| Avance (presentación 15 min) | 28-sep / 1-oct | 5% |
| Final | 23 / 26-nov | 10% |

Desglose del Final: modelos comparados 25%, repo reproducible 20%, presentación 15%,
demo funcional 15%, informe técnico 15%, data card 10%.

**Para el Avance se exige tener ambos baselines ya entrenados** con métricas iniciales,
más problema/motivación, datos con criterio de etiquetado y split, y riesgos concretos
con mitigación.

---

## 2. Entorno

| | |
|---|---|
| GPU | NVIDIA RTX 3060 Ti, 8,6 GB — **CUDA operativo** |
| Python | 3.12.2 |
| torch | 2.6.0+cu124 · torchvision 0.21.0+cu124 |
| numpy / pillow / sklearn | 2.5.2 / 12.3.0 / 1.7.2 |
| Disco libre | ~121 GB |
| Enlace de red | ~1,2 MB/s en stream único, ~2,6 MB/s con 12 hilos |

Referencia: `matmul 4096² ×10` en GPU tarda 0,38 s.

**Cuidado:** el entorno traía torch CPU-only y hubo que reinstalarlo con
`--index-url https://download.pytorch.org/whl/cu124`. Eso arrastró numpy de la serie 1.x
a la 2.x y bajó torch de 2.13 a 2.6 (el índice cu124 no publica más nuevo). Todo lo del
proyecto se verificó funcionando después del cambio. Si aparece un
`numpy.dtype size changed` en algún paquete nuevo, viene de ahí.

---

## 3. Datos

```
data/
├── raw/
│   ├── Fish_Dataset/Fish_Dataset/     Kaggle: 9 clases × 1000 RGB + 1000 máscaras GT
│   ├── NA_Fish_Dataset/               430 fotos originales sin aumentar (50/clase, 30 Trout)
│   └── fishnet/
│       ├── images/                    20.002 jpg (4,49 GB)
│       ├── foid_labels_v100.csv       549.209 cajas, 34 clases, split oficial
│       ├── subset_manifest.csv        qué se descargó y por qué
│       └── zip_index_subset.json      caché del índice del ZIP remoto
└── splits/
    ├── kaggle_split.csv               path, clase, group_id, split
    ├── kaggle_groups.csv              validación del agrupamiento
    ├── fishnet_crops.csv              path, caja, etiquetas, split
    └── README.md
```

`archive.zip` (3,48 GB) sigue en la raíz; es la descarga original de Kaggle y se puede borrar.

### Fuentes y licencias — para la data card (10%)

| Fuente | Contenido | Licencia | Cita |
|---|---|---|---|
| Kaggle *A Large Scale Fish Dataset* | 9 especies, 1000 img + 1000 máscaras c/u, 590×445 | **CC BY 4.0** | Ulucan, Karakaya & Turkan, ASYU 2020 |
| Fishnet Open Images v1.0.0 | subconjunto de 20.002 img, resoluciones variadas | **CC BY-SA 4.0** | The Nature Conservancy (2022) |
| OzFish | no descargado | **CC BY 3.0 AU** | AIMS/UWA/Curtin (2019) |

Las tres licencias son distintas y hay que declararlas. **CC BY-SA es vírica**: un derivado
combinado que se redistribuya arrastraría ShareAlike. Vale mencionarlo en el informe.

El de Kaggle se tomó en un **supermercado de Izmir, Turquía**, con dos cámaras
(Kodak Z650 y Samsung ST60), redimensionado a 590×445 y aumentado por espejado y rotación.

---

## 4. Decisiones metodológicas ya tomadas

Estas están discutidas y cerradas. **No re-litigar sin motivo nuevo.**

### 4.1 El dataset de Kaggle tiene fuga de datos si se splitea al azar

Las 1000 imágenes por clase **no son 1000 fotos**: son rotaciones de ángulo arbitrario y
espejados de solo 30-50 fotos originales. Un split aleatorio deja rotaciones de la misma
foto en train y en test, y la accuracy sube cerca del 100% sin que el modelo generalice.

Estructura descubierta empíricamente: las aumentaciones están en **bloques contiguos de
tamaño `ceil(1000/n_originales)`** — 20 imágenes por original en las clases de 50 originales,
34 en Trout que tiene 30 (el último bloque queda truncado a 14).

El agrupamiento en `scripts/build_splits.py` cruza dos evidencias independientes: la
estructura de bloques y un histograma de color invariante a rotación contra
`NA_Fish_Dataset`. Su nivel de acuerdo se reporta como validación. Cuando dos bloques
apuntan al mismo original **se fusionan** — conservador a propósito.

| clase | bloques | grupos | acuerdo |
|---|---|---|---|
| Black Sea Sprat | 50×20 | 50 | 1,000 |
| Gilt-Head Bream | 50×20 | 48 | 0,970 |
| Hourse Mackerel | 50×20 | 50 | 1,000 |
| Red Mullet | 50×20 | 50 | 1,000 |
| Red Sea Bream | 50×20 | **35** | 0,980 |
| Sea Bass | 50×20 | 50 | 1,000 |
| Shrimp | 50×20 | 49 | 0,982 |
| Striped Red Mullet | 50×20 | 49 | 1,000 |
| Trout | 30×34 | 30 | 0,998 |

**Split resultante:** 6.328 train / 1.336 val / 1.336 test (287/62/62 grupos),
estratificado por clase. Verificado: **0 grupos cruzan particiones**.

### 4.2 Fishnet: se descargó un subconjunto, no el dataset completo

El ZIP de imágenes pesa 36,4 GB. El S3 acepta *range requests*, así que
`scripts/download_fishnet_subset.py` baja solo los miembros necesarios del ZIP remoto.

**Criterio, definido a priori:** se incluye toda imagen que contenga al menos una caja de
una clase fuera de las 6 cabeza (Human, Yellowfin tuna, Albacore, Skipjack tuna,
Bigeye tuna, No fish), más las imágenes de "No fish".

Resultado: 20.002 imágenes (13,9% de los bytes) que conservan el **100% de las cajas de las
28 clases de cola** y las 34 clases presentes. Solo las 5 clases cabeza quedan submuestreadas,
y aun así llegan ~10.043 aleta amarilla, ~6.851 albacora, ~3.023 barrilete y ~53.784 humano
por venir en las mismas fotos.

Esto **no es trampa**: la decisión se tomó antes de entrenar, está documentada, se aplica
igual a ResNet y a ViT, y no se filtró el test según el desempeño del modelo. Es submuestreo
de clases cabeza y va declarado en la data card.

### 4.3 Espacio de etiquetas de Fishnet

El dataset trae 34 clases nominales, pero **6 tienen CERO cajas en el test oficial**
(Mola mola, Roudie scolar, Snake mackerel, Rainbow runner, Thresher shark, Brama) y otras
tres son casi inservibles (Oilfish 3 cajas de test, Sickle pomfret 8, Lancetfish 9).
"Usar las 34 clases como vienen" no existe como opción evaluable.

`fishnet_crops.csv` trae cuatro columnas para decidir sin regenerar nada:

- `label_l1` — especie (34 clases)
- `label_l2` — grupo grueso (15 clases)
- `label_model` — l1 con la cola colapsada: **19 clases + `Other_rare`**
- `is_fish` — 0 para Human / No fish / Water / Oil

**Recomendación pendiente de ejecutar:** modelar jerárquicamente y reportar ambos niveles.
Colapsar la cola en `Other_rare` en vez de borrarla preserva el análisis de cola larga, que
es donde ResNet y ViT suelen diferir — material directo para el 25% de "comparación
discutida con evidencia, no solo tabla".

### 4.4 OzFish no se usa para entrenar

Rol acordado: **sondeo fuera de distribución**, no fuente de entrenamiento.

Razones: es submarino (BRUVS, peces vivos nadando en agua turbia) — la brecha de dominio es
mayor que la de las otras dos fuentes; son 507 especies de arrecife australiano que no se
solapan con ninguna otra fuente; los recortes vienen de frames de video y muchos serán de
pocas decenas de píxeles.

Como los espacios de etiquetas son disjuntos, **no se puede calcular accuracy sobre OzFish**.
Lo que sí mide, y es lo valioso:

1. **Calibración de confianza** ante entradas imposibles — ¿el modelo dice "trucha" con 0,98
   o duda? Comparar la distribución de max-softmax entre ResNet y ViT.
2. **Grad-CAM / attention rollout** sobre esas imágenes — ¿siguen mirando al pez o se
   enganchan al fondo? Si el modelo aprendió "fondo blanco de estudio = trucha", con agua
   turbia se ve al instante. Ataca directo la interpretabilidad que pide el 25%.
3. Análisis del espacio de embeddings.

**Descarga pendiente:** `crops-zip.zip`, 3,18 GB, ~20 min. No bajar antes de la fase de
interpretabilidad. Las otras piezas son `frames.zip` (106 GB, descartada por tamaño) y
`labelled-zip.zip` (2,91 GB).

Está alojada en **storage.pawsey.org.au/public/m/FDFML**, no en fishnet.ai. Ojo: ese portal
es una SPA con API por token y los archivos **no** se pueden bajar por URL directa —
lo intenté por ruta directa, por API con Bearer y por el endpoint de listado, y devuelven
404 y 401. Habrá que resolver la autenticación o bajarlo a mano desde el navegador.

---

## 5. Hallazgos que afectan el modelado

- **El dataset de Kaggle no soporta "contar".** Cada imagen tiene un solo pez sobre fondo
  liso; no hay peces amontonados ni superpuestos. La parte de conteo y oclusión del objetivo
  declarado no se puede entrenar con esa fuente. Fishnet sí tiene cubierta con peces
  amontonados (~4 cajas por imagen).
- **Ninguna especie chilena está en ningún dataset.** La brecha de dominio es obligatoria en
  el informe y tiene fila propia en la rúbrica del Final.
- **Anomalía en el split oficial de Fishnet:** Oilfish tiene 82 cajas en train pero 693 en val.
  Por eso `Other_rare` sale con más ejemplos en val que en train (539 / 756 / 105). Es del
  split oficial, no del filtrado nuestro, pero hay que mencionarlo si se usa esa clase.
- **Desbalance brutal en Fishnet:** Human es el 51% de todas las cajas; de los peces, ~90% son
  cuatro especies de atún. Métricas macro sí o sí, no solo accuracy.
- **Resoluciones heterogéneas.** Kaggle es uniforme 590×445; Fishnet va de 1200×675 a
  1920×1030. El pipeline de preprocesamiento tiene que normalizar.
- **Las máscaras GT de Kaggle están disponibles** y son un activo: permiten medir cuánto
  depende el modelo del fondo blanco de estudio.

---

## 6. Qué falta

**Inmediato (bloquea el Avance del 28-sep):**

1. Baseline **ResNet** sobre `data/splits/kaggle_split.csv` con métricas.
2. Baseline **ViT** sobre el mismo split.
3. **Demostración de la fuga:** entrenar lo mismo con split aleatorio y poner las dos
   accuracies lado a lado. Es la evidencia para la fila "riesgos identificados" del Avance,
   y probablemente el resultado más vendible de la presentación.

**Después:**

4. Entrenar sobre Fishnet (`fishnet_crops.csv`) — decidir `label_model` vs `label_l2`.
5. Interpretabilidad: Grad-CAM en ResNet, attention rollout en ViT.
6. Sondeo OOD con OzFish (bajar los crops entonces).
7. Demo Gradio/Streamlit (15% del Final).
8. Data card (10%) — buena parte del material ya está en la sección 3 de este documento.
9. Informe técnico y presentación final.

**Recordatorio de rúbrica:** el repo vale 20% y exige **commits incrementales visibles**.
El proyecto todavía no es un repositorio git — conviene inicializarlo antes de empezar a
programar los modelos, no al final.

---

## 7. Scripts

| Script | Qué hace |
|---|---|
| `scripts/download_fishnet_subset.py` | Descarga selectiva del ZIP remoto de Fishnet por range requests. Reanudable, valida cada JPEG, genera manifiesto. |
| `scripts/build_splits.py` | Recupera los grupos de aumentación de Kaggle y genera los tres CSV de `data/splits/`. |

Ambos son idempotentes y se pueden volver a correr:

```bash
python scripts/build_splits.py
```
