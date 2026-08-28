# Splits generados

Generado por `scripts/build_splits.py` (semilla 20260827).

## Kaggle - A Large Scale Fish Dataset

Split a nivel de **foto original**, no de imagen: las ~34 rotaciones y
espejados de una misma foto caen siempre en la misma particion.

| clase | imgs | bloques | grupos | acuerdo medio | acuerdo min |
|---|---|---|---|---|---|
| Black Sea Sprat | 1000 | 50 | 50 | 1.000 | 1.000 |
| Gilt-Head Bream | 1000 | 50 | 48 | 0.970 | 0.600 |
| Hourse Mackerel | 1000 | 50 | 50 | 1.000 | 1.000 |
| Red Mullet | 1000 | 50 | 50 | 1.000 | 1.000 |
| Red Sea Bream | 1000 | 50 | 35 | 0.980 | 0.700 |
| Sea Bass | 1000 | 50 | 50 | 1.000 | 1.000 |
| Shrimp | 1000 | 50 | 49 | 0.982 | 0.700 |
| Striped Red Mullet | 1000 | 50 | 49 | 1.000 | 1.000 |
| Trout | 1000 | 30 | 30 | 0.998 | 0.941 |

Imagenes por particion: {'train': 6328, 'test': 1336, 'val': 1336}

`acuerdo` = fraccion de imagenes del bloque que votan al mismo original
por histograma de color. Es una verificacion independiente: la estructura
de bloques y el histograma no comparten informacion, asi que solo
coinciden si ambas identifican la misma foto de origen.

## Fishnet Open Images v1.0.0

Split **oficial** del CSV, respetado sin cambios (mitad del test
proviene de camaras no vistas en entrenamiento).

Recortes por particion: {'train': 72885, 'val': 12927, 'test': 13207}

Columnas de etiqueta: `label_l1` (especie), `label_l2` (grupo grueso),
`label_model` (l1 con la cola colapsada en `Other_rare`), `is_fish`
(0 para Human/No fish/Water/Oil).
