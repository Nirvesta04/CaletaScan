# Tabla de métricas: los cinco modelos comparados

Generada por `python -m scripts.tabla_metricas` a partir de `runs/*.json`. Los mismos números en formato plano: `runs/tabla_metricas.csv`.

## Modelos

| Modelo | Familia | Año | ImageNet top-1 | Parámetros | Macro-F1 | Margen 95 % | Semillas (0 / 1 / 2) | Acierto 15 especies | Cámaras vistas | Cámaras nuevas | Atunes agrupados | Reparto oficial (con fuga) | Min. entrenamiento |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | CNN | 2015 | 80,86 % | 23,5 M | **0,352** | 0,306 – 0,385 | 0,363 / 0,342 / 0,352 | 46 % | 0,355 | 0,253 | 0,378 | 0,616 | 30 |
| ViT-B/16 | transformer | 2020 | 81,07 % | 85,8 M | **0,393** | 0,349 – 0,424 | 0,389 / 0,413 / 0,379 | 55 % | 0,396 | 0,280 | 0,408 | 0,656 | 71 |
| DeiT-S | transformer | 2021 | 79,86 % | 21,7 M | **0,354** | 0,311 – 0,383 | 0,360 / 0,344 / 0,357 | 50 % | 0,372 | 0,245 | 0,367 | 0,625 | 45 |
| ConvNeXt-T | CNN | 2022 | 82,52 % | 27,8 M | **0,423** | 0,372 – 0,457 | 0,453 / 0,407 / 0,407 | 58 % | 0,426 | 0,336 | 0,437 | 0,675 | 39 |
| Swin-T | transformer | 2021 | 81,47 % | 27,5 M | **0,424** | 0,369 – 0,457 | 0,407 / 0,421 / 0,443 | 61 % | 0,421 | 0,338 | 0,441 | 0,673 | 40 |

## Comparaciones pareadas

Diferencia de macro-F1 (primero menos segundo). «Se distingue» = el margen no incluye el cero.

| Comparación | Qué se controla | Diferencia | Margen 95 % | ¿Se distingue? | En cámaras nuevas | Margen 95 % | ¿Se distingue? |
|---|---|---|---|---|---|---|---|
| DeiT-S − ResNet-50 | CNN vs transformer, mismo tamaño | +0,003 | −0,036 – 0,040 | no | −0,009 | −0,071 – 0,049 | no |
| Swin-T − ConvNeXt-T | CNN vs transformer, misma época | −0,001 | −0,029 – 0,027 | no | +0,009 | −0,032 – 0,055 | no |
| ViT-B/16 − ResNet-50 | par del enunciado | +0,042 | 0,007 – 0,077 | sí | +0,029 | −0,027 – 0,088 | no |
| ConvNeXt-T − ResNet-50 | CNN moderna vs antigua | +0,069 | 0,035 – 0,105 | sí | +0,080 | 0,022 – 0,141 | sí |
| Swin-T − ViT-B/16 | transformer moderno vs antiguo | +0,026 | −0,003 – 0,055 | no | +0,059 | 0,004 – 0,114 | sí |

## Cómo leer la tabla

- Todo sobre la **prueba limpia de FishNet**: 7.159 cajas de 612 lances, sin lances compartidos con entrenamiento. Promedio de 3 semillas; mismo presupuesto para los cinco (9.104 pasos, lote 64, 224 px).
- **Macro-F1**: promedio simple del F1 de las 15 especies (cada especie pesa lo mismo), con ajuste de logits τ elegido en validación y crédito jerárquico para la clase genérica `Tuna`. Un modelo al azar obtiene ≈ 0,02.
- **Margen 95 %**: bootstrap por clúster (2.000 réplicas remuestreando lances completos). Las comparaciones entre modelos son **pareadas**: los mismos lances remuestreados para los dos.
- **Acierto 15 especies**: porcentaje de peces de las 15 especies bien clasificados (1.450 cajas).
- **Cámaras vistas / nuevas**: cajas de cámaras que aparecen o no en entrenamiento (31 % / 69 % de la prueba).
- **Reparto oficial (con fuga)**: el mismo modelo evaluado en la prueba oficial, donde el 45,8 % de las cajas comparte lance con entrenamiento. Se muestra solo para ver cuánto infla la fuga; no es el resultado.
- **Parámetros**: contados en el punto de control, con la cabeza de 20 clases. **Min. entrenamiento**: semilla 0 en una RTX 3060 Ti.

## Conclusión

A igual tamaño o época, CNN y transformer no se distinguen (DeiT-S − ResNet-50 y Swin-T − ConvNeXt-T, diferencias de 0,003 y −0,001). Lo que sí mejora el resultado es pasar de un modelo antiguo a uno moderno: ConvNeXt-T supera a ResNet-50 por 0,069, siendo los dos CNN, y en cámaras nuevas la ventaja moderna se distingue en las dos familias. Los dos mejores modelos, ConvNeXt-T (0,423) y Swin-T (0,424), empatan.
