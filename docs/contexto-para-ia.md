# Caleta-Scan (Grupo G3) — Archivo de contexto para asistentes de IA

> **Para la IA que lee esto:** este documento contiene todo el estado real de un proyecto
> universitario. Tu tarea es **ayudar a un integrante del grupo a entenderlo y a prepararse
> para defenderlo en una presentación oral**. Reglas:
>
> 1. **No inventes números.** Todas las cifras del proyecto están acá. Si te preguntan algo
>    que no aparece, dilo: "eso no está en el contexto que me dieron".
> 2. **Explica en lenguaje simple.** El estudiante puede no dominar la jerga de machine
>    learning. Evita fórmulas salvo que te las pidan.
> 3. **Ojo con las confusiones de la sección 10.** Son errores que ya cometieron integrantes
>    del grupo. Si detectas que el estudiante cae en una, corrígela de inmediato.
> 4. **Si te piden practicar**, haz de profesor evaluador y pregunta cosas incómodas.
>
> Documento escrito el 2026-09-20 y actualizado el 2026-11-14 para la entrega final (16-nov-2026).
> **Las cifras y el mensaje de la entrega final están en las secciones 1 y 7.** Si algo de otra
> sección contradice esas dos, mandan esas dos.

---

## 1. Qué es el proyecto

**Caleta-Scan** es un proyecto del ramo *Aprendizaje Automático II* (PUCV). La idea aplicada:
en la **Caleta Portales (Valparaíso)**, Sernapesca anota a mano qué especie y cuánto se
desembarca. Es lento y depende de cuántos inspectores haya ese día. La propuesta es que una
foto de la faena entregue una lista de especies con su nivel de confianza, que el inspector
revisa y corrige.

**El trabajo académico real es una comparación de arquitecturas:**

> ¿Rinde mejor una **red convolucional (CNN)** o un **transformer de visión** clasificando
> especies de pescado, con el mismo presupuesto de entrenamiento?

Se compararon **cinco modelos en tres pares**, cada par con pesos de partida de calidad pareja:

| Par | CNN | Transformer | Qué responde |
|---|---|---|---|
| Original | ResNet-50 (2015) | ViT-B/16 (2020) | La pregunta del curso |
| Mismo tamaño | ResNet-50 | DeiT-S (22 M parámetros, como el ResNet) | ¿El ViT ganaba por ser transformer o por ser 3,4 veces más grande? |
| Moderno | ConvNeXt-T (2022) | Swin-T (2021) | ¿El ResNet perdía por ser antiguo? |

**Mensaje central de la entrega final:**

> **A igual tamaño y época, CNN y transformer empatan. Lo que mejora el resultado es usar un
> modelo moderno, no elegir entre CNN y transformer.**

El Avance decía «el ViT gana». Eso sigue siendo cierto para el par original, pero los pares
nuevos muestran que esa ventaja no viene de la atención del transformer.

### Alcance honesto (esto es importante en la nota)

**Qué SÍ hace:**
- Decir la especie de un pez **cuyo recuadro ya está marcado** en la foto.
- Comparar las dos arquitecturas con presupuesto idéntico.
- Contestar «no sé» cuando la imagen no se parece a nada de lo que aprendió.

**Qué NO hace:**
- **No detecta.** No encuentra los peces solo. Alguien tiene que dibujar el recuadro primero
  (en FishNet vienen anotados; en la demo los marca el usuario con el mouse). Un detector es
  otro modelo distinto y no está en el alcance.
- **No reconoce especies chilenas** (reineta, merluza, congrio): no existen en datos públicos.
- **No funciona en el muelle todavía**: habría que recolectar fotos locales primero.

Lo que se demuestra es **el método**, no un producto listo.

---

## 2. Los dos conjuntos de datos

### Kaggle — "A Large Scale Fish Dataset"
- 9.000 imágenes, 9 especies, 1.000 imágenes por especie.
- Pescado sobre una mesa, fondo de estudio, luz pareja.
- Trae la **máscara de segmentación** (la silueta exacta del pez), lo que permite experimentos
  que en el otro conjunto son imposibles.
- **Trampa clave:** esas 1.000 imágenes por especie son en realidad **30 a 50 fotos originales**
  rotadas y espejadas (~21 copias por foto). En total hay **411 fotos originales**, no 9.000.
- **Papel en el proyecto:** laboratorio de diagnóstico. Se entrena y se evalúa en él, pero
  **no se usa para decidir** qué arquitectura es mejor.

### FishNet Open Images
- Subconjunto usado: **20.002 imágenes**, **99.019 recortes** (cajas de peces o personas),
  **77 cámaras**, **4.624 secuencias**.
- Monitoreo electrónico a bordo de barcos pesqueros reales: de noche, cubierta mojada, peces
  amontonados y tapados unos con otros.
- Una **secuencia** = fotogramas consecutivos del mismo lance (la misma pesca).
- **Papel en el proyecto:** aquí se toma la decisión ResNet vs ViT.

### Tres niveles distintos en FishNet (fuente frecuente de confusión)
```
1 secuencia  →  ~2,5 fotos  →  ~4 recortes por foto
```
Un **recorte** es un pez o una persona marcada con un recuadro. Una foto puede tener varios.

---

## 3. Espacio de etiquetas

El modelo de FishNet predice **20 clases**. Ese 20 sale de: las **19 etiquetas** que tienen al
menos 100 ejemplos de entrenamiento, más **`Other_rare`**, que agrupa las 14 que no llegan.

> **Cuidado:** esas 19 NO son 19 especies. Entre ellas están `Human`, `No fish` y `Unknown`.
> Son 19 *etiquetas*. Algunas láminas del proyecto dicen "19 especies" y eso es incorrecto.

**La métrica principal (macro-F1) se calcula sobre 15 clases**, porque se excluyen **cinco**
de las 20 (definido en `src/data.py`, constante `NON_SPECIES`):

| Excluida | Por qué |
|---|---|
| `Human` | No es un pez |
| `No fish` | No es un pez |
| `Unknown` | Es un pez, pero el anotador no resolvió la especie |
| `Tuna` | Es un pez, pero "atún" a secas tampoco resuelve la especie |
| `Other_rare` | Es un agregado de 14 especies distintas, no una especie |

20 − 5 = **15 clases de especie en el macro-F1**. Las cinco excluidas sí se aprenden, porque
hay que saber distinguirlas, pero no entran en la nota.

**Se reportan con su número individual 13 clases**: las que tienen 30 o más ejemplos de prueba
limpios. Dos de ellas (`Tuna` y `Other_rare`) se muestran **marcadas con asterisco**, porque
aparecen en la tabla pero no cuentan para el macro-F1.

**Resumen de los cuatro números:** 20 predichas · 19 + 1 es de dónde sale ese 20 · 15 cuentan
para el macro-F1 · 13 se reportan por separado.

**Caso especial — atunes:** la etiqueta `Tuna` a secas es ambigua (el anotador no resolvió
cuál era). Si el modelo dice `Yellowfin tuna` sobre un `Tuna`, se cuenta como acierto: el
error está en la etiqueta, no en el modelo. A esto se le llama *crédito jerárquico*.

---

## 4. Cómo se reparten los datos (train / val / test)

### Los tres conjuntos y para qué sirve cada uno

| Conjunto | Para qué | ¿El modelo aprende de él? |
|---|---|---|
| **Entrena (train)** | El modelo ajusta sus pesos | Sí |
| **Valida (val)** | Elegir en qué época quedarse | No aprende, pero decidimos mirándolo |
| **Prueba (test)** | Se toca una sola vez, al final | No, y nunca se usó para decidir |

Separar val de test evita "estudiar con el examen final".

### Kaggle — se reparte por **foto original**, no por imagen

| | Imágenes | Fotos realmente distintas |
|---|---|---|
| Entrena | 6.328 | **287** |
| Valida | 1.336 | **62** |
| Prueba | 1.336 | **62** |

Las ~21 copias de una misma foto van **todas juntas** al mismo lado. Si se repartieran al azar,
el modelo vería en la prueba fotos que ya memorizó y el resultado saldría inflado.

### FishNet — se respeta el split **oficial** del dataset, y luego se purga

| | Recortes | Imágenes | Secuencias |
|---|---|---|---|
| Entrena | 72.885 | 14.413 | 3.489 |
| Valida | 12.927 | 2.724 | 1.040 |
| Prueba | 13.207 | 2.872 | 1.160 |
| **Prueba PURGADA (la que se reporta)** | **7.159** | **1.750** | **612** |

Se respeta el split oficial para poder compararse con otros trabajos, pero ese split tiene una
fuga (ver riesgo 2), así que se reporta siempre el **test limpio de 7.159 recortes**.

**Sí se entrena con FishNet.** Son dos experimentos completos y separados:
- Entrenar en Kaggle → evaluar en Kaggle.
- Entrenar en FishNet → evaluar en FishNet.

Nunca se mezclan los dos conjuntos en un mismo entrenamiento.

---

## 5. Los cuatro problemas encontrados en los datos

Esta es la parte más valiosa del proyecto y la que más puntos da en la rúbrica.

### Riesgo 1 — El atajo del fondo (Kaggle)
Cada especie se fotografió en **su propia sesión**, sobre su propia mesa y su propia luz. Así
que el fondo lleva información de la especie.

**Experimento:** se borró el pez con la máscara de segmentación, dejando solo el fondo, y se
reentrenó desde cero.

| | ResNet | ViT |
|---|---|---|
| Foto completa | 0,997 | 1,000 |
| **Solo el fondo (sin pez)** | **0,971** | **0,970** |
| Solo el pez (sin fondo) | 0,999 | 0,999 |

Una red entrenada **solo con fondos** saca 0,971. El modelo aprende la mesa, no el pescado.

**Importante:** este problema **NO se arregla con el split**, porque el fondo identifica a la
*especie entera*, no a una foto concreta. Todas las fotos de Sea Bass comparten mesa, incluidas
las de prueba.

**Mitigación:** Kaggle no decide la comparación entre arquitecturas; eso se hace en FishNet,
donde el fondo cambia en cada foto.

### Riesgo 2 — Fuga en el reparto oficial de FishNet
El split que publica FishNet reparte **foto por foto**, sin mirar a qué secuencia pertenece
cada una. Resultado: el fotograma 5 de un lance queda en entrenamiento y el fotograma 7 del
**mismo lance** queda en prueba. Son casi la misma imagen.

- **45,8 %** de las cajas de prueba oficiales (39,1 % de las imágenes) comparte secuencia con el entrenamiento, en nuestro subconjunto. El 44,5 % que se dijo en el Avance era del archivo oficial completo.
- Correlación de píxeles entre dos fotogramas del mismo lance: **0,87**. Entre dos fotos
  cualquiera: **0,03**.
- Usar ese reparto sin corregir **infla el resultado ~80 %** (cifras sin ajuste de logits; con el ajuste de los resultados finales, entre 59 % y 77 % según el modelo):

| | Test oficial (con fuga) | Test limpio |
|---|---|---|
| ResNet | 0,6018 | **0,3331** |
| ViT | 0,6412 | **0,3704** |

> Nota: 0,3331 y 0,3704 son los valores sin ajuste posterior; los resultados finales
> reportados (0,352 y 0,393) incluyen el ajuste de logits y el crédito jerárquico.

**Este error no lo cometió el grupo: viene en el dataset publicado.** El grupo lo detectó y lo
corrigió.

**Mitigación:** se saca de la prueba toda imagen cuya secuencia aparezca también en
entrenamiento (quedan 7.159 de 13.207), y se reportan siempre los dos números.

### Riesgo 3 — Márgenes de error que engañan
El **margen de error** responde: *si hubiéramos usado otras fotos de prueba, ¿cuánto habría
cambiado el resultado?*

Contar 1.336 imágenes de Kaggle supone 1.336 pruebas independientes. En realidad hay **62**,
porque 20 rotaciones de una foto son una sola prueba.

| Cómo se cuenta | Margen sobre un resultado de 0,95 |
|---|---|
| Contando las 1.336 imágenes | ± 1,2 puntos porcentuales |
| **Contando las 62 fotos reales** | **± 5,4 puntos porcentuales** |

El margen falso es **4,6 veces más angosto** (√(1.336/62) ≈ 4,6).

**Analogía útil:** preguntarle veinte veces lo mismo a una persona no es encuestar a veinte
personas.

**Mitigación:** todos los intervalos se calculan con *bootstrap de clúster*: se sortean grupos
completos (la foto original en Kaggle, la secuencia en FishNet), nunca imágenes sueltas.

### Riesgo 4 — Desbalance brutal
`Human` es el **53,7 %** del entrenamiento de FishNet, y el **68 %** de la prueba limpia
(4.901 de 7.159 recortes). Entre la clase más frecuente y la más rara hay una razón de
**369 a 1**.

Por eso la métrica principal es macro-F1 (que pesa igual a todas las especies) y la accuracy
se reporta explícitamente etiquetada como poco informativa.

---

## 6. El protocolo de comparación justa

Para que la comparación no esté trucada, ambos modelos recibieron **exactamente el mismo trato**:

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| Pasos de entrenamiento (FishNet) | 9.104 | 9.104 |
| Pasos (Kaggle) | 2.940 | 2.940 |
| Tamaño de lote efectivo | 64 | 64 (32 × 2 acumulación) |
| Épocas | 8 | 8 |
| Semillas | 0, 1, 2 | 0, 1, 2 |
| Pesos de partida (ImageNet) | IMAGENET1K_V2 → 80,86 | IMAGENET1K_V1 → 81,07 |
| Label smoothing | 0,1 | 0,1 |

**Qué es un "paso":** una actualización de los pesos usando un lote de 64 imágenes.
9.104 pasos × 64 ≈ **583.000 imágenes vistas**, unas 8 vueltas al conjunto.

**Por qué importan los pesos de partida:** la trampa más común en este tipo de comparación es
enfrentar un ViT moderno contra la versión antigua del ResNet, que arranca ~5 puntos más abajo
en ImageNet. Aquí se usaron versiones de calidad pareja (80,86 vs 81,07), así que la diferencia
no viene del preentrenamiento.

**Mejor época alcanzada** (de 8, contando desde 1): ResNet en las épocas 2, 5 y 3 según la
semilla (promedio 3,3). ViT en las épocas 8, 6 y 2 (promedio 5,3). En **una sola** de las tres
corridas el ViT seguía mejorando al agotarse el presupuesto.

---

## 7. Resultados

### Resultado final: los cinco modelos (FishNet, prueba limpia, 3 semillas cada uno)

| Modelo | Tipo | macro-F1 | Margen (IC 95 %) |
|---|---|---|---|
| ResNet-50 | CNN, 2015 | 0,352 | 0,306 – 0,385 |
| ViT-B/16 | Transformer, 2020 | 0,393 | 0,349 – 0,424 |
| DeiT-S | Transformer pequeño | 0,354 | 0,311 – 0,383 |
| **ConvNeXt-T** | CNN moderna, 2022 | **0,423** | 0,372 – 0,457 |
| **Swin-T** | Transformer moderno, 2021 | **0,424** | 0,369 – 0,457 |

| Comparación (pareada) | Diferencia | Margen | ¿Se distingue? |
|---|---|---|---|
| ViT-B/16 − ResNet-50 | +0,042 | +0,007 a +0,077 | Sí |
| DeiT-S − ResNet-50 (mismo tamaño) | +0,003 | −0,036 a +0,040 | No |
| Swin-T − ConvNeXt-T (modernos) | −0,001 | −0,029 a +0,027 | No |
| ConvNeXt-T − ResNet-50 (CNN moderna vs antigua) | **+0,069** | +0,035 a +0,105 | **Sí** |
| Swin-T − ViT-B/16 (transformer moderno vs antiguo) | +0,026 | −0,003 a +0,055 | No |

**Cómo se lee:**
1. A igual tamaño (DeiT-S) o a igual época (ConvNeXt-T contra Swin-T), CNN y transformer **no se
   distinguen**. La ventaja del ViT-B/16 no se mantiene en un transformer del tamaño del ResNet.
2. **Lo que sí se distingue es modernizar:** ConvNeXt-T supera al ResNet-50 por +0,069.
3. **Advertencia honesta:** el ResNet-50 se entrenó con su propia receta (otra tasa de aprendizaje)
   y ConvNeXt-T con la de su paper. Ese +0,069 mide **modelo y receta juntos**.
4. ¿Cuál es el mejor? **ConvNeXt-T y Swin-T empatan arriba** (0,423 y 0,424). No se puede elegir
   uno entre esos dos con estos datos.

**Barcos que el modelo no vio** (el 69 % de la prueba viene de cámaras que no estaban en
entrenamiento): todos caen. ResNet-50 0,253, ViT-B/16 0,280, DeiT-S 0,245, ConvNeXt-T 0,336,
Swin-T 0,338. Ahí los modernos también superan a su versión antigua (ConvNeXt-T − ResNet-50
+0,080; Swin-T − ViT-B/16 +0,059, las dos se distinguen). Es la limitación más importante: en la
caleta, todo barco sería nuevo.

**Atunes agrupados** (los cuatro atunes como una sola clase): la ventaja del ViT-B/16 sobre el
ResNet-50 baja a +0,031 y deja de distinguirse. Parte de su ventaja era separar un atún de otro.

### Tabla del par original (Avance)

| | ResNet-50 | ViT-B/16 |
|---|---|---|
| **Kaggle — macro-F1** | 0,997 | **1,000** |
| **FishNet limpia — macro-F1** | 0,352 | **0,393** |
| Margen (IC 95 %) | 0,306 – 0,385 | 0,349 – 0,424 |
| FishNet — aciertos totales | 85,6 % | 88,2 % |

**Cuidado con el 88,2 %:** está inflado porque el 68 % de la prueba son personas, y un modelo
que respondiera «persona» a todo ya sacaría **68,5 %**.

### El par original en detalle (lo central del Avance)

| Dato | Valor | Qué significa |
|---|---|---|
| Ventaja del ViT en FishNet | **+0,042** | Diferencia de macro-F1 (0,393 − 0,352) |
| Margen de esa ventaja | **+0,007 a +0,077** | No llega al cero → la ventaja no es casualidad |
| Probabilidad de que sea > 0 | **0,990** | |
| Comparaciones entre semillas | **9 de 9** favorecen al ViT | Su peor corrida supera a la mejor del ResNet |
| Diferencia en Kaggle | **+0,003**, margen 0,000 a 0,011 | Ahí **no** hay diferencia demostrable |

La comparación es **pareada**: enfrenta modelo contra modelo sobre **las mismas 612 secuencias**,
no dos promedios sueltos. Por eso los intervalos individuales pueden solaparse y aun así la
diferencia ser real.

### Rendimiento por clase en FishNet (ViT, prueba limpia)

| Clase | F1 | Ejemplos |
|---|---|---|
| Human* | 0,975 | 4.901 |
| Mahi mahi | 0,873 | 528 |
| Tuna* | 0,881 | 351 |
| Striped marlin | 0,279 | 230 |
| Unknown* | 0,815 | 212 |
| No fish* | 0,971 | 175 |
| Swordfish | 0,475 | 141 |
| Albacore | 0,541 | 129 |
| Yellowfin tuna | 0,308 | 83 |
| Other_rare* | 0,284 | 70 |
| Wahoo | 0,367 | 69 |
| Shark | 0,581 | 57 |
| Opah | 0,899 | 46 |
| Escolar | 0,269 | 43 |
| Long snouted lancetfish | 0,323 | 41 |
| Blue marlin | 0,022 | 31 |
| Bigeye tuna | 0,000 | 22 |
| Great barracuda | 0,048 | 12 |
| Skipjack tuna | 0,937 | 11 |
| Shortbill spearfish | 0,000 | 7 |

`*` = no entra en el macro-F1 de especies. Son cinco: Human, No fish, Unknown, Tuna y
Other_rare. Las otras 15 son las que promedia el macro-F1.

**Por qué el macro-F1 parece bajo (0,393):** porque promedia dando el mismo peso a las 15
especies, y varias raras están cerca de cero (Bigeye tuna 0,000, Shortbill spearfish 0,000,
Blue marlin 0,022). Para referencia, **responder al azar entre 20 clases daría ~0,05**, así que
0,393 es casi ocho veces el azar. Comprobación: el promedio de los F1 de esas 15 clases da
0,3947, que coincide con el 0,393 reportado.

### La hipótesis del enunciado NO se sostiene

El enunciado del proyecto sugería que el ViT ganaría gracias a su atención global sobre peces
amontonados y tapados. **Se midió y es falso:** la ventaja del ViT **no crece** con el nivel de
oclusión (correlación de Spearman **−0,600**, es decir, va en dirección contraria). El ViT gana,
pero no por esa razón.

---

## 8. La auditoría (se buscaron errores en los propios resultados)

Se hicieron cuatro pruebas para intentar tumbar los resultados. Ninguna cambió la conclusión.

| Prueba | Resultado |
|---|---|
| ¿Alguna imagen de prueba está también en entrenamiento? | **No, en nada de lo reportado.** El que sí filtraba era el split oficial de FishNet: 6.048 cajas y 7 imágenes enteras, todas eliminadas por la purga |
| Comparar **píxeles**, sin creerle a ninguna etiqueta | La purga funcionó: lo eliminado tenía un casi gemelo el **38,1 %** de las veces; lo que quedó, solo el **2,6 %** |
| Quitar además ese 2,6 % restante | El resultado casi no se mueve: la ventaja del ViT queda en **+0,0415 [+0,005, +0,081]** |
| ¿Cuánto memoriza cada modelo? | Hay sobreajuste (ver abajo), pero el que más memoriza es también el que mejor generaliza |

### Sobreajuste medido (macro-F1 en train vs test)

| | Train | Test | Brecha |
|---|---|---|---|
| ResNet en FishNet | 0,783 | 0,340 | 0,443 |
| ViT en FishNet | 0,925 | 0,370 | 0,555 |
| ResNet en Kaggle | 1,000 | 0,997 | 0,003 |
| ViT en Kaggle | 1,000 | 1,000 | 0,000 |

**Cómo defenderlo:** FishNet tiene 20 clases muy desbalanceadas y el presupuesto de
entrenamiento es corto, así que memorizar era esperable. Lo relevante es que el modelo que más
memoriza (ViT) es también el que mejor generaliza. Eso señala dónde queda margen: más
regularización y más datos.

### El veredicto sobre Kaggle: no hay fuga, el examen era demasiado fácil

Un clasificador de **vecino más cercano sobre miniaturas de 32×32 píxeles en blanco y negro**
—sin entrenar nada, solo copiando la etiqueta de la imagen más parecida— acierta:

- **82,9 %** en Kaggle
- **55,9 %** en FishNet (donde las redes llegan a 82-85 % de accuracy)

O sea: en Kaggle un método con cero inteligencia ya saca 82,9, y la red llega a 99,7. La red
solo aportó ~17 puntos. En FishNet el mismo truco saca 55,9 y las redes ~83: ahí sí hay
aprendizaje real.

---

## 9. Hallazgos adicionales

### Saber decir «no sé» (detección de fuera de distribución)
Se tomaron los modelos **entrenados en FishNet** y se les mostraron fotos de **Kaggle**, cuyas
9 especies mediterráneas no están entre las 20 que conocen. La pregunta: ¿admiten que no saben,
o responden con seguridad?

Con el umbral ajustado para que ambos conserven el 95 % de los peces que sí conocen:

| | ResNet | ViT |
|---|---|---|
| Acepta desconocidos como si fueran suyos (max-softmax) | **94 %** | **49 %** |
| Lo mismo usando energía libre | 84 % | 27 % |
| AUROC (max-softmax) | 0,683 | **0,893** |
| AUROC (energía) | 0,766 | **0,926** |

Diferencia de AUROC: **+0,209 [+0,171, +0,249]** a favor del ViT. **Es la diferencia más grande
encontrada entre las dos arquitecturas.** (Medido solo para el par original; los modelos nuevos no
pasaron por esta prueba.)

**Especies chilenas** (entrega final): 45 fotos de iNaturalist de reineta, merluza y congrio. Lo
correcto es decir «no sé». Con el umbral elegido en **validación**: el ResNet rechaza el 10 % y el
ViT el 32 %; AUROC 0,639 contra 0,801. Cuando aceptan, casi siempre dicen «atún aleta amarilla».
Ninguno está listo para la caleta.

Por qué importa: en el muelle van a aparecer especies que el sistema nunca vio. Un sistema que
las etiqueta con seguridad contamina el registro sin que nadie se entere.

### Calibración
Los modelos están **subconfiados con `Human`** (−0,10 / −0,09) y **sobreconfiados con los peces**
(+0,165 / +0,194). Como los errores van en direcciones opuestas, **una sola temperatura escalar
no arregla nada**. ResNet y ViT no se distinguen en calibración cruda.

### Velocidad (CPU: i5-12400F, 6 hilos, lote 1)

| Modelo | Tiempo por pez | MACs |
|---|---|---|
| ResNet-50 | 43 ms | 4,09 G |
| ViT-B/16 | 119 ms (2,7× más lento) | 16,85 G |
| ViT-B/32 | 42 ms | 4,36 G |

En el Avance se eligió el ViT-B/16 porque la latencia no cambiaba qué era factible. **Con los
resultados finales, los mejores son ConvNeXt-T y Swin-T**, que además cuestan lo mismo que el
ResNet-50 en cómputo (~4,5 G contra 4,1 G). Su latencia en CPU no se midió.

### Interpretabilidad (dónde mira cada red — *pointing game* en Kaggle)

| Método | Acierto | |
|---|---|---|
| ResNet · Grad-CAM | **0,582** | |
| ViT · Chefer | 0,408 | |
| ViT · attention rollout | 0,204 | |
| Línea base (azar) | 0,169 | |

Mide qué fracción de la atención cae dentro de la silueta real del pez.

**Corrección de la entrega final (FishNet):** con el pez siempre al centro del recorte, un mapa
fijo en el centro que no mira la imagen ya sacaba 0,757, más que cualquier modelo. La medida
premiaba mirar el centro. Se corrigió poniendo el pez en posición al azar y restando la atención
que cae en la caja de **otro** recorte. Resultado: Grad-CAM (ResNet) +0,090, Chefer (ViT) +0,087,
rollout +0,016. **Los dos siguen al pez en una medida parecida: el ViT no mira menos el fondo.**
Los 0,582 / 0,408 de Kaggle probablemente tienen el mismo sesgo al centro.

---

## 10. Confusiones frecuentes (CORREGIR SI APARECEN)

Estas son equivocaciones reales que tuvieron integrantes del grupo. Si el estudiante cae en
alguna, corrígela de inmediato.

**«Entonces trabajamos con 20 copias idénticas por foto, eso arruina el modelo»**
→ No. Sobre cada copia se aplica **aumento aleatorio en cada pasada** (recorte aleatorio entre
70 % y 100 %, espejado, cambio de brillo/contraste/color), así que el modelo nunca ve dos veces
los mismos píxeles. La duplicación no empeora el resultado: hace que **la muestra sea pequeña**
(411 fotos) y que el examen parezca más preciso de lo que es.

**«El problema del fondo se arregla repartiendo bien los datos»**
→ No. El fondo identifica a **la especie entera**, no a una foto. Todas las fotos de una especie
comparten mesa, incluidas las de prueba. Son dos problemas independientes.

**«El riesgo 2 es un error nuestro»**
→ No. La fuga viene **en el reparto que publica FishNet**. El grupo la detectó y la corrigió.

**«0,349 a 0,424 es la escala de la métrica»**
→ No. Macro-F1 siempre va de 0 a 1. Ese rango es el **margen de error** alrededor del 0,393
medido: cuánto se habría movido el resultado con otras secuencias de prueba.

**«No entrenamos con FishNet»**
→ Falso, y es un error grave. Se entrena con los dos conjuntos, por separado. FishNet es el
**principal**: 72.885 recortes de entrenamiento, y es donde se toma la decisión.

**«Valida es lo mismo que prueba»**
→ No. Son tres conjuntos. Valida sirve para elegir la época; prueba se toca una sola vez al final.

**«El modelo encuentra los peces en la foto»**
→ No. No hay detector. Alguien tiene que marcar el recuadro primero.

**«El 88,2 % de aciertos es el resultado principal»**
→ No. Está inflado por el 68 % de personas. El resultado principal es el macro-F1 (0,393).

**«Los intervalos se solapan, entonces empatan»**
→ No. La comparación es **pareada** sobre las mismas 612 secuencias. Dos intervalos individuales
pueden solaparse y la diferencia pareada ser distinta de cero. Eso es exactamente lo que pasa acá.

**«El macro-F1 promedia sobre las 20 clases» o «sobre 19 especies»**
→ No. Promedia sobre **15**. Se excluyen cinco: Human y No fish (no son peces), Unknown y Tuna
(son peces sin especie resuelta) y Other_rare (es un agregado). El "19" que aparece en algunas
láminas cuenta *etiquetas con ≥100 ejemplos*, no especies, e incluye Human, No fish y Unknown.

**«Entonces el transformer es mejor que la CNN»**
→ No. Solo el ViT-B/16 le gana al ResNet-50, y ese par no es parejo en tamaño ni en época. A igual
tamaño (DeiT-S) o a igual época (ConvNeXt-T contra Swin-T), no se distinguen.

**«No se distingue significa que son iguales»**
→ No. Significa que con esta prueba no se puede decir cuál es mejor. El margen incluye el cero.

**«ConvNeXt-T le gana al ResNet-50 solo por la arquitectura»**
→ No se puede afirmar. También cambió la receta de entrenamiento. Mide modelo y receta juntos.

**«El tercer modelo era ViT-B/32»**
→ Estaba planeado en el Avance, pero sus pesos tienen 75,9 % en ImageNet contra 80,9 % del
ResNet-50: compararlos rompía la regla de pesos parejos. Se reemplazó por DeiT-S (79,9 %) y por el
par moderno.

**«pp es lo mismo que %»
→ No. **pp = puntos porcentuales.** ± 5,4 pp sobre un 95 % significa bajar hasta ~90 %. No es
el 5,4 % *de* 95.

---

## 11. Glosario en lenguaje simple

| Término | Qué significa |
|---|---|
| **macro-F1** | Se calcula qué tan bien le va al modelo en **cada especie por separado** y se promedian esos valores dándole **el mismo peso a todas**. Va de 0 a 1. Es más exigente que la accuracy porque una especie con 7 ejemplos pesa igual que una con 4.901 |
| **F1 de una clase** | Combina dos cosas: cuando el modelo dijo "X", ¿acertó? y de los X reales, ¿cuántos encontró? |
| **Accuracy / aciertos totales** | Porcentaje de recortes bien clasificados. Engañosa cuando hay desbalance |
| **Margen de error / IC 95 %** | Cuánto se movería el resultado si se repitiera la medición con otras fotos de prueba |
| **pp** | Puntos porcentuales |
| **Bootstrap de clúster** | Forma de calcular el margen sorteando **grupos completos** (fotos o secuencias) en vez de imágenes sueltas |
| **Comparación pareada** | Enfrentar los dos modelos sobre exactamente los mismos datos, en vez de comparar dos promedios sueltos |
| **Secuencia (FishNet)** | Fotogramas consecutivos del mismo lance de pesca |
| **Recorte / caja / bounding box** | El rectángulo que marca un pez o una persona dentro de una foto |
| **Fuga de datos (data leakage)** | Que algo del examen final se haya colado en el material de estudio |
| **Purga** | Sacar de la prueba todo lo que comparte secuencia con el entrenamiento |
| **Sobreajuste (overfitting)** | Que el modelo memorice el entrenamiento en vez de aprender a generalizar |
| **OOD (fuera de distribución)** | Datos de un tipo que el modelo nunca vio al entrenar |
| **AUROC** | Qué tan bien el modelo separa lo conocido de lo desconocido. 0,5 = azar, 1,0 = perfecto |
| **Semilla (seed)** | Número que fija el azar del entrenamiento. Se entrena 3 veces con semillas distintas para ver cuánto varía el resultado por pura suerte |
| **Paso (step)** | Una actualización de los pesos con un lote de imágenes |
| **Época (epoch)** | Una vuelta completa a todo el conjunto de entrenamiento |
| **Label smoothing** | Truco que evita que el modelo se ponga demasiado seguro de sí mismo |
| **Ajuste de logits (τ)** | Corrección posterior que compensa el desbalance de clases. Vale 0,18 en el ResNet y 0,05 en el ViT |
| **Pointing game** | Medir qué fracción de la "atención" del modelo cae sobre el pez de verdad |

---

## 12. Estado del proyecto

**Hecho:**
- Los dos baselines entrenados, 3 semillas cada uno, en los dos conjuntos (12 corridas).
- Los cuatro problemas de los datos, medidos y mitigados.
- Auditoría de fuga y sobreajuste (4 pruebas).
- Calibración y detección de «no sé».
- Velocidad vs precisión.
- Interpretabilidad (pointing game).
- **Demo funcional** que corre las dos arquitecturas en vivo (`python -m src.demo` →
  http://127.0.0.1:8000). Solo biblioteca estándar, sin gradio ni streamlit.
- Informe técnico y data card.

- **Entrega final:** DeiT-S, ConvNeXt-T y Swin-T entrenados (9 corridas más); análisis por
  especie, atunes agrupados y cámaras nuevas; interpretabilidad en FishNet corregida; prueba con
  45 fotos chilenas; mazo final (22 láminas) e informe de 6 páginas.

**Falta (próximos pasos, no se hará para la entrega):**
- Un detector para encontrar y contar los peces.
- Fotos locales de la Caleta Portales.
- Entrenar y evaluar con un reparto por cámara (barcos nuevos).

---

## 13. Mapa del repositorio

```
src/
  data.py         Carga de datos y aumento (recorte aleatorio, espejado, color)
  train.py        Entrenamiento, reanudable
  evaluate.py     Métricas, bootstrap de clúster, comparación pareada
  interpret.py    Grad-CAM, Chefer, attention rollout, pointing game
  calibrate.py    Temperatura, ECE/Brier/NLL y detección OOD
  speed.py        Latencia, MACs, decisión de arquitectura
  demo.py         Demo web local
scripts/
  build_splits.py Genera los CSV de particiones (idempotente)
  auditoria.py    Las 4 pruebas de la auditoría
  entrenar_pares.py  Entrena DeiT-S, ConvNeXt-T y Swin-T; se puede parar y retomar
  analisis_final.py  Tablas de la entrega final (pares, especies, atunes, cámaras)
  interpret_fishnet.py, chile_ood.py  Interpretabilidad en FishNet y especies chilenas
  build_pptx_final.py, build_informe.py  Mazo e informe, con las cifras leídas de runs/
data/splits/      Los CSV del contrato experimental (versionados)
runs/             Resultados en JSON + checkpoints + logits guardados
docs/             Informe, data card, plan y presentación
tests/            Validación de las estadísticas contra sklearn
```

**Archivos de resultados con los números crudos:**
`runs/comparacion_e4_e5_limpio.json`, `runs/comparacion_e1_e2.json`,
`runs/comparacion_e7.json`, `runs/auditoria.json`, `runs/calibracion_ood.json`,
`runs/velocidad.json`, `runs/pointing_game.json`, y para la entrega final
`runs/analisis_final.json`, `runs/pointing_game_fishnet.json` y `runs/chile_ood.json`.

---

## 14. Las preguntas que más probablemente hará el profesor

**1. ¿Por qué el macro-F1 es tan bajo en FishNet (0,393)?**
Son 20 clases, varias casi idénticas entre sí (cuatro tipos de atún), muy desbalanceadas, y el
promedio parejo castiga fuerte los errores en las raras. Al azar sería 0,05.

**2. ¿No será que el ViT gana solo por tener más parámetros?**
Eso fue lo que se midió con DeiT-S, un transformer del tamaño del ResNet: no se distingue de él
(+0,003, margen −0,036 a +0,040). Y entre dos modelos modernos del mismo tamaño, ConvNeXt-T y
Swin-T, tampoco hay diferencia. Conclusión: la ventaja del ViT-B/16 se explica mejor por su tamaño
y su modernidad que por la atención.

**2b. Entonces, ¿qué modelo recomiendan?**
ConvNeXt-T o Swin-T: empatan arriba (0,423 y 0,424), resisten mejor los barcos nuevos y cuestan
en cómputo lo mismo que el ResNet-50. No se puede elegir uno de los dos con estos datos.

**3. ¿Esto sirve en Chile?**
Hoy no. Ninguna especie chilena está en los datos públicos, y habría que recolectar fotos en
la caleta primero. Lo que se demuestra es el método.

**4. ¿Cómo sabemos que no hay fuga de datos con esos resultados?**
Se hicieron cuatro pruebas distintas, incluida una que compara píxeles sin creerle a ninguna
etiqueta. Ninguna cambió la conclusión. Además, el 99,7 % de Kaggle se explica sin fuga: un
método sin aprendizaje ya saca 82,9 % ahí.
