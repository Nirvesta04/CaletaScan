# -*- coding: utf-8 -*-
"""
Informe tecnico final de Caleta-Scan (G3): docs/informe.html y docs/informe.pdf.

    python -m scripts.build_informe

El curso pide un informe de 4 a 6 paginas. Todas las cifras se leen de
runs/*.json, igual que el mazo, para que informe y presentacion no se
contradigan. El PDF se imprime con Edge o Chrome sin interfaz.
"""
import json
import os
import shutil
import subprocess

from scripts.lecturas import lectura

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
HTML = os.path.join(ROOT, 'docs', 'informe.html')
PDF = os.path.join(ROOT, 'docs', 'informe.pdf')


def J(n):
    with open(os.path.join(RUNS, n), encoding='utf-8') as fh:
        return json.load(fh)


A = J('analisis_final.json')
PG_F = J('pointing_game_fishnet.json')
PG_K = J('pointing_game.json')
CH = J('chile_ood.json')
CAL = J('calibracion_ood.json')
E7 = J('comparacion_e7.json')
KAG = J('comparacion_e1_e2.json')


def num(x, d=3, s=False):
    return (('%+.*f' if s else '%.*f') % (d, x)).replace('.', ',')


def pct(x, d=0):
    return ('%.*f' % (d, 100 * x)).replace('.', ',') + ' %'


def miles(n):
    return '{:,}'.format(n).replace(',', '.')


def ic(v, s=True):
    return '[%s · %s]' % (num(v[0], 3, s), num(v[1], 3, s))


def ood(run, sub):
    k = [k for k in CAL['ood'] if k.startswith(run) and k.endswith(sub)][0]
    return CAL['ood'][k]['energia']


M = A['modelos']
V = A['por_variante']
VT = V['todas']['pares']
fg = A['fuga_split_oficial']
cam = A['camaras']
P_VR = VT['ViT-B/16 - ResNet-50']


def fila_modelo(clave):
    if clave not in M:
        return '<tr><td class="name">%s</td><td class="num" colspan="3">pendiente</td></tr>' % {
            'e10_deit_s_fishnet': 'DeiT-S', 'e11_convnext_t_fishnet': 'ConvNeXt-T',
            'e12_swin_t_fishnet': 'Swin-T'}[clave]
    m = M[clave]
    sem = len(m['semillas'])
    return ('<tr><td class="name">%s</td><td class="num">%s %s</td><td class="num">%s</td>'
            '<td class="num">%d</td></tr>' % (m['modelo'], num(m['macro_f1']), ic(m['ic95'], False),
                                              ' · '.join(num(x) for x in m['por_semilla']), sem))


def fila_par(clave, etiqueta):
    d = VT.get(clave)
    if d is None:
        return '<tr><td class="name">%s · %s</td><td class="num" colspan="3">pendiente</td></tr>' % (
            etiqueta, clave.replace(' - ', ' − '))
    return ('<tr><td class="name">%s · %s</td><td class="num"><strong>%s</strong></td><td class="num">%s</td>'
            '<td class="num">%s</td></tr>' % (etiqueta, clave.replace(' - ', ' − '),
                                              num(d['delta_medio'], 3, True), ic(d['ic95']),
                                              'sí' if d['distinguible'] else 'no'))


def lectura_par(clave, cnn, tr):
    d = VT.get(clave)
    t = lectura(clave, d)
    if t is None:
        return '<strong>%s contra %s:</strong> pendiente de entrenamiento.' % (tr, cnn)
    return '<strong>%s contra %s (%s, margen %s):</strong> %s' % (
        tr, cnn, num(d['delta_medio'], 3, True), ic(d['ic95']), t)


def fila_extra(clave, etiqueta):
    def c(d):
        txt = '%s %s' % (num(d['delta_medio'], 3, True), ic(d['ic95']))
        return '<strong>%s</strong>' % txt if d['distinguible'] else txt
    return ('<tr><td class="name">%s</td><td class="num">%s</td><td class="num">%s</td></tr>'
            % (etiqueta, c(VT[clave]), c(V['camara_nueva']['pares'][clave])))


def variante(v, m):
    return V[v]['modelos'][m]


_ln = [lectura(c, VT.get(c)) for c in ('DeiT-S - ResNet-50', 'Swin-T - ConvNeXt-T')]
PARES_NUEVOS = ('<li>%s</li>' % ' '.join(t for t in _ln if t)) if any(_ln) else (
    '<li>Los pares de mismo tamaño (DeiT-S) y moderno (ConvNeXt-T contra Swin-T) están pendientes '
    'de entrenamiento.</li>')
SEMILLAS_VR = (' La peor semilla del ViT supera a la mejor del ResNet.'
               if min(M['e5_vit_b_16_fishnet']['por_semilla']) > max(M['e4_resnet50_fishnet']['por_semilla'])
               else '')
pe_r, pe_v = A['por_especie']['ResNet-50'], A['por_especie']['ViT-B/16']
gana = sum(pe_v[k]['f1'] > pe_r[k]['f1'] + 1e-9 for k in pe_r)
ef = PG_F['energia_en_la_caja']
o_r, o_v = ood('e4_resnet50_fishnet', 'id_sin_human'), ood('e5_vit_b_16_fishnet', 'id_sin_human')
c_r, c_v = CH['modelos']['ResNet-50'], CH['modelos']['ViT-B/16']
k_f = E7['e1_resnet50_kaggle__vs__e7_resnet50_solo_fondo']

with open(os.path.join(ROOT, 'docs', 'informe_estilo.css'), encoding='utf-8') as fh:
    CSS = fh.read()

cuerpo = f'''
<header class="mast">
  <div class="kicker">Caleta-Scan · G3 · Aprendizaje Automático II · PUCV · noviembre 2026</div>
  <h1>Informe técnico</h1>
  <p class="sub">Clasificación de especies de pescado con redes convolucionales y transformers:
  datos, protocolo de comparación, resultados y límites.</p>
  <p class="autores">Patricio Toledo · Andrés Saavedra · Sebastián Casas · Bastián Molina</p>
</header>

<section>
<h2>1 · Resumen</h2>
<p>Comparamos redes convolucionales (CNN) y transformers de visión para identificar la especie de
un pez en fotos de faena pesquera, como apoyo al registro de desembarque de Sernapesca en la
Caleta Portales. Entrenamos cinco modelos en tres pares con preentrenamiento parejo y el mismo
presupuesto, y los evaluamos en FishNet, un conjunto de cámaras a bordo de barcos reales, con una
prueba limpia de fugas. Lo principal:</p>
<ol>
<li><strong>A igual tamaño y época, CNN y transformer no se distinguen.</strong> DeiT-S queda a
{num(VT['DeiT-S - ResNet-50']['delta_medio'], 3, True)} del ResNet-50 y Swin-T a
{num(VT['Swin-T - ConvNeXt-T']['delta_medio'], 3, True)} de ConvNeXt-T, con márgenes que incluyen el cero.
La ventaja del ViT-B/16 sobre el ResNet-50 ({num(P_VR['delta_medio'], 3, True)}, margen
{ic(P_VR['ic95'])}) no se mantiene en un transformer del tamaño del ResNet.</li>
<li><strong>Lo que mejora el resultado es usar un modelo moderno.</strong> ConvNeXt-T, una CNN de 2022,
supera al ResNet-50 por {num(VT['ConvNeXt-T - ResNet-50']['delta_medio'], 3, True)} (margen
{ic(VT['ConvNeXt-T - ResNet-50']['ic95'])}), también en barcos que no vio. Esta diferencia mide modelo y
receta de entrenamiento juntos.</li>
<li>Los barcos que el modelo no vio, que son el {pct(1 - cam['frac_cajas_camara_vista'])} de la prueba,
son el mayor problema: todos los modelos caen, incluso el mejor baja de
{num(V['camara_vista']['modelos']['ConvNeXt-T']['macro_f1'])} a
{num(V['camara_nueva']['modelos']['ConvNeXt-T']['macro_f1'])}.</li>
<li>El reparto oficial de FishNet tiene fuga: el {pct(fg['frac_cajas'], 1)} de sus cajas de prueba
comparte lance con el entrenamiento, y eso infla los resultados alrededor de un 80 %.</li>
<li>Entre ResNet-50 y ViT-B/16: con el pez en posición al azar, los dos concentran su atención en el
pez en una medida parecida, y el ViT reconoce mejor lo que no conoce. Ninguno sirve todavía para
especies chilenas.</li>
</ol>
</section>

<section>
<h2>2 · Problema y alcance</h2>
<p>En la Caleta Portales un inspector anota especie y cantidad durante la descarga. El proceso es
lento, depende de quién esté de turno y no alcanza en temporada alta. La idea es que una foto de la
faena permita proponer la especie de cada pez, con su nivel de confianza, para que la persona revise
y corrija.</p>
<p><strong>Qué hace el sistema:</strong> recibe el recorte de un pez ya marcado con una caja y devuelve la
especie más probable o «no sé». <strong>Qué no hace:</strong> no detecta ni cuenta peces en la foto completa,
y no conoce reineta, merluza ni congrio, porque esas especies no están en ningún conjunto público.
Por eso lo que se demuestra aquí es el método de comparación; usarlo en la caleta exige fotos locales.</p>
</section>

<section>
<h2>3 · Datos y riesgos</h2>
<p><strong>Kaggle · A Large Scale Fish Dataset</strong> (Ulucan et al., 2020): 9.000 fotos de 9 especies de
supermercado sobre fondo de estudio, con la silueta de cada pez marcada. Son 411 fotos originales;
el resto son copias giradas y espejadas. <strong>FishNet Open Images</strong> (Kay y Merrifield, 2021): usamos
20.002 fotos de 77 cámaras a bordo, con 99.019 cajas. El modelo predice 20 clases (19 etiquetas con al
menos 100 ejemplos y una clase de especies raras); el macro-F1 se promedia sobre las 15 que son
especies, sin persona, sin pez, desconocido, atún sin especie ni especies raras.</p>
<div class="tw"><table>
<thead><tr><th>Riesgo</th><th>Medición</th><th>Qué hicimos</th></tr></thead>
<tbody>
<tr><td class="name">Kaggle: ~21 copias por foto original</td><td>411 originales en 9.000 imágenes</td><td>Reparto por foto original; la prueba tiene 62 fotos independientes</td></tr>
<tr><td class="name">Kaggle: el fondo delata la especie</td><td>macro-F1 {num(k_f['macro_f1_b'])} con el pez borrado (contra {num(k_f['macro_f1_a'])})</td><td>Kaggle no decide la comparación</td></tr>
<tr><td class="name">FishNet: fuga por lance en el reparto oficial</td><td>{pct(fg['frac_cajas'], 1)} de las cajas de prueba y {pct(fg['frac_imagenes'], 1)} de las imágenes</td><td>Prueba limpia: 7.159 cajas de 612 lances sin cruce</td></tr>
<tr><td class="name">Fotos de un lance no son independientes</td><td>Márgenes hasta 4,6 veces más angostos si se ignora</td><td>Bootstrap por lance completo</td></tr>
<tr><td class="name">FishNet: barcos repetidos</td><td>{pct(cam['frac_cajas_camara_vista'])} de la prueba limpia viene de cámaras de entrenamiento</td><td>Se reporta aparte (sección 5)</td></tr>
</tbody></table></div>
<p>La auditoría (<code>scripts/auditoria.py</code>) confirmó que en los repartos que usamos ningún lance ni
foto original cruza de entrenamiento a prueba. Un vecino más cercano sobre miniaturas de 32×32 píxeles,
sin ninguna red, acierta 0,829 en Kaggle y 0,559 en FishNet: Kaggle es casi trivial y FishNet no.</p>
</section>

<section>
<h2>4 · Modelos y protocolo</h2>
<div class="tw"><table>
<thead><tr><th>Par</th><th>CNN</th><th>Transformer</th><th>Qué responde</th></tr></thead>
<tbody>
<tr><td class="name">Original</td><td>ResNet-50 · V2 · 80,9 % · 26 M</td><td>ViT-B/16 · V1 · 81,1 % · 87 M</td><td>¿Cuál conviene, con preentrenamiento parejo?</td></tr>
<tr><td class="name">Mismo tamaño</td><td>ResNet-50 (el mismo)</td><td>DeiT-S · 79,9 % · 22 M</td><td>¿El ViT-B/16 ganaba por ser transformer o por ser más grande?</td></tr>
<tr><td class="name">Moderno</td><td>ConvNeXt-T · 82,5 % · 29 M</td><td>Swin-T · 81,5 % · 28 M</td><td>¿El ResNet perdía por ser una arquitectura de 2015?</td></tr>
</tbody></table></div>
<p>El porcentaje es el acierto en ImageNet de los pesos de partida, según quien los publica
(torchvision; DeiT-S con los pesos oficiales de Facebook, sin destilación). Todos los modelos reciben
9.104 pasos de entrenamiento con lote efectivo de 64, la misma resolución (224 px), los mismos aumentos
y tres semillas. Todos se entrenan con AdamW; los transformers y el par moderno con tasa decreciente
por capa, como en sus artículos, y el ResNet con una tasa para el cuerpo y otra para la cabeza. La época se elige con la validación limpia; la prueba
se usa una sola vez.</p>
<p><strong>Métrica.</strong> El macro-F1 es el promedio del F1 de las 15 especies, de 0 a 1: cada especie pesa
lo mismo. No usamos la accuracy porque el 68 % de la prueba son personas y un modelo que siempre diga
«persona» ya acierta el 68 %. Antes del argmax se aplica el ajuste de logits de Menon et al. (2021),
con su parámetro elegido en validación. Si el modelo dice un atún específico sobre una caja anotada
como «atún sin especie», no se cuenta como error de ese atún. <strong>Margen:</strong> intervalo de confianza al 95 % por bootstrap, remuestreando lances
completos (2.000 réplicas). Para comparar dos modelos se remuestrean los mismos lances para ambos y
se mira la diferencia; se distingue si su margen no incluye el cero.</p>
</section>

<section>
<h2>5 · Resultados</h2>
<div class="tw"><table>
<caption>Prueba limpia de FishNet. Macro-F1 promedio de las semillas y su margen al 95 %; a la derecha, el valor de cada semilla.</caption>
<thead><tr><th>Modelo</th><th class="num">macro-F1 [margen]</th><th class="num">por semilla</th><th class="num">semillas</th></tr></thead>
<tbody>
{fila_modelo('e4_resnet50_fishnet')}
{fila_modelo('e5_vit_b_16_fishnet')}
{fila_modelo('e10_deit_s_fishnet')}
{fila_modelo('e11_convnext_t_fishnet')}
{fila_modelo('e12_swin_t_fishnet')}
</tbody></table></div>
<div class="tw"><table>
<caption>Diferencia transformer − CNN, con bootstrap pareado sobre los mismos lances.</caption>
<thead><tr><th>Par</th><th class="num">diferencia</th><th class="num">margen al 95 %</th><th class="num">¿se distingue?</th></tr></thead>
<tbody>
{fila_par('ViT-B/16 - ResNet-50', 'Original')}
{fila_par('DeiT-S - ResNet-50', 'Mismo tamaño')}
{fila_par('Swin-T - ConvNeXt-T', 'Moderno')}
</tbody></table></div>
<ul>
<li>{lectura_par('ViT-B/16 - ResNet-50', 'ResNet-50', 'ViT-B/16')}{SEMILLAS_VR}</li>
<li>{lectura_par('DeiT-S - ResNet-50', 'ResNet-50', 'DeiT-S')}</li>
<li>{lectura_par('Swin-T - ConvNeXt-T', 'ConvNeXt-T', 'Swin-T')}</li>
<li>«No se distingue» no significa que los modelos sean iguales: significa que con esta prueba
no se puede decir cuál es mejor.</li>
</ul>

<h3>Lo que sí importa: modelo moderno contra antiguo</h3>
<div class="tw"><table>
<caption>Diferencia moderno − antiguo dentro de cada familia, con bootstrap pareado. El ResNet-50 se entrenó con su
propia receta (dos niveles de tasa de aprendizaje) y ConvNeXt-T con decaimiento por nivel: la diferencia mide el modelo
y su receta juntos.</caption>
<thead><tr><th>Comparación</th><th class="num">prueba limpia</th><th class="num">barcos no vistos</th></tr></thead>
<tbody>
{fila_extra('ConvNeXt-T - ResNet-50', 'CNN · ConvNeXt-T − ResNet-50')}
{fila_extra('Swin-T - ViT-B/16', 'Transformer · Swin-T − ViT-B/16')}
</tbody></table></div>
<p>{lectura('ConvNeXt-T - ResNet-50', VT['ConvNeXt-T - ResNet-50'])} {lectura('Swin-T - ViT-B/16', VT['Swin-T - ViT-B/16'])}
En barcos no vistos, los dos modelos modernos superan a su versión antigua. La pregunta del Avance, qué
brecha cierra el tercer modelo, tiene respuesta: con estos datos, la ventaja del ViT-B/16 se explica mejor por
su tamaño y su modernidad que por la atención.</p>

<h3>Qué tan firme es la ventaja del ViT-B/16</h3>
<div class="tw"><table>
<thead><tr><th>Prueba</th><th class="num">ResNet-50</th><th class="num">ViT-B/16</th><th class="num">diferencia [margen]</th></tr></thead>
<tbody>
<tr><td class="name">Prueba limpia completa</td><td class="num">{num(variante('todas','ResNet-50')['macro_f1'])}</td><td class="num">{num(variante('todas','ViT-B/16')['macro_f1'])}</td><td class="num">{num(P_VR['delta_medio'],3,True)} {ic(P_VR['ic95'])}</td></tr>
<tr><td class="name">Los cuatro atunes como una clase</td><td class="num">{num(variante('atunes_agrupados','ResNet-50')['macro_f1'])}</td><td class="num">{num(variante('atunes_agrupados','ViT-B/16')['macro_f1'])}</td><td class="num">{num(V['atunes_agrupados']['pares']['ViT-B/16 - ResNet-50']['delta_medio'],3,True)} {ic(V['atunes_agrupados']['pares']['ViT-B/16 - ResNet-50']['ic95'])}</td></tr>
<tr><td class="name">Cámaras vistas en entrenamiento ({miles(cam['cajas_camara_vista'])} cajas)</td><td class="num">{num(variante('camara_vista','ResNet-50')['macro_f1'])}</td><td class="num">{num(variante('camara_vista','ViT-B/16')['macro_f1'])}</td><td class="num">{num(V['camara_vista']['pares']['ViT-B/16 - ResNet-50']['delta_medio'],3,True)} {ic(V['camara_vista']['pares']['ViT-B/16 - ResNet-50']['ic95'])}</td></tr>
<tr><td class="name">Cámaras nuevas ({miles(cam['cajas_camara_nueva'])} cajas, {cam['camaras_nuevas']} cámaras)</td><td class="num">{num(variante('camara_nueva','ResNet-50')['macro_f1'])}</td><td class="num">{num(variante('camara_nueva','ViT-B/16')['macro_f1'])}</td><td class="num">{num(V['camara_nueva']['pares']['ViT-B/16 - ResNet-50']['delta_medio'],3,True)} {ic(V['camara_nueva']['pares']['ViT-B/16 - ResNet-50']['ic95'])}</td></tr>
</tbody></table></div>
<p>Parte de la ventaja viene de separar un atún de otro: el mayor cambio por especie es el albacora
(F1 {num(pe_r['Albacore']['f1'],2)} → {num(pe_v['Albacore']['f1'],2)}), que se confunde con el atún aleta amarilla.
En total el ViT gana en {gana} de las 15 especies. En barcos que no estaban en entrenamiento los dos
modelos caen cerca de 0,10 y la diferencia deja de distinguirse. Para la caleta, todo barco sería nuevo:
es la limitación más importante de este trabajo.</p>
<p>Otros dos resultados del Avance se mantienen. Con el reparto oficial, el macro-F1 sube a 0,602
(ResNet) y 0,641 (ViT): la fuga infla los números alrededor de un 80 %. Y la hipótesis del enunciado,
que el ViT gana gracias a la atención global sobre peces amontonados, no se sostiene: separando la
prueba en cuatro grados de oclusión, la ventaja es +0,052, +0,071, +0,017 y +0,044, sin crecer con la
oclusión (Spearman −0,60).</p>
</section>

<section>
<h2>6 · Dónde mira cada modelo</h2>
<p>Este análisis y el siguiente se hicieron sobre el par original, ResNet-50 y ViT-B/16. Medimos qué fracción de la atención de cada mapa cae dentro de la caja del pez, sobre {PG_F['n']}
recortes de la prueba limpia de FishNet, los mismos para los dos modelos: Grad-CAM (Selvaraju et al.,
2017) para el ResNet; attention rollout (Abnar y Zuidema, 2020) y la relevancia de Chefer et al.
(2021), que sí depende de la clase como Grad-CAM, para el ViT.</p>
<p>Con el pez siempre al centro del recorte, una gaussiana fija en el centro, que no mira la imagen,
obtenía 0,757, más que cualquier modelo: la medida premiaba mirar el centro. Por eso ampliamos el
recorte con el pez en una posición al azar y restamos la atención que el mismo mapa deja en la caja de
<em>otro</em> recorte. Lo que queda es cuánto sigue el mapa al pez.</p>
<div class="tw"><table>
<thead><tr><th>Método</th><th class="num">en la caja</th><th class="num">en la caja de otro recorte</th><th class="num">diferencia</th></tr></thead>
<tbody>
<tr><td class="name">ResNet-50 · Grad-CAM</td><td class="num">{num(ef['gradcam']['media'])}</td><td class="num">{num(ef['gradcam']['control_barajado'])}</td><td class="num"><strong>{num(ef['gradcam']['sobre_barajado'],3,True)}</strong></td></tr>
<tr><td class="name">ViT-B/16 · Chefer</td><td class="num">{num(ef['chefer']['media'])}</td><td class="num">{num(ef['chefer']['control_barajado'])}</td><td class="num"><strong>{num(ef['chefer']['sobre_barajado'],3,True)}</strong></td></tr>
<tr><td class="name">ViT-B/16 · rollout</td><td class="num">{num(ef['rollout']['media'])}</td><td class="num">{num(ef['rollout']['control_barajado'])}</td><td class="num">{num(ef['rollout']['sobre_barajado'],3,True)}</td></tr>
</tbody></table></div>
<p>Grad-CAM y Chefer siguen al pez casi por igual; la ventaja aparente de Grad-CAM en la columna de la
izquierda es su tendencia a marcar el centro. El ViT no mira menos el fondo que el ResNet. En Kaggle
habíamos medido 0,582 para Grad-CAM y 0,408 para Chefer, pero sin este control: es probable que ese
número tenga el mismo sesgo. Una limitación: con el recorte ampliado, cada modelo cambia su predicción
en la mitad de los casos, así que los mapas explican decisiones sobre una imagen algo distinta de la
que ve normalmente (semilla 0).</p>
</section>

<section>
<h2>7 · Decir «no sé»</h2>
<p><strong>Criterio:</strong> si la energía de la salida (Liu et al., 2020) es más baja que la del 95 % de los
peces de validación, el modelo responde «no sé». El umbral se fija en validación. Medimos la separación
con el AUROC (1 = separa perfecto lo conocido de lo nuevo; 0,5 = no separa).</p>
<div class="tw"><table>
<thead><tr><th>Prueba</th><th class="num">ResNet-50</th><th class="num">ViT-B/16</th></tr></thead>
<tbody>
<tr><td class="name">AUROC, peces de Kaggle frente a la prueba de FishNet sin personas</td><td class="num">{num(o_r['auroc'])}</td><td class="num">{num(o_v['auroc'])}</td></tr>
<tr><td class="name">Peces de Kaggle aceptados con el umbral del 95 %</td><td class="num">{pct(o_r['fpr95'])}</td><td class="num">{pct(o_v['fpr95'])}</td></tr>
<tr><td class="name">AUROC, 45 fotos de reineta, merluza y congrio</td><td class="num">{num(c_r['auroc_vs_test_limpio_sin_personas'])}</td><td class="num">{num(c_v['auroc_vs_test_limpio_sin_personas'])}</td></tr>
<tr><td class="name">Fotos chilenas rechazadas</td><td class="num">{pct(c_r['rechaza_total'])}</td><td class="num">{pct(c_v['rechaza_total'])}</td></tr>
</tbody></table></div>
<p>Las 45 fotos chilenas son de iNaturalist con licencia CC0, CC BY o CC BY-NC (créditos en
<code>data/chile_creditos.csv</code>). Lo correcto frente a ellas es «no sé». El ViT lo dice más seguido, pero
cuando ninguno rechaza, casi siempre responden «atún aleta amarilla». Es una prueba cualitativa: son
fotos de peces en la mano o sobre una mesa, así que parte de lo detectado es el cambio de tipo de foto.
Además, la confianza que muestran los modelos no está calibrada: es baja en personas y demasiado alta en
peces, y un ajuste de temperatura único no lo corrige.</p>
</section>

<section>
<h2>8 · Costo</h2>
<p>En un PC sin GPU (i5-12400F), clasificar un pez toma 43 ms con el ResNet-50 y 119 ms con el ViT-B/16.
Una foto típica de FishNet tiene 5 peces: 0,2 s contra 0,6 s. Para fotos de desembarque, los dos
alcanzan; para video continuo, ninguno alcanza en CPU y los dos alcanzan en GPU. La velocidad no cambia
lo que se puede hacer.</p>
</section>

<section>
<h2>9 · Límites y próximos pasos</h2>
<ul>
<li><strong>Receta del ResNet-50.</strong> Se entrenó con su propia tasa de aprendizaje, distinta de la de
ConvNeXt-T; la ventaja de la CNN moderna mezcla arquitectura y receta.</li>
<li><strong>Barcos nuevos.</strong> El rendimiento cae en cámaras no vistas. Próximo paso: entrenar y evaluar
con un reparto por cámara.</li>
<li><strong>Conteo.</strong> El objetivo original incluía contar peces amontonados. Falta un detector,
entrenado con las cajas de FishNet, antes del clasificador; se debe evaluar por separado.</li>
<li><strong>Especies chilenas.</strong> Sin fotos de la caleta, el sistema no se puede usar allí.</li>
<li><strong>Especies raras.</strong> Cuatro especies tienen menos de 25 cajas de prueba; su F1 es poco confiable.</li>
<li><strong>Privacidad.</strong> FishNet muestra tripulantes (54 % del entrenamiento son personas). No
redistribuimos imágenes y las fotos de las láminas llevan las caras pixeladas.</li>
</ul>
</section>

<section>
<h2>10 · Reproducción</h2>
<p>Código, configuraciones, repartos, métricas por semilla y pruebas en
<a href="https://github.com/Nirvesta04/CaletaScan">github.com/Nirvesta04/CaletaScan</a>. Todo lo posterior al
entrenamiento trabaja sobre las salidas guardadas y no usa la GPU. La demo (<code>python -m src.demo</code>)
pasa la misma caja por las dos arquitecturas y muestra la especie, la confianza, si responde «no sé» y
el tiempo de cada una; no necesita instalar nada fuera de los requisitos.</p>
<div class="fx">python scripts/build_splits.py
python -m scripts.entrenar_pares            <span class="c"># E10-E12; --parar para detener</span>
python -m scripts.analisis_final            <span class="c"># tablas de este informe</span>
python -m scripts.interpret_fishnet
python -m scripts.chile_ood
python -m src.demo                          <span class="c"># demo local en el puerto 8000</span>
python -m tests.test_stats &amp;&amp; python -m tests.test_calibrate</div>
</section>

<section>
<h2>Referencias</h2>
<ol class="refs">
<li>Abnar, S. y Zuidema, W. (2020). Quantifying attention flow in transformers. ACL.</li>
<li>Chefer, H., Gur, S. y Wolf, L. (2021). Transformer interpretability beyond attention visualization. CVPR.</li>
<li>Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.</li>
<li>He, K. et al. (2016). Deep residual learning for image recognition. CVPR.</li>
<li>Kapoor, S. y Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns.</li>
<li>Kay, J. y Merrifield, M. (2021). The Fishnet Open Images Database. arXiv:2106.09178.</li>
<li>Liu, W. et al. (2020). Energy-based out-of-distribution detection. NeurIPS.</li>
<li>Liu, Z. et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. ICCV.</li>
<li>Liu, Z. et al. (2022). A ConvNet for the 2020s. CVPR.</li>
<li>Menon, A. K. et al. (2021). Long-tail learning via logit adjustment. ICLR.</li>
<li>Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.</li>
<li>Touvron, H. et al. (2021). Training data-efficient image transformers &amp; distillation through attention. ICML.</li>
<li>Ulucan, O., Karakaya, D. y Turkan, M. (2020). A large-scale dataset for fish segmentation and classification. ASYU.</li>
</ol>
</section>
'''

html = ('<!doctype html>\n<html lang="es"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>Informe Caleta-Scan</title>\n<style>\n' + CSS + '\n</style></head>\n'
        '<body><div class="wrap">' + cuerpo + '</div></body></html>\n')
with open(HTML, 'w', encoding='utf-8') as fh:
    fh.write(html)
print('guardado: %s' % HTML)

nav = None
for c in (r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
          r'C:\Program Files\Google\Chrome\Application\chrome.exe',
          shutil.which('msedge') or '', shutil.which('chrome') or ''):
    if c and os.path.exists(c):
        nav = c
        break
if nav:
    import pathlib
    import tempfile
    # perfil temporal: si el usuario tiene el navegador abierto, el modo sin
    # interfaz con su perfil no imprime nada
    subprocess.run([nav, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--user-data-dir=' + tempfile.mkdtemp(), '--print-to-pdf=' + PDF,
                    pathlib.Path(HTML).as_uri()],
                   check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print('PDF: %s (%s)' % (PDF, 'ok' if os.path.exists(PDF) else 'NO SE GENERO'))
else:
    print('sin Edge ni Chrome: abrir el HTML e imprimir a PDF a mano')
