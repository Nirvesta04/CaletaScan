# -*- coding: utf-8 -*-
"""
Mazo de la presentacion final de Caleta-Scan (G3). El mismo mazo sirve para la
clase y para la ExpoData.

    python -m scripts.figuras_mazo          # fotos de los datos (no versionadas)
    python -m scripts.analisis_final        # cifras
    python -m scripts.graficos_mazo         # graficos de resultados (docs/figs/g_*.png)
    python -m scripts.build_pptx_final      # -> entrega/Caleta-Scan-Final-G3.pptx

Todas las cifras se leen de runs/*.json: nada se copia a mano. El PPTX lleva
fotos de Kaggle, FishNet e iNaturalist, por eso se escribe en entrega/, que no
se versiona.

Criterios tras el Avance: un mensaje por lamina, poco texto, fotos de los datos
donde se habla de ellos y graficos en vez de tablas. La explicacion va en las
notas del orador. Colores del logo (azul marino); en los graficos, las CNN en
naranjo y los transformers en azul.
"""
import json
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

from scripts.lecturas import lectura

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
PRIV = os.path.join(ROOT, 'docs', 'figs_privadas')
FIGS = os.path.join(ROOT, 'docs', 'figs')
SALIDA = os.path.join(ROOT, 'entrega', 'Caleta-Scan-Final-G3.pptx')

TITULO = '¿CNN o transformer para reconocer peces?'
ESLOGAN = 'Con un buen modelo, el límite es la foto.'


def J(nombre):
    with open(os.path.join(RUNS, nombre), encoding='utf-8') as fh:
        return json.load(fh)


A = J('analisis_final.json')
OF = J('oficial_por_modelo.json')
PG_F = J('pointing_game_fishnet.json')
PG_K = J('pointing_game.json')
CHILE = J('chile_ood.json')
CAL = J('calibracion_ood.json')
E7 = J('comparacion_e7.json')
KAG = J('comparacion_e1_e2.json')
INV = J('investigacion.json')


def ood(run, sub):
    k = [k for k in CAL['ood'] if k.startswith(run) and k.endswith(sub)][0]
    return CAL['ood'][k]['energia']


def num(x, d=3, signo=False):
    s = ('%+.*f' if signo else '%.*f') % (d, x)
    # signo menos tipografico: con el guion, PowerPoint puede cortar la linea
    # entre el signo y el numero
    return s.replace('.', ',').replace('-', '−')


def pct(x, d=0):
    return ('%.*f' % (d, 100 * x)).replace('.', ',') + ' %'


def ic(v, signo=False):
    return '%s a %s' % (num(v[0], 3, signo), num(v[1], 3, signo))


# ---------------------------------------------------------------- estilo
PX = 6350
def E(px):  return Emu(int(round(px * PX)))
def P(px):  return Pt(px / 2.0)
def C(h):   return RGBColor.from_string(h)

DARK, LIGHT, WHITE, BORDER = '0D2240', 'F5F7FA', 'FFFFFF', 'DCE2EA'
ACC, ACCL, WARM, WARMBG = '1F5F99', '7FB3E0', 'B3541E', 'F7EADF'
OK = '0C6F66'                                  # "se distingue", igual que en los graficos
INK, MUTED, FOOT = '14213D', '55657A', '7D8A99'
ONDARK, DMUTED = 'F2F5F9', 'A9B8CC'
RESN, VITC = 'B3541E', '1B3F73'
FONT = 'Segoe UI'
LOGO = os.path.join(FIGS, 'logo_circulo.png')

prs = Presentation()
prs.slide_width, prs.slide_height = E(1920), E(1080)
BLANK = prs.slide_layouts[6]
N = [0]


def nueva(fondo=LIGHT):
    s = prs.slides.add_slide(BLANK)
    f = s.background.fill
    f.solid(); f.fore_color.rgb = C(fondo)
    N[0] += 1
    return s


def texto(s, x, y, w, h, partes, size, color=INK, bold=False, line=1.25, align='l',
          anchor=None, space_after=0):
    box = s.shapes.add_textbox(E(x), E(y), E(w), E(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if anchor:
        tf.vertical_anchor = {'m': MSO_ANCHOR.MIDDLE, 'b': MSO_ANCHOR.BOTTOM}[anchor]
    if isinstance(partes, str):
        partes = [partes]
    parrafos = partes if isinstance(partes[0], list) else [partes]
    for i, par in enumerate(parrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {'l': PP_ALIGN.LEFT, 'r': PP_ALIGN.RIGHT, 'c': PP_ALIGN.CENTER}[align]
        p.line_spacing = line
        if space_after:
            p.space_after = P(space_after)
        for t in (par if isinstance(par, list) else [par]):
            if isinstance(t, str):
                t = (t, bold, color)
            elif len(t) == 2:
                t = (t[0], t[1], color)
            r = p.add_run(); r.text = t[0]
            r.font.size = P(size); r.font.bold = t[1]
            r.font.name = FONT; r.font.color.rgb = C(t[2])
    return box


def caja(s, x, y, w, h, fill=WHITE, borde=BORDER, radio=18):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, E(x), E(y), E(w), E(h))
    sh.adjustments[0] = radio / float(min(w, h))
    sh.fill.solid(); sh.fill.fore_color.rgb = C(fill)
    if borde:
        sh.line.color.rgb = C(borde); sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def titulo(s, ojo, t, sub=None, oscuro=False):
    texto(s, 120, 96, 1680, 40, ojo.upper(), 24, ACCL if oscuro else ACC, bold=True)
    texto(s, 120, 146, 1680, 140, t, 62, ONDARK if oscuro else INK, bold=True, line=1.08)
    if sub:
        texto(s, 120, 238, 1680, 60, sub, 30, DMUTED if oscuro else MUTED, line=1.3)


def pie(s, izq='', oscuro=False):
    col = DMUTED if oscuro else FOOT
    texto(s, 120, 1010, 1560, 40, izq or 'Caleta-Scan · G3', 20, col)
    texto(s, 1680, 1010, 120, 40, str(N[0]), 22, col, align='r')


def notas(s, t):
    s.notes_slide.notes_text_frame.text = t


def imagen(s, ruta, x, y, w=None, h=None):
    if not os.path.exists(ruta):
        caja(s, x, y, w or 400, h or 300, 'EEEEEE', 'CCCCCC')
        texto(s, x + 20, y + 20, (w or 400) - 40, 60, 'falta: ' + os.path.basename(ruta), 20, WARM)
        return None
    return s.shapes.add_picture(ruta, E(x), E(y), E(w) if w else None, E(h) if h else None)


def figura(s, nombre, x, y, w=None, h=None):
    return imagen(s, os.path.join(FIGS, nombre), x, y, w, h)


def foto(s, nombre, x, y, w=None, h=None):
    return imagen(s, os.path.join(PRIV, nombre), x, y, w, h)


def leyenda(s, x, y, w, t, size=22):
    texto(s, x, y, w, 40, t, size, MUTED, line=1.25)


def cifra(s, x, y, w, valor, etiqueta, col=ACC, vsize=96, esize=28, ecol=MUTED):
    texto(s, x, y, w, 120, valor, vsize, col, bold=True, line=1.0)
    texto(s, x, y + vsize * 1.25, w, 140, etiqueta, esize, ecol, line=1.3)


def tabla(s, x, y, w, filas, anchos, alto_fila=64, size=26, cab_fill=DARK):
    """filas[0] es la cabecera. Celdas: str o (str, bold, color)."""
    nf, nc = len(filas), len(filas[0])
    t = s.shapes.add_table(nf, nc, E(x), E(y), E(w), E(alto_fila * nf)).table
    tot = float(sum(anchos))
    for j, a in enumerate(anchos):
        t.columns[j].width = E(w * a / tot)
    for i, fila in enumerate(filas):
        t.rows[i].height = E(alto_fila)
        for j, celda in enumerate(fila):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = E(16)
            c.margin_top = c.margin_bottom = E(6)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            if isinstance(celda, str):
                celda = (celda, i == 0, ONDARK if i == 0 else INK)
            tf = c.text_frame
            tf.text = ''
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = celda[0]
            r.font.size = P(size); r.font.bold = celda[1]
            r.font.name = FONT; r.font.color.rgb = C(celda[2])
            c.fill.solid()
            c.fill.fore_color.rgb = C(cab_fill if i == 0 else (WHITE if i % 2 else 'EEF2F6'))
    return t


# ---------------------------------------------------------------- grafico de anexo
def figura_vit_resnet():
    """Bosque ViT-B/16 - ResNet-50 en las tres pruebas (anexo)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Segoe UI'
    filas = []
    for v, etiqueta in (('todas', 'prueba limpia completa'), ('atunes_agrupados', 'atunes como una clase'),
                        ('camara_nueva', 'solo barcos no vistos')):
        d = A['por_variante'][v]['pares'].get('ViT-B/16 - ResNet-50')
        if d:
            filas.append(('ViT-B/16 − ResNet-50' + chr(10) + '(%s)' % etiqueta, d))
    fig, ax = plt.subplots(figsize=(8, 1.2 + 1.0 * len(filas)), dpi=200)
    for i, (nombre, d) in enumerate(filas[::-1]):
        col = '#0C6F66' if d['distinguible'] else '#8A989B'
        ax.plot(d['ic95'], [i, i], color=col, lw=4, solid_capstyle='round')
        ax.plot([d['delta_medio']], [i], 'o', color=col, ms=11)
        ax.text(max(d['ic95'][1], 0.0) + 0.006, i, num(d['delta_medio'], 3, True),
                va='center', fontsize=13, color=col, fontweight='bold')
    ax.axvline(0, color='#B3541E', lw=1.5, ls='--')
    ax.set_yticks(range(len(filas)), [n for n, _ in filas[::-1]], fontsize=12)
    ax.set_xlabel('diferencia de macro-F1 con su margen al 95 %', fontsize=12)
    for lado in ('top', 'right', 'left'):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(-0.06, 0.13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'diferencias.png'), transparent=True)
    plt.close(fig)


# ================================================================ LAMINAS
NOM = ['Patricio Toledo', 'Andrés Saavedra', 'Sebastián Casas', 'Bastián Molina']
M = {v['modelo']: v for v in A['modelos'].values()}
PV = A['por_variante']
VT = PV['todas']['pares']
fg = A['fuga_split_oficial']
cam = A['camaras']
figura_vit_resnet()

# 1 ----------------------------------------------------------------- portada
s = nueva(DARK)
imagen(s, LOGO, 120, 170, h=740)
x0 = 960
texto(s, x0, 210, 840, 40, 'PUCV · APRENDIZAJE AUTOMÁTICO II · GRUPO G3', 22, ACCL, bold=True)
texto(s, x0, 270, 840, 110, 'Caleta-Scan', 96, ONDARK, bold=True, line=1.0)
texto(s, x0, 400, 840, 220, TITULO, 58, ONDARK, bold=True, line=1.1)
s.shapes.add_connector(1, E(x0), E(640), E(x0 + 120), E(640)).line.color.rgb = C(ACCL)
texto(s, x0, 670, 840, 140, ESLOGAN, 34, ACCL, line=1.3)
texto(s, x0, 860, 840, 120, [[n] for n in NOM], 24, DMUTED, line=1.2)
notas(s, 'Somos el grupo G3 y nuestro proyecto es Caleta-Scan. La pregunta es simple: para reconocer '
         'la especie de un pez en una foto de faena, ¿conviene una red convolucional (CNN) o un '
         'transformer? La respuesta corta: a igual tamaño y época, da lo mismo. Y con un modelo '
         'moderno, lo que más limita el resultado deja de ser el modelo y pasa a ser la foto: qué tan '
         'grande se ve el pez, la luz y la cámara. En 15 minutos mostramos los datos, cómo comparamos '
         'de forma justa, qué encontramos y qué recomendamos para la caleta.')

# 2 ----------------------------------------------------------------- problema
s = nueva()
titulo(s, 'Problema', 'En la Caleta Portales, el desembarque se registra a mano')
for i, (t, d) in enumerate([('Lento', 'Se anota pez por pez mientras se descarga.'),
                             ('Variable', 'Depende de quién esté de turno.'),
                             ('Insuficiente', 'En temporada alta llegan más botes que inspectores.')]):
    x = 120 + i * 570
    caja(s, x, 360, 530, 300)
    texto(s, x + 44, 400, 450, 60, t, 44, ACC, bold=True)
    texto(s, x + 44, 480, 450, 160, d, 30, MUTED, line=1.35)
texto(s, 120, 740, 1680, 120, [('Objetivo: ', True, INK),
                              ('que una foto proponga la especie de cada pez, para que el inspector '
                               'revise y corrija.', False, INK)], 34, line=1.3)
pie(s)
notas(s, 'Hoy un inspector de Sernapesca anota a mano la especie y la cantidad durante la descarga. '
         'Es lento, depende de quién esté de turno y no alcanza en temporada alta. La idea es apoyar, '
         'no reemplazar: el sistema propone una especie con su confianza, o dice "no sé", y la persona '
         'decide.')

# 3 ----------------------------------------------------------------- setup
s = nueva()
titulo(s, 'Qué hace el sistema', 'Recibe un pez marcado y responde su especie o «no sé»')
pasos = [('Entrada', 'El recorte de un pez. La caja la marca una persona o viene anotada.'),
         ('Modelo', 'Una CNN o un transformer, preentrenados en ImageNet y ajustados a peces.'),
         ('Salida', 'La especie más probable, o «no sé» si la imagen no se parece a lo aprendido.')]
for i, (t, d) in enumerate(pasos):
    x = 120 + i * 580
    caja(s, x, 330, 520, 330)
    texto(s, x + 40, 366, 440, 50, '%d · %s' % (i + 1, t), 36, ACC, bold=True)
    texto(s, x + 40, 436, 440, 210, d, 29, INK, line=1.35)
    if i < 2:
        texto(s, x + 524, 450, 56, 80, '→', 54, FOOT, align='c')
caja(s, 120, 720, 1680, 200, WARMBG, None)
texto(s, 170, 752, 1580, 150,
      [[('No detecta peces ni los cuenta. ', True, WARM),
        ('Encontrar y contar los peces queda como próximo paso.', False, INK)],
       [('No conoce especies chilenas. ', True, WARM),
        ('Reineta, merluza y congrio no aparecen en ningún conjunto público.', False, INK)]],
      30, line=1.35, space_after=10)
pie(s)
notas(s, 'La tarea es de clasificación: el pez ya viene marcado con una caja. No detectamos ni '
         'contamos; eso queda como próximo paso. Y no hay especies chilenas en los datos públicos, así '
         'que lo que mostramos es el método; al final probamos qué hacen los modelos con fotos de '
         'especies chilenas.')

# 4 ----------------------------------------------------------------- modelos
s = nueva()
titulo(s, 'Modelos', 'Cinco modelos en tres pares con pesos parejos')
filas = [['Par', 'CNN', 'Transformer', 'Qué responde'],
         ['Original', 'ResNet-50 · 2015 · 26 M', 'ViT-B/16 · 2020 · 87 M', 'La pregunta del curso'],
         ['Mismo tamaño', 'ResNet-50 · 26 M', 'DeiT-S · 22 M', '¿Ganaba el ViT por ser más grande?'],
         ['Moderno', 'ConvNeXt-T · 2022 · 29 M', 'Swin-T · 2021 · 28 M', '¿Perdía el ResNet por antiguo?']]
tabla(s, 120, 330, 1680, filas, [2.0, 3.0, 3.0, 4.0], alto_fila=100, size=28)
texto(s, 120, 780, 1680, 160,
      'M = millones de parámetros. Pesos de ImageNet con acierto parejo dentro de cada par '
      '(diferencia de un punto o menos): ResNet-50 80,9 %, ViT-B/16 81,1 %, DeiT-S 79,9 %, '
      'ConvNeXt-T 82,5 %, Swin-T 81,5 %.', 24, MUTED, line=1.35)
pie(s)
notas(s, 'Respuesta a la pregunta 1 del profesor: qué modelos y con qué pesos exactos. '
         'Par original: ResNet-50 con pesos IMAGENET1K_V2 contra ViT-B/16 IMAGENET1K_V1, con casi el '
         'mismo acierto en ImageNet. Par de mismo tamaño: DeiT-S, con los pesos oficiales de Facebook, '
         'tiene el tamaño del ResNet-50; responde si el ViT-B/16 ganaba por ser transformer o por ser '
         '3,4 veces más grande. Par moderno: ConvNeXt-T y Swin-T, del mismo tamaño y costo; responde '
         'si el ResNet perdía por ser de 2015.')

# 5 ----------------------------------------------------------------- datos kaggle
s = nueva()
titulo(s, 'Datos · Kaggle', 'Fondo de estudio, y el mismo pez girado 20 veces')
foto(s, 'kaggle_especies.jpg', 120, 300, w=830)
foto(s, 'kaggle_rotaciones.jpg', 1010, 300, w=790)
leyenda(s, 120, 690, 830, 'Las 9 especies: cada una sobre su propia mesa y con su propia luz.')
leyenda(s, 1010, 650, 790, '8 de las 20 copias giradas y espejadas de una misma foto original.')
cifra(s, 120, 790, 800, '9.000', 'imágenes, pero solo 411 fotos originales', vsize=72, esize=26)
cifra(s, 1010, 790, 790, '62', 'fotos independientes en la prueba, no 1.336', vsize=72, esize=26, col=WARM)
pie(s, 'Fotos: Ulucan et al. (2020), A Large Scale Fish Dataset, CC BY 4.0')
notas(s, 'Kaggle es un conjunto de estudio: peces de supermercado fotografiados sobre una mesa, con '
         'buena luz. Cada especie tiene su mesa y su iluminación. Y las 1.000 imágenes por especie no '
         'son 1.000 fotos: son unas 21 copias giradas y espejadas de cada foto original. Por eso '
         'repartimos por foto original, y la prueba tiene 62 fotos independientes, no 1.336.')

# 6 ----------------------------------------------------------------- datos fishnet
s = nueva()
titulo(s, 'Datos · FishNet', 'Barcos reales: y 1 de cada 9 fotos no tiene ningún pez')
foto(s, 'fishnet_ejemplo_3.jpg', 120, 300, w=900)
for i, f in enumerate(['fishnet_vacia_1.jpg', 'fishnet_vacia_3.jpg', 'fishnet_solo_personas.jpg']):
    foto(s, f, 1060, 300 + i * 205, w=330)
leyenda(s, 1410, 330, 390, 'Cubierta vacía, de noche')
leyenda(s, 1410, 535, 390, 'Cubierta vacía, de día')
leyenda(s, 1410, 740, 390, 'Solo tripulantes')
leyenda(s, 120, 830, 900, 'Cajas anotadas: cada pez marcado por un anotador. Peces amontonados, '
        'tapados y de noche.')
texto(s, 1060, 930, 740, 50, '%s de las fotos no tiene peces' % pct(2166 / 20002, 1), 26, WARM, bold=True)
pie(s, 'Fotos: Kay y Merrifield (2021), FishNet Open Images, CC BY-SA 4.0 · caras pixeladas')
notas(s, 'FishNet son cámaras fijas en la cubierta de barcos pesqueros: 20.002 fotos de 77 cámaras, '
         'con 99.019 cajas anotadas. A la izquierda, una foto con peces amontonados y sus cajas. A la '
         'derecha, lo que el sistema también tiene que manejar: 2.166 fotos (10,8 %) no tienen ningún '
         'pez, solo cubierta vacía, agua o tripulantes. Por eso el modelo tiene clases "persona" y '
         '"sin pez", además de las especies. Respuesta a la pregunta 4 del profesor: qué es FishNet.')

# 7 ----------------------------------------------------------------- composicion
s = nueva()
titulo(s, 'Datos · qué hay en la prueba', 'Dos de cada tres cajas son personas, no peces')
figura(s, 'g_composicion.png', 120, 300, w=1680)
caja(s, 120, 830, 1680, 140, 'EEF2F6', None)
texto(s, 160, 852, 1600, 110,
      [[('Por eso no usamos el porcentaje de aciertos: ', True, INK),
        ('un modelo que siempre diga «persona» acierta el 68 %. Usamos macro-F1: el promedio del F1 '
         'de las 15 especies, de 0 a 1, donde cada especie pesa lo mismo.', False, MUTED)]],
      26, line=1.35)
pie(s, 'Prueba limpia de FishNet: 7.159 cajas de 612 lances')
notas(s, 'Así se reparte la prueba limpia: el 68 % de las cajas son personas, un 2 % son cubiertas sin '
         'pez, un 9 % son peces sin especie resuelta (desconocido, atún sin especie o especies raras), '
         'y solo el 20 % son las 15 especies que evaluamos. Con ese desbalance, la accuracy engaña. El '
         'macro-F1 promedia las 15 especies por igual: va de 0 a 1 y no es un porcentaje de aciertos.')

# 8 ----------------------------------------------------------------- fuga
lo, hi = (min(OF[m] / M[m]['macro_f1'] for m in OF) - 1, max(OF[m] / M[m]['macro_f1'] for m in OF) - 1)
s = nueva()
titulo(s, 'Riesgo 1 · fuga', 'Fotos de un mismo lance quedaban a ambos lados')
for i in range(4):
    nombre = [f for f in os.listdir(PRIV) if f.startswith('fishnet_seq_%d_' % (i + 1))] if os.path.isdir(PRIV) else []
    x = 120 + i * 425
    if nombre:
        foto(s, nombre[0], x, 290, w=400)
        es_test = 'test' in nombre[0]
        caja(s, x, 522, 400, 46, WARM if es_test else ACC, None, radio=10)
        texto(s, x, 522, 400, 46, 'prueba' if es_test else 'entrenamiento', 24, WHITE, bold=True,
              align='c', anchor='m')
figura(s, 'g_fuga.png', 120, 600, h=400)
cifra(s, 1060, 620, 740, pct(fg['frac_cajas'], 1),
      'de la prueba oficial comparte lance con el entrenamiento. Lo sacamos: la prueba limpia '
      'tiene 7.159 cajas de 612 lances.', col=WARM, vsize=80, esize=25)
pie(s, 'Arriba: cuatro fotos seguidas de un mismo lance · abajo: macro-F1 con el reparto oficial y con la prueba limpia')
notas(s, 'Arriba, cuatro fotos seguidas del mismo lance: el reparto oficial de FishNet puso tres en '
         'entrenamiento y una en prueba, así que el modelo se examina con una foto casi igual a una que '
         'ya vio. Le pasa al %s de las cajas de prueba (%s de las imágenes). Abajo, el efecto: con el '
         'reparto oficial todos los modelos salen entre %s y %s más altos. Todo lo que sigue usa la '
         'prueba limpia.' % (pct(fg['frac_cajas'], 1), pct(fg['frac_imagenes'], 1), pct(lo), pct(hi)))

# 9 ----------------------------------------------------------------- atajo kaggle
k_f = E7['e1_resnet50_kaggle__vs__e7_resnet50_solo_fondo']
s = nueva()
titulo(s, 'Riesgo 2 · atajo', 'En Kaggle, el fondo basta para saber la especie')
for i, (f, et) in enumerate([('kaggle_original.jpg', 'Foto completa'), ('kaggle_sin_pez.jpg', 'Sin el pez'),
                             ('kaggle_solo_pez.jpg', 'Solo el pez')]):
    x = 120 + i * 300
    foto(s, f, x, 300, w=280)
    texto(s, x, 520, 280, 40, et, 24, INK, bold=True, align='c')
figura(s, 'g_kaggle.png', 1040, 290, w=760)
cifra(s, 120, 640, 860, num(k_f['macro_f1_b']),
      'de macro-F1 sin ver el pez: la mesa y la luz delatan la especie. Por eso Kaggle no decide '
      'la comparación.', col=WARM, vsize=84, esize=26)
pie(s, 'Foto: Ulucan et al. (2020), CC BY 4.0 · gris = zona borrada con la silueta anotada')
notas(s, 'Borramos el pez con la silueta que trae Kaggle y reentrenamos los dos modelos. Sin ver el '
         'pez siguen en 0,97: la mesa y la luz identifican la especie. Repartir por foto original no '
         'arregla esto, porque el fondo es de la especie, no de la foto. Por eso la comparación se '
         'decide en FishNet.')

# 10 ---------------------------------------------------------------- protocolo
s = nueva()
titulo(s, 'Protocolo', 'Los cinco modelos reciben exactamente lo mismo')
for i, (v, e) in enumerate([('9.104', 'pasos de entrenamiento (8 épocas, lote de 64)'),
                            ('3', 'semillas por modelo'),
                            ('224 px', 'misma resolución y mismos aumentos')]):
    cifra(s, 120 + i * 570, 320, 520, v, e, vsize=84)
caja(s, 120, 620, 1680, 320)
texto(s, 170, 652, 1580, 50, 'Cómo leer los resultados', 32, INK, bold=True)
texto(s, 170, 712, 1580, 220,
      [[('macro-F1: ', True, INK), ('de 0 a 1; cada una de las 15 especies pesa lo mismo.', False, MUTED)],
       [('Margen: ', True, INK), ('intervalo de confianza al 95 %, remuestreando lances completos.', False, MUTED)],
       [('Se distingue: ', True, INK), ('cuando el margen de la diferencia no incluye el cero.', False, MUTED)]],
      28, line=1.35, space_after=12)
pie(s)
notas(s, 'Misma cantidad de pasos, mismo lote, mismas semillas, mismos datos y aumentos. Lo único '
         'que cambia es la arquitectura y su receta de optimización, la de sus artículos. La época se '
         'elige con la validación limpia; la prueba se usa una sola vez. El margen sale de remuestrear '
         'lances completos, porque las fotos de un mismo lance no son independientes.')

# 11 ---------------------------------------------------------------- modelos: resultado
s = nueva()
titulo(s, 'Resultado', 'Los dos modelos modernos son los mejores')
figura(s, 'g_modelos.png', 120, 280, w=1240)
cifra(s, 1420, 330, 380, num(M['ConvNeXt-T']['macro_f1']), 'ConvNeXt-T (CNN)', col='E8913A', vsize=80, esize=26)
cifra(s, 1420, 560, 380, num(M['Swin-T']['macro_f1']), 'Swin-T (transformer)', col='2F7FC1', vsize=80, esize=26)
texto(s, 1420, 790, 380, 120, 'Empatan: no se puede elegir uno con estos datos.', 24, MUTED, line=1.3)
pie(s, 'macro-F1 en la prueba limpia de FishNet · promedio de 3 semillas · línea negra = margen al 95 %')
notas(s, 'Los cinco modelos en la prueba limpia. Arriba quedan los dos modernos, ConvNeXt-T (una CNN '
         'de 2022) con %s y Swin-T (un transformer de 2021) con %s: entre ellos no hay diferencia. '
         'Abajo, el ResNet-50 (%s) y DeiT-S (%s), y en el medio el ViT-B/16 (%s). Ojo: macro-F1 no es '
         'porcentaje de aciertos; al azar daría alrededor de 0,05.'
      % tuple(num(M[m]['macro_f1']) for m in ('ConvNeXt-T', 'Swin-T', 'ResNet-50', 'DeiT-S', 'ViT-B/16')))

# 12 ---------------------------------------------------------------- pares
s = nueva()
titulo(s, 'Resultado · CNN contra transformer', 'A igual tamaño y época, CNN y transformer empatan')
figura(s, 'g_pares.png', 120, 280, w=1240)
texto(s, 1420, 330, 380, 600,
      [[('Arriba: ', True, INK), ('a igual tamaño o época, la diferencia es casi cero.', False, MUTED)],
       [('Abajo: ', True, INK), ('la CNN moderna sí supera a la antigua.', False, MUTED)],
       [('Verde: ', True, OK), ('se distingue de cero.', False, MUTED)]],
      26, line=1.35, space_after=26)
pie(s)
notas(s, 'Cada línea es una diferencia entre dos modelos, con su margen. Si la línea cruza la raya '
         'punteada del cero, no se distingue. Las dos de arriba son la respuesta principal: '
         '%s %s El ViT-B/16 le gana al ResNet-50, pero ese par no es parejo en tamaño ni en época. '
         'Abajo: %s Respuesta a la pregunta 2 del profesor: el tercer modelo muestra que la ventaja del '
         'ViT se explica mejor por su tamaño y modernidad que por la atención. "No se distingue" no '
         'significa que sean iguales: significa que con esta prueba no se puede decir cuál es mejor.'
      % (lectura('DeiT-S - ResNet-50', VT['DeiT-S - ResNet-50']),
         lectura('Swin-T - ConvNeXt-T', VT['Swin-T - ConvNeXt-T']),
         lectura('ConvNeXt-T - ResNet-50', VT['ConvNeXt-T - ResNet-50'])))

# 13 ---------------------------------------------------------------- tamano del pez
MT = INV['margenes']['tamano']
chico, medio = MT['< 80 px']['Swin-T'], MT['130-220 px']['Swin-T']
alb = INV['por_especie_y_tamano'].get('Albacore')
s = nueva()
titulo(s, 'Resultado · dónde gana el moderno', 'La ventaja aparece cuando el pez se ve grande')
figura(s, 'g_tamano.png', 120, 280, w=1240)
cifra(s, 1420, 310, 380, num(chico['delta_medio'] * 100, 0, True).replace(',0', '') + ' pts',
      'de acierto de Swin-T sobre el ResNet-50 con peces de menos de 80 px', col='9AA5AE', vsize=64, esize=23)
cifra(s, 1420, 540, 380, num(medio['delta_medio'] * 100, 0, True).replace(',0', '') + ' pts',
      'con peces de 130 a 220 px', col='2F7FC1', vsize=64, esize=23)
if alb:
    texto(s, 1420, 760, 380, 200,
          [[('Albacora grande: ', True, INK),
            ('%s de acierto con Swin-T, %s con el ResNet-50.'
             % (pct(alb['Swin-T']['grande']), pct(alb['ResNet-50']['grande'])), False, MUTED)]],
          23, line=1.3)
pie(s, 'Acierto en peces de las 15 especies, prueba limpia · pts = puntos porcentuales de acierto')
notas(s, 'Este es el hallazgo central. La ventaja de los modelos modernos no es pareja: con peces chicos '
         'en la foto, de menos de 80 píxeles, todos fallan parecido, porque la imagen no trae el detalle. '
         'Cuando el pez se ve grande, los modernos aprovechan ese detalle y el ResNet no: Swin-T le saca '
         '%s puntos con peces chicos (margen %s a %s) y %s con peces de 130 a 220 px (margen %s a %s). '
         'Pasa también dentro de una misma especie, así que no es que los peces grandes sean otras '
         'especies. El caso más claro es el albacora. Qué significa: elegir un buen modelo solo rinde si '
         'la foto le da suficientes píxeles al pez.'
      % (num(100 * chico['delta_medio'], 0, True), num(100 * chico['ic95'][0], 0, True),
         num(100 * chico['ic95'][1], 0, True), num(100 * medio['delta_medio'], 0, True),
         num(100 * medio['ic95'][0], 0, True), num(100 * medio['ic95'][1], 0, True)))

# 14 ---------------------------------------------------------------- barcos nuevos
cv, cn = PV['camara_vista']['modelos'], PV['camara_nueva']['modelos']
s = nueva()
titulo(s, 'Resultado · barcos nuevos', 'En barcos nuevos todos caen; los modernos, menos')
figura(s, 'g_camaras.png', 120, 280, w=1240)
cifra(s, 1420, 330, 380, pct(1 - cam['frac_cajas_camara_vista']),
      'de la prueba viene de cámaras que no estaban en entrenamiento', col=WARM, vsize=80, esize=25)
texto(s, 1420, 640, 380, 300, 'Para la caleta, todo barco sería nuevo: es la limitación más importante.',
      26, INK, bold=True, line=1.3)
pie(s, 'Punto lleno: cámaras vistas · punto vacío: cámaras nuevas · macro-F1 en la prueba limpia')
notas(s, 'Hallazgo que nos marcó el profesor: el 31 %% de la prueba viene de cámaras que también '
         'están en entrenamiento. Separando, en cámaras nuevas todos bajan: ConvNeXt-T de %s a %s, '
         'ResNet-50 de %s a %s. Los modernos caen menos y siguen arriba. Es la limitación más '
         'importante del trabajo, porque en la caleta toda cámara sería nueva.'
      % (num(cv['ConvNeXt-T']['macro_f1']), num(cn['ConvNeXt-T']['macro_f1']),
         num(cv['ResNet-50']['macro_f1']), num(cn['ResNet-50']['macro_f1'])))

# 14b --------------------------------------------------------------- camara y luz
CA = INV['camaras']['ConvNeXt-T']
LZ = INV['luz']
s = nueva()
titulo(s, 'Resultado · la foto', 'La cámara y la luz pesan más que el modelo')
figura(s, 'g_camaras_acierto.png', 120, 280, w=1000)
figura(s, 'g_luz.png', 1180, 280, w=620)
texto(s, 120, 880, 1680, 100,
      [[('Mismo modelo, otra cámara: ', True, INK),
        ('ConvNeXt-T acierta entre %s y %s según la cámara. ' % (pct(CA['min']), pct(CA['max'])), False, MUTED),
        ('Con poca luz, ', True, INK), ('todos aciertan menos.', False, MUTED)]], 26, line=1.3)
pie(s, 'Acierto en peces de las 15 especies · cámaras con 40 peces o más en la prueba · luz = brillo del recorte en tercios')
notas(s, 'Con el mismo modelo, ConvNeXt-T, el acierto va de %s a %s según la cámara: esa diferencia es '
         'mucho mayor que la que hay entre modelos. Influye el ángulo, la distancia y la luz de cada '
         'barco. Con la luz pasa lo mismo: Swin-T acierta %s en las fotos más oscuras y %s en las más '
         'claras. Para la caleta, cómo y dónde se pone la cámara importa tanto como el modelo. Solo 8 '
         'cámaras tienen suficientes peces para compararlas, así que es una tendencia, no una medición fina.'
      % (pct(CA['min']), pct(CA['max']), pct(LZ['oscuro']['Swin-T']), pct(LZ['claro']['Swin-T'])))

# 15 ---------------------------------------------------------------- especies
pe = A['por_especie']
s = nueva()
titulo(s, 'Resultado · por especie', 'Las especies comunes se reconocen; las raras, no')
figura(s, 'g_especies.png', 120, 270, h=720)
texto(s, 1000, 320, 800, 600,
      [[('Bien: ', True, OK), ('Mahi mahi, Opah y Skipjack tuna, sobre 0,75 en todos los modelos.', False, MUTED)],
       [('Mal: ', True, WARM), ('Blue marlin, Bigeye tuna y Shortbill spearfish, cerca de 0 en todos.', False, MUTED)],
       [('Entre medio: ', True, INK), ('los atunes y los peces de pico, que se parecen entre sí.', False, MUTED)]],
      28, line=1.35, space_after=28)
pie(s, 'F1 por especie (0 a 1) en la prueba limpia · entre paréntesis, cajas de prueba · más oscuro = mejor')
notas(s, 'Cada fila es una especie, de la más común a la más rara; cada columna, un modelo. Lo que '
         'más explica el resultado no es el modelo, sino la especie: las fáciles se reconocen bien en '
         'todos, y las raras, con pocas cajas de prueba, quedan cerca de cero en todos. Esas especies '
         'raras son las que bajan el macro-F1.')

# 16 ---------------------------------------------------------------- errores compartidos
ER = INV['errores']
tf = ER['todos_fallan_por_especie']
n_tf = sum(tf.values())
s = nueva()
titulo(s, 'Resultado · lo que ningún modelo resuelve', 'Uno de cada cuatro peces falla en los cinco modelos')
figura(s, 'g_errores.png', 120, 280, w=1680)
figura(s, 'g_atunes.png', 120, 550, h=430)
texto(s, 860, 580, 940, 380,
      [[('Un tercio de esos errores ', True, INK),
        ('es una sola especie: el marlín rayado (%d de %d), que se confunde con los otros peces de pico.'
         % (tf.get('Striped marlin', 0), n_tf), False, MUTED)],
       [('Los atunes ', True, INK),
        ('se confunden entre sí (a la izquierda, ConvNeXt-T): el aleta amarilla pasa por albacora en casi '
         'un tercio de los casos.', False, MUTED)],
       [('Cambiar de modelo no lo arregla: ', True, WARM),
        ('hacen falta más ejemplos de esos grupos.', False, INK)]],
      25, line=1.3, space_after=18)
pie(s, 'Peces de las 15 especies en la prueba limpia, semilla 0 de cada modelo · matriz: porcentaje de cada fila')
notas(s, 'En el %s de los peces fallan los cinco modelos a la vez, y en el %s aciertan los cinco. Los '
         'errores son en gran parte los mismos para todos los diseños. Se concentran en dos grupos de '
         'especies parecidas: los peces de pico (el marlín rayado solo es un tercio de esos errores) y '
         'los atunes. Como ningún diseño los resuelve, el camino no es otro modelo sino más ejemplos de '
         'esos grupos, o reportarlos a nivel de grupo («atún», «pez de pico»).'
      % (pct(ER['todos_fallan']), pct(ER['todos_aciertan'])))

# 17 ---------------------------------------------------------------- dónde mira
ef = PG_F['energia_en_la_caja']
s = nueva()
titulo(s, 'Interpretabilidad · ResNet-50 y ViT-B/16', 'Los dos miran el pez en una medida parecida')
foto(s, 'mapas_3.png', 120, 290, h=660)
for j, t in enumerate(['Recorte', 'ResNet · Grad-CAM', 'ViT · Chefer']):
    texto(s, 120 + j * 245, 258, 245, 30, t, 20, MUTED, bold=True, align='c')
x0 = 1000
texto(s, x0, 300, 800, 50, 'Atención en el pez, por sobre el azar', 30, INK, bold=True)
for i, (k, et, col) in enumerate([('gradcam', 'ResNet-50 · Grad-CAM', RESN),
                                   ('chefer', 'ViT-B/16 · Chefer', VITC),
                                   ('rollout', 'ViT-B/16 · rollout', '8A989B')]):
    y = 380 + i * 150
    texto(s, x0, y, 500, 50, et, 28, INK)
    texto(s, x0 + 520, y - 12, 280, 80, num(ef[k]['sobre_barajado'], 3, True), 56, col, bold=True, align='r')
texto(s, x0, 840, 800, 120, '0 = no sigue al pez. Medido con el pez en una posición al azar.', 24, MUTED, line=1.35)
pie(s, 'FishNet, %d recortes de la prueba limpia · caja blanca = pez' % PG_F['n'])
notas(s, 'Calculamos los mapas sobre los mismos recortes de FishNet. Con el pez siempre al centro, un '
         'mapa fijo que solo marca el centro ya "acertaba" más que cualquier modelo; por eso pusimos el '
         'pez en una posición al azar y restamos la atención que cae en la caja de otro recorte. '
         'Resultado: Grad-CAM y Chefer siguen al pez casi igual (%s y %s). El ViT no mira menos el '
         'fondo que el ResNet. Ejemplos: arriba los dos sobre el pez; al medio el ResNet mira el brazo; '
         'abajo el ViT se dispersa. Solo se midió para este par.'
      % (num(ef['gradcam']['sobre_barajado'], 3, True), num(ef['chefer']['sobre_barajado'], 3, True)))

# 18 ---------------------------------------------------------------- no sé
ch_r, ch_v = CHILE['modelos']['ResNet-50'], CHILE['modelos']['ViT-B/16']
s = nueva()
titulo(s, 'Decir «no sé» · ResNet-50 y ViT-B/16', 'El ViT reconoce mejor lo que nunca vio')
for i, esp in enumerate(['reineta', 'merluza', 'congrio']):
    foto(s, 'chile_%s.jpg' % esp, 120 + i * 270, 300, w=250, h=250)
    texto(s, 120 + i * 270, 560, 250, 40, esp.capitalize(), 24, INK, bold=True, align='c')
figura(s, 'g_nose.png', 960, 280, w=840)
caja(s, 120, 640, 780, 320, WARMBG, None)
texto(s, 160, 670, 700, 270,
      [[('Lo correcto es decir «no sé»: ', True, INK), ('ninguna de estas especies está en los datos.', False, MUTED)],
       [('Ninguno está listo: ', True, WARM), ('cuando aceptan, casi siempre dicen «atún».', False, INK)]],
      26, line=1.35, space_after=16)
pie(s, 'Fotos de iNaturalist, CC BY-NC: jamillam, Benjamín Ignacio, Tomás Vargas Lavín · umbral elegido en validación')
notas(s, 'Probamos el "no sé" con 45 fotos de reineta, merluza y congrio de iNaturalist, y con los '
         'peces de Kaggle. El criterio: si la energía de la salida es más baja que la del 95 %% de los '
         'peces de validación, responde "no sé". El ViT lo dice más seguido: %s contra %s con las '
         'especies chilenas. Separación (AUROC): %s contra %s. Limitación: son fotos de otro tipo '
         '(peces en la mano o en una mesa); es una prueba cualitativa.'
      % (pct(ch_v['rechaza_total']), pct(ch_r['rechaza_total']),
         num(ch_v['auroc_vs_test_limpio_sin_personas']), num(ch_r['auroc_vs_test_limpio_sin_personas'])))

# 18b --------------------------------------------------------------- derivar
DV = INV['derivar']['Swin-T']
s = nueva()
titulo(s, 'Cómo usarlo hoy', 'Que responda solo cuando está seguro y derive el resto')
figura(s, 'g_derivar.png', 120, 280, w=1240)
cifra(s, 1420, 320, 380, pct(DV['60 %']), 'de acierto si Swin-T responde solo 6 de cada 10 peces', vsize=80, esize=24)
cifra(s, 1420, 580, 380, pct(DV['40 %']), 'si responde 4 de cada 10', vsize=80, esize=24)
texto(s, 1420, 820, 380, 140, 'El resto va al inspector, que hoy revisa todo.', 24, INK, bold=True, line=1.3)
pie(s, 'Peces de la prueba limpia, ordenados por la confianza del modelo · promedio de 3 semillas')
notas(s, 'Con un macro-F1 de 0,42 el sistema no puede reemplazar al inspector, pero sí repartir el '
         'trabajo. Si responde solo los peces en los que más confía, acierta mucho más: con Swin-T, el '
         '%s en los 6 de cada 10 peces más seguros y el %s en los 4 de cada 10 más seguros. El resto lo '
         'revisa el inspector, igual que hoy. Respondiendo todo, acierta el %s. Es la forma honesta de '
         'usar estos resultados hoy.' % (pct(DV['60 %']), pct(DV['40 %']), pct(DV['100 %'])))

# 18c --------------------------------------------------------------- recomendaciones
s = nueva(DARK)
titulo(s, 'Recomendaciones para la Caleta Portales', ESLOGAN, oscuro=True)
for i, (t, d) in enumerate([
        ('Una buena foto', 'Cámara fija cerca de donde se ordena el pescado, para que cada pez ocupe más '
                           'de 130 píxeles, y con buena luz.'),
        ('Un modelo moderno', 'ConvNeXt-T o Swin-T: rinden más y cuestan en cómputo lo mismo que el ResNet-50. '
                              'Combinados suben un poco más.'),
        ('Responder solo lo seguro', 'El sistema propone en los peces que reconoce con confianza y deriva '
                                     'el resto al inspector.')]):
    x = 120 + i * 570
    caja(s, x, 340, 530, 520, '16304F', None)
    texto(s, x + 44, 380, 450, 50, '%d' % (i + 1), 44, ACCL, bold=True)
    texto(s, x + 44, 450, 450, 60, t, 34, ONDARK, bold=True)
    texto(s, x + 44, 530, 450, 320, d, 27, DMUTED, line=1.4)
pie(s, oscuro=True)
notas(s, 'Lo que se desprende de los resultados, en orden de impacto. Primero, la foto: el modelo '
         'moderno solo marca diferencia cuando el pez se ve grande, y la cámara y la luz pesan más que el '
         'modelo. Segundo, un modelo moderno de cualquiera de las dos familias; combinar ConvNeXt-T y '
         'Swin-T da %s de macro-F1 (+%s sobre Swin-T solo, margen %s a %s). Tercero, usarlo como apoyo: '
         'que responda solo cuando está seguro. Además, para los atunes y los peces de pico hacen falta '
         'más ejemplos.'
      % (num(INV['combinar']['ConvNeXt-T + Swin-T']['macro_f1']),
         num(INV['margenes']['ensamble_vs_swin']['delta_medio']),
         num(INV['margenes']['ensamble_vs_swin']['ic95'][0], 3, True),
         num(INV['margenes']['ensamble_vs_swin']['ic95'][1], 3, True)))

# 20 ---------------------------------------------------------------- conclusiones
d_cr = VT['ConvNeXt-T - ResNet-50']
s = nueva(DARK)
titulo(s, 'Conclusiones', 'Qué podemos afirmar', oscuro=True)
conc = [
    ('A igual tamaño y época, CNN y transformer no se distinguen (DeiT-S %s frente al ResNet-50; Swin-T '
     '%s frente a ConvNeXt-T).'
     % (num(VT['DeiT-S - ResNet-50']['delta_medio'], 3, True), num(VT['Swin-T - ConvNeXt-T']['delta_medio'], 3, True))),
    ('Los modelos de diseño moderno rinden más (ConvNeXt-T %s sobre el ResNet-50), pero esa ventaja aparece '
     'sobre todo cuando el pez se ve grande en la foto.' % num(d_cr['delta_medio'], 3, True)),
    ('Con un buen modelo, lo que más limita es la foto: el tamaño del pez, la cámara y la luz.'),
    ('Uno de cada cuatro peces falla en todos los modelos: atunes y peces de pico necesitan más ejemplos.'),
    ('Hoy sirve como apoyo: responder solo lo seguro (%s de acierto en 6 de cada 10 peces) y derivar el resto.'
     % pct(INV['derivar']['Swin-T']['60 %'])),
]
for i, c in enumerate(conc):
    y = 320 + i * 124
    texto(s, 120, y, 80, 80, str(i + 1), 46, ACCL, bold=True)
    texto(s, 220, y + 6, 1560, 120, c, 29, ONDARK, line=1.25)
pie(s, oscuro=True)
notas(s, 'Cada afirmación tiene su evidencia en una lámina anterior. La respuesta a la pregunta del '
         'título es la primera: CNN o transformer da lo mismo a igual tamaño y época. Lo que aprendimos '
         'y no esperábamos es la segunda y la tercera: con un modelo moderno, el límite pasa a ser la foto.')

# 21 ---------------------------------------------------------------- próximos pasos
s = nueva()
titulo(s, 'Próximos pasos', 'Qué haría falta para usarlo en la caleta')
for i, (t, d) in enumerate([
        ('Contar', 'Un detector entrenado con las cajas de FishNet, para encontrar y contar los peces '
                   'antes de clasificarlos.'),
        ('Fotos locales', 'Fotos de la Caleta Portales con reineta, merluza y congrio, tomadas con la '
                          'cámara cerca y con buena luz.'),
        ('Barcos nuevos', 'Entrenar y evaluar con un reparto por cámara, para medir el rendimiento en '
                          'un barco que el modelo nunca vio.')]):
    x = 120 + i * 570
    caja(s, x, 320, 530, 480)
    texto(s, x + 44, 360, 450, 60, t, 40, ACC, bold=True)
    texto(s, x + 44, 440, 450, 340, d, 29, INK, line=1.4)
pie(s)
notas(s, 'El objetivo original incluía contar peces amontonados; no lo hicimos. El camino es un '
         'detector ajustado con las cajas de FishNet y después nuestro clasificador sobre cada caja. '
         'Además hacen falta datos locales y medir con barcos nuevos, que es donde los modelos más caen.')

# 22 ---------------------------------------------------------------- referencias
s = nueva()
titulo(s, 'Referencias', '')
refs = [
    'He, K. et al. (2016). Deep residual learning for image recognition. CVPR.',
    'Dosovitskiy, A. et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.',
    'Touvron, H. et al. (2021). Training data-efficient image transformers & distillation through attention. ICML.',
    'Liu, Z. et al. (2021). Swin Transformer: Hierarchical vision transformer using shifted windows. ICCV.',
    'Liu, Z. et al. (2022). A ConvNet for the 2020s. CVPR.',
    'Ulucan, O., Karakaya, D. y Turkan, M. (2020). A large-scale dataset for fish segmentation and classification. ASYU.',
    'Kay, J. y Merrifield, M. (2021). The Fishnet Open Images Database. arXiv:2106.09178.',
    'Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.',
    'Abnar, S. y Zuidema, W. (2020). Quantifying attention flow in transformers. ACL.',
    'Chefer, H., Gur, S. y Wolf, L. (2021). Transformer interpretability beyond attention visualization. CVPR.',
    'Menon, A. K. et al. (2021). Long-tail learning via logit adjustment. ICLR.',
    'Liu, W. et al. (2020). Energy-based out-of-distribution detection. NeurIPS.',
    'Kapoor, S. y Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns.',
]
texto(s, 120, 230, 1680, 760, [[r] for r in refs], 22, INK, line=1.25, space_after=9)
pie(s)
notas(s, 'Referencias de los modelos, los datos y los métodos usados.')

# 23 ---------------------------------------------------------------- cierre
s = nueva(DARK)
imagen(s, LOGO, 1180, 200, h=620)
texto(s, 120, 330, 1000, 160, 'Gracias', 120, ONDARK, bold=True)
texto(s, 120, 500, 1000, 140, ESLOGAN, 36, ACCL, line=1.3)
texto(s, 120, 700, 1000, 60, '¿Preguntas?', 40, DMUTED)
texto(s, 120, 880, 1000, 50, 'github.com/Nirvesta04/CaletaScan', 30, ACCL, bold=True)
notas(s, 'Cerrar con el eslogan e invitar a preguntas. El repositorio tiene el código, los '
         'resultados, la data card y el informe.')

# ================================================================ ANEXOS
# A0 --------------------------------------------------------------- riesgos y auditoría
s = nueva()
titulo(s, 'Anexo · riesgos y auditoría', 'Cinco problemas de los datos, medidos y corregidos')
filas = [['Problema', 'Medición', 'Qué hicimos'],
         ['Kaggle: ~21 copias por foto', '411 fotos originales en 9.000', 'Reparto por foto original'],
         ['Kaggle: el fondo delata la especie', '%s sin ver el pez' % num(k_f['macro_f1_b']), 'Kaggle no decide'],
         ['FishNet: fuga por lance', '%s de la prueba oficial' % pct(fg['frac_cajas'], 1), 'Prueba limpia'],
         ['Fotos de un lance no son independientes', 'Margen hasta 4,6 veces más angosto', 'Remuestreo por lance'],
         ['FishNet: barcos repetidos', '%s de la prueba limpia' % pct(cam['frac_cajas_camara_vista']), 'Se reporta aparte']]
tabla(s, 120, 300, 1680, filas, [3.4, 2.8, 2.4], alto_fila=88, size=26)
texto(s, 120, 860, 1680, 100, 'Auditoría: ningún lance ni foto original cruza de entrenamiento a prueba '
      'en los repartos que usamos.', 26, MUTED, line=1.35)
pie(s)
notas(s, 'Resumen de la auditoría, que en el Avance ocupaba cuatro láminas. Lo nuevo es la última '
         'fila: el reparto por lance limpia los casi duplicados, pero no evalúa barcos nuevos; lo '
         'reportamos aparte.')

# A1 --------------------------------------------------------------- épocas
s = nueva()
titulo(s, 'Anexo · entrenamiento', 'Macro-F1 de validación por época')
cd = CategoryChartData()
cd.categories = [str(i) for i in range(1, 9)]
cols = {'ResNet-50': 'B3541E', 'ViT-B/16': '1B3F73', 'DeiT-S': '7FA7D6', 'ConvNeXt-T': 'E8913A', 'Swin-T': '2F7FC1'}
series = []
for m, por_semilla in A['epocas'].items():
    if not por_semilla:
        continue
    ep = [[e['val_macro_f1'] for e in h] for h in por_semilla.values()]
    cd.add_series(m, [round(sum(c) / len(c), 3) for c in zip(*ep)])
    series.append(m)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, E(120), E(280), E(1100), E(660), cd)
ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = P(22)
for ser, m in zip(ch.plots[0].series, series):
    ser.format.line.color.rgb = C(cols[m]); ser.format.line.width = Pt(2.5)
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = C(cols[m])
    ser.smooth = False
ch.value_axis.tick_labels.font.size = P(20)
ch.category_axis.tick_labels.font.size = P(20)
ch.value_axis.major_gridlines.format.line.color.rgb = C('E4E6EA')
filas = [['Modelo', 'Mejor época', 'Minutos']]
for m, por_semilla in A['epocas'].items():
    ep = [[e['val_macro_f1'] for e in h] for h in por_semilla.values()]
    medias = [sum(c) / len(c) for c in zip(*ep)]
    mins = [sum(e['min'] for e in h) for h in por_semilla.values()]
    filas.append([m, str(medias.index(max(medias)) + 1), '%.0f' % (sum(mins) / len(mins))])
tabla(s, 1280, 300, 520, filas, [2.2, 1.7, 1.5], alto_fila=70, size=22)
texto(s, 1280, 760, 520, 160, 'Promedio de 3 semillas. Minutos por semilla, con la lectura de imágenes.', 22, MUTED, line=1.35)
pie(s)
notas(s, 'Validación limpia por época. El ResNet-50 se estanca temprano; los demás siguen mejorando '
         'un poco más. El tiempo incluye la lectura de imágenes, que en este equipo es el cuello de botella.')

# A2 --------------------------------------------------------------- ViT contra ResNet
s = nueva()
titulo(s, 'Anexo · ViT-B/16 contra ResNet-50', 'La ventaja del ViT se achica en pruebas exigentes')
figura(s, 'diferencias.png', 120, 300, w=1060)
texto(s, 1240, 330, 560, 600,
      [[('Atunes como una clase: ', True, INK), ('parte de la ventaja era separar un atún de otro.', False, MUTED)],
       [('Barcos no vistos: ', True, INK), ('ahí los dos bajan y dejan de distinguirse.', False, MUTED)]],
      27, line=1.35, space_after=22)
pie(s)
notas(s, 'Detalle del par original. Con los atunes agrupados, la ventaja del ViT baja de +0,042 a '
         '+0,031 y deja de distinguirse; en barcos no vistos también.')

# A3 --------------------------------------------------------------- matrices
s = nueva()
titulo(s, 'Anexo · confusión', 'Qué especie predice cada modelo, por especie real')
figura(s, 'confusion_e4.png', 120, 270, h=700)
figura(s, 'confusion_e11.png', 980, 270, h=700)
pie(s, 'ResNet-50 (izquierda) y ConvNeXt-T (derecha) · porcentaje de cada fila · prueba limpia')
notas(s, 'Matriz de confusión entre especies del modelo antiguo y del mejor moderno. Cada fila suma '
         '100 %. La confusión principal es entre atunes.')

# A4 --------------------------------------------------------------- auditoría
s = nueva()
titulo(s, 'Anexo · auditoría', 'Lo que revisamos en nuestros propios resultados')
kd = KAG['comparacion']
filas = [['Prueba', 'Resultado'],
         ['Lances o fotos originales que cruzan de entrenamiento a prueba', '0'],
         ['Casi duplicados por píxeles, prueba oficial / limpia', '38,1 % / 2,6 %'],
         ['Vecino más cercano sobre miniaturas 32×32 (sin aprendizaje)', 'Kaggle 0,829 · FishNet 0,559'],
         ['Kaggle: ViT-B/16 − ResNet-50', '%s (margen %s)' % (num(kd['delta_medio'], 3, True), ic(kd['ic95']))],
         ['Sobreajuste en FishNet: entrenamiento → prueba', 'ResNet 0,783 → 0,340 · ViT 0,925 → 0,370'],
         ['Tiempo por pez en CPU (i5-12400F)', 'ResNet 43 ms · ViT-B/16 119 ms']]
tabla(s, 120, 290, 1680, filas, [5.2, 3.6], alto_fila=88, size=25)
pie(s)
notas(s, 'Detalle de la auditoría. El vecino más cercano muestra por qué Kaggle es trivial: sin ninguna '
         'red, comparar miniaturas acierta 0,829.')

os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
prs.save(SALIDA)
print('guardado: %s (%d láminas)' % (SALIDA, N[0]))
