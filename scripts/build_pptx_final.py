# -*- coding: utf-8 -*-
"""
Mazo de la presentacion final de Caleta-Scan (G3).

    python -m scripts.figuras_mazo          # fotos de los datos (no versionadas)
    python -m scripts.analisis_final        # cifras
    python -m scripts.build_pptx_final      # -> entrega/Caleta-Scan-Final-G3.pptx

Todas las cifras se leen de runs/*.json: nada se copia a mano. El PPTX lleva
fotos de Kaggle, FishNet e iNaturalist, por eso se escribe en entrega/, que no
se versiona.

Orden pedido por el profesor tras el Avance: problema, datos con imagenes,
modelos y protocolo, resultados, riesgos y auditoria resumida, proximos pasos;
el detalle va a anexos. Un mensaje por lamina; la explicacion va en las notas.
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

from scripts.lecturas import ETIQUETA, lectura

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')
PRIV = os.path.join(ROOT, 'docs', 'figs_privadas')
FIGS = os.path.join(ROOT, 'docs', 'figs')
SALIDA = os.path.join(ROOT, 'entrega', 'Caleta-Scan-Final-G3.pptx')


def J(nombre):
    with open(os.path.join(RUNS, nombre), encoding='utf-8') as fh:
        return json.load(fh)


A = J('analisis_final.json')
PG_F = J('pointing_game_fishnet.json')
PG_K = J('pointing_game.json')
CHILE = J('chile_ood.json')
CAL = J('calibracion_ood.json')
E7 = J('comparacion_e7.json')
KAG = J('comparacion_e1_e2.json')
VEL = J('velocidad.json')


def ood(run, sub):
    k = [k for k in CAL['ood'] if k.startswith(run) and k.endswith(sub)][0]
    return CAL['ood'][k]['energia']


def num(x, d=3, signo=False):
    s = ('%+.*f' if signo else '%.*f') % (d, x)
    return s.replace('.', ',')


def pct(x, d=0):
    return ('%.*f' % (d, 100 * x)).replace('.', ',') + ' %'


def ic(v, signo=False):
    return '%s a %s' % (num(v[0], 3, signo), num(v[1], 3, signo))


# ---------------------------------------------------------------- estilo
PX = 6350
def E(px):  return Emu(int(round(px * PX)))
def P(px):  return Pt(px / 2.0)
def C(h):   return RGBColor.from_string(h)

DARK, LIGHT, WHITE, BORDER = '0F2226', 'F7F6F2', 'FFFFFF', 'DEDCD3'
TEAL, TEALL, WARM, WARMBG = '0C6F66', '5CC0B0', 'B3541E', 'F6E9DD'
INK, MUTED, FOOT = '16282C', '56676B', '7A888B'
ONDARK, DMUTED = 'F0EFE9', 'A9BCBD'
RESN, VITC = 'B3541E', '0C6F66'          # colores fijos: ResNet calido, ViT verde
FONT = 'Segoe UI'

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


def rect(s, x, y, w, h, fill):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, E(x), E(y), E(w), E(h))
    b.fill.solid(); b.fill.fore_color.rgb = C(fill)
    b.line.fill.background(); b.shadow.inherit = False
    return b


def titulo(s, ojo, t, sub=None, oscuro=False):
    texto(s, 120, 96, 1680, 40, ojo.upper(), 24, TEALL if oscuro else TEAL, bold=True)
    texto(s, 120, 146, 1680, 140, t, 62, ONDARK if oscuro else INK, bold=True, line=1.08)
    if sub:
        texto(s, 120, 300, 1680, 80, sub, 32, DMUTED if oscuro else MUTED, line=1.3)


def pie(s, izq='', oscuro=False):
    col = DMUTED if oscuro else FOOT
    texto(s, 120, 1006, 1560, 40, izq or 'Caleta-Scan · G3', 20, col)
    texto(s, 1680, 1006, 120, 40, str(N[0]), 22, col, align='r')


def notas(s, t):
    s.notes_slide.notes_text_frame.text = t


def imagen(s, ruta, x, y, w=None, h=None):
    if not os.path.exists(ruta):
        caja(s, x, y, w or 400, h or 300, 'EEEEEE', 'CCCCCC')
        texto(s, x + 20, y + 20, (w or 400) - 40, 60, 'falta: ' + os.path.basename(ruta), 20, WARM)
        return None
    return s.shapes.add_picture(ruta, E(x), E(y), E(w) if w else None, E(h) if h else None)


def cifra(s, x, y, w, valor, etiqueta, col=TEAL, vsize=96, esize=28, ecol=MUTED):
    texto(s, x, y, w, 120, valor, vsize, col, bold=True, line=1.0)
    texto(s, x, y + vsize * 1.25, w, 140, etiqueta, esize, ecol, line=1.3)


def tabla(s, x, y, w, filas, anchos, alto_fila=64, size=26, cab_fill=DARK, resaltar=None):
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
            if i == 0:
                c.fill.fore_color.rgb = C(cab_fill)
            elif resaltar and i in resaltar:
                c.fill.fore_color.rgb = C('E3F1EE')
            else:
                c.fill.fore_color.rgb = C(WHITE if i % 2 else 'F1F0EA')
    return t


# ---------------------------------------------------------------- figuras de datos
def figuras():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIGS, exist_ok=True)
    plt.rcParams['font.family'] = 'Segoe UI'

    def bosque(filas, ruta):
        """Una fila por diferencia, con su margen al 95 %."""
        fig, ax = plt.subplots(figsize=(8, 1.2 + 1.0 * len(filas)), dpi=200)
        for i, (nombre, d) in enumerate(filas[::-1]):
            col = '#0C6F66' if d['distinguible'] else '#8A989B'
            ax.plot(d['ic95'], [i, i], color=col, lw=4, solid_capstyle='round')
            ax.plot([d['delta_medio']], [i], 'o', color=col, ms=11)
            ax.text(max(d['ic95'][1], 0.0) + 0.006, i, num(d['delta_medio'], 3, True),
                    va='center', fontsize=13, color=col, fontweight='bold')
        ax.axvline(0, color='#B3541E', lw=1.5, ls='--')
        ax.set_yticks(range(len(filas)), [n for n, _ in filas[::-1]], fontsize=12)
        ax.set_xlabel('diferencia de macro-F1 (transformer − CNN), con su margen al 95 %', fontsize=12)
        for lado in ('top', 'right', 'left'):
            ax.spines[lado].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(min(-0.06, min(d['ic95'][0] for _, d in filas) - 0.01),
                    max(0.12, max(d['ic95'][1] for _, d in filas) + 0.04))
        fig.tight_layout()
        fig.savefig(ruta, transparent=True)
        plt.close(fig)

    # lamina 10: solo ViT-B/16 contra ResNet-50, en las tres pruebas
    filas = []
    for v, etiqueta in (('todas', 'prueba limpia completa'), ('atunes_agrupados', 'atunes como una clase'),
                        ('camara_nueva', 'solo barcos no vistos')):
        d = A['por_variante'][v]['pares'].get('ViT-B/16 - ResNet-50')
        if d:
            filas.append(('ViT-B/16 − ResNet-50' + chr(10) + '(%s)' % etiqueta, d))
    bosque(filas, os.path.join(FIGS, 'diferencias.png'))


# ================================================================ LAMINAS
NOM = 'Patricio Toledo · Andrés Saavedra · Sebastián Casas · Bastián Molina'
M = A['modelos']
VT = A['por_variante']['todas']
P_VR = VT['pares'].get('ViT-B/16 - ResNet-50')

figuras()

# 1 ----------------------------------------------------------------- portada
s = nueva(DARK)
texto(s, 120, 120, 1680, 40, 'PUCV · APRENDIZAJE AUTOMÁTICO II · PRESENTACIÓN FINAL', 24, TEALL, bold=True)
texto(s, 120, 360, 1680, 200, 'Caleta-Scan', 150, ONDARK, bold=True, line=1.0)
texto(s, 120, 560, 1400, 140, 'Clasificar especies de pescado en fotos de faena:\nCNN contra transformer, con una comparación justa',
      40, DMUTED, line=1.3)
texto(s, 120, 900, 1600, 50, 'Grupo G3 · ' + NOM, 28, ONDARK, bold=True)
notas(s, 'Somos el grupo G3. Caleta-Scan clasifica la especie de un pez en una foto de faena y '
         'compara dos familias de modelos: redes convolucionales (CNN) y transformers. '
         'En 15 minutos: el problema, los datos, cómo comparamos, qué encontramos y qué falta.')

# 2 ----------------------------------------------------------------- problema
s = nueva()
titulo(s, 'Problema', 'En la Caleta Portales, el desembarque se registra a mano',
       'Un inspector de Sernapesca anota especie y cantidad durante la descarga.')
for i, (t, d) in enumerate([('Lento', 'Se anota pez por pez mientras se descarga.'),
                             ('Variable', 'Depende de quién esté de turno.'),
                             ('Insuficiente', 'En temporada alta llegan más botes que inspectores.')]):
    x = 120 + i * 570
    caja(s, x, 460, 530, 300)
    texto(s, x + 44, 500, 450, 60, t, 44, TEAL, bold=True)
    texto(s, x + 44, 580, 450, 160, d, 30, MUTED, line=1.35)
texto(s, 120, 830, 1680, 80, [('Objetivo: ', True, INK),
                             ('proponer la especie a partir de una foto, para que la persona revise y corrija.', False, INK)],
      32)
pie(s)
notas(s, 'El registro manual es lento, depende de quién esté y no escala en temporada alta. '
         'La idea es apoyar, no reemplazar: el sistema propone una especie con su confianza, o '
         'dice "no sé", y la persona decide.')

# 3 ----------------------------------------------------------------- setup
s = nueva()
titulo(s, 'Qué hace el sistema', 'Recibe un pez marcado y responde su especie o «no sé»')
pasos = [('Entrada', 'Recorte de un pez en la foto. La caja la marca una persona o viene anotada.'),
         ('Modelo', 'Una CNN o un transformer, preentrenados en ImageNet y ajustados a peces.'),
         ('Salida', 'La especie más probable, o «no sé» si la imagen no se parece a lo aprendido.')]
for i, (t, d) in enumerate(pasos):
    x = 120 + i * 580
    caja(s, x, 330, 520, 330)
    texto(s, x + 40, 366, 440, 50, '%d · %s' % (i + 1, t), 36, TEAL, bold=True)
    texto(s, x + 40, 436, 440, 210, d, 29, INK, line=1.35)
    if i < 2:
        texto(s, x + 524, 450, 56, 80, '→', 54, FOOT, align='c')
caja(s, 120, 720, 1680, 200, WARMBG, None)
texto(s, 170, 752, 1580, 150,
      [[('No detecta peces ni los cuenta. ', True, WARM),
        ('Encontrar y contar los peces en la foto completa queda como próximo paso.', False, INK)],
       [('No conoce especies chilenas. ', True, WARM),
        ('Reineta, merluza y congrio no aparecen en ningún conjunto público.', False, INK)]],
      30, line=1.35, space_after=10)
pie(s)
notas(s, 'Lámina de setup que pidió el profesor. La tarea es de clasificación: el pez ya viene '
         'marcado con una caja. No detectamos ni contamos; lo discutimos al final como próximo paso. '
         'Y no hay especies chilenas en los datos públicos, así que lo que mostramos es el método; '
         'al final probamos qué hacen los modelos con fotos de especies chilenas.')

# 4 ----------------------------------------------------------------- modelos
s = nueva()
titulo(s, 'Modelos', 'Tres pares de modelos con pesos de partida parejos')
filas = [['Par', 'Modelo', 'Tipo', 'Pesos de partida', 'ImageNet', 'Parámetros'],
         ['Original', 'ResNet-50', 'CNN', 'torchvision IMAGENET1K_V2', '80,9 %', '26 M'],
         ['', 'ViT-B/16', 'Transformer', 'torchvision IMAGENET1K_V1', '81,1 %', '87 M'],
         ['Mismo tamaño', 'ResNet-50', 'CNN', '(el mismo de arriba)', '80,9 %', '26 M'],
         ['', 'DeiT-S', 'Transformer', 'Facebook, deit_small (sin destilar)', '79,9 %', '22 M'],
         ['Moderno', 'ConvNeXt-T', 'CNN', 'torchvision IMAGENET1K_V1', '82,5 %', '29 M'],
         ['', 'Swin-T', 'Transformer', 'torchvision IMAGENET1K_V1', '81,5 %', '28 M']]
tabla(s, 120, 330, 1680, filas, [2.2, 2.2, 2.0, 4.6, 1.6, 1.7], alto_fila=72, size=26)
texto(s, 120, 870, 1680, 100,
      'ImageNet: acierto de los pesos de partida en ImageNet-1k, según quien los publica. '
      'Dentro de cada par, la diferencia es de un punto o menos.', 26, MUTED, line=1.35)
pie(s)
notas(s, 'Respuesta a la pregunta 1 del profesor: qué modelos y con qué pesos exactos. '
         'Par original: ResNet-50 con pesos V2 contra ViT-B/16, porque tienen casi el mismo acierto en '
         'ImageNet. Si usáramos los pesos por defecto del ResNet (76,1 %), compararíamos recetas de '
         'entrenamiento y no arquitecturas. '
         'Par mismo tamaño: DeiT-S tiene el tamaño del ResNet-50. Responde si el ViT-B/16 ganaba por '
         'ser transformer o por ser 3,4 veces más grande (pregunta 2 del profesor). '
         'Par moderno: ConvNeXt-T y Swin-T son de 2021-2022, del mismo tamaño y costo. Responde si '
         'el ResNet perdía por ser de 2015.')

# 5 ----------------------------------------------------------------- datos
s = nueva()
titulo(s, 'Datos', 'Kaggle es de estudio; FishNet, de barcos reales')
imagen(s, os.path.join(PRIV, 'kaggle_especies.jpg'), 120, 300, w=800)
imagen(s, os.path.join(PRIV, 'fishnet_ejemplo_3.jpg'), 1000, 300, w=800)
texto(s, 120, 700, 800, 50, 'Kaggle · A Large Scale Fish Dataset', 32, INK, bold=True)
texto(s, 120, 752, 800, 220, [['9.000 fotos de 9 especies, con fondo de estudio y luz pareja.'],
                              ['Son 411 fotos originales: el resto son copias giradas y espejadas.']],
      27, MUTED, line=1.35, space_after=8)
texto(s, 1000, 790, 800, 50, 'FishNet Open Images', 32, INK, bold=True)
texto(s, 1000, 842, 800, 140, '20.002 fotos de cámaras en barcos de pesca, con 99.019 cajas anotadas. '
      '77 cámaras, de día y de noche, con peces amontonados.', 27, MUTED, line=1.35)
pie(s, 'Fotos: Ulucan et al. (2020), CC BY 4.0 · Kay y Merrifield (2021), CC BY-SA 4.0 · caras pixeladas')
notas(s, 'Respuesta a la pregunta 4 del profesor: qué es FishNet. A la izquierda, Kaggle: una foto '
         'por especie, cada una sobre su mesa y con su luz. A la derecha, FishNet: una cubierta real, '
         'con las cajas de los peces dibujadas. Kaggle sirve para diagnosticar; FishNet es donde se '
         'decide la comparación. En FishNet predecimos 20 clases: 19 etiquetas con al menos 100 '
         'ejemplos más una clase "otras especies raras". El macro-F1 se promedia sobre 15 especies: '
         'se excluyen persona, sin pez, desconocido, atún sin especie y otras raras.')

# 6 ----------------------------------------------------------------- secuencia
fg = A['fuga_split_oficial']
s = nueva()
titulo(s, 'Riesgo 1 · fuga', 'Fotos de un mismo lance quedan a ambos lados')
for i in range(4):
    nombre = [f for f in os.listdir(PRIV) if f.startswith('fishnet_seq_%d_' % (i + 1))] if os.path.isdir(PRIV) else []
    x = 120 + i * 425
    if nombre:
        imagen(s, os.path.join(PRIV, nombre[0]), x, 300, w=400)
        es_test = 'test' in nombre[0]
        caja(s, x, 545, 400, 56, WARM if es_test else TEAL, None, radio=10)
        texto(s, x, 545, 400, 56, 'prueba' if es_test else 'entrenamiento', 26, WHITE, bold=True,
              align='c', anchor='m')
cifra(s, 120, 650, 760, pct(fg['frac_cajas'], 1),
      'de las cajas de prueba del reparto oficial comparten lance con el entrenamiento '
      '(%s de las imágenes)' % pct(fg['frac_imagenes'], 1), col=WARM, vsize=88)
texto(s, 1000, 660, 800, 50, 'Lo que hicimos', 32, INK, bold=True)
texto(s, 1000, 715, 800, 240,
      'Sacamos de la prueba todo lance que también está en entrenamiento. Quedan 7.159 cajas de '
      '612 lances. Con el reparto oficial los resultados salían alrededor de 80 % más altos.',
      28, MUTED, line=1.35)
pie(s, 'Cuatro fotos seguidas de un mismo lance de FishNet · caras pixeladas')
notas(s, 'Cuatro fotogramas consecutivos del mismo lance. El reparto oficial puso tres en '
         'entrenamiento y uno en prueba: el modelo se examina sobre una foto casi igual a una que '
         'ya vio. La cifra corregida: 45,8 % de las cajas de prueba comparten lance con '
         'entrenamiento (39,1 % de las imágenes); el 44,5 % del Avance era sobre el archivo completo '
         'de FishNet, no sobre nuestro subconjunto. Todo lo que sigue usa la prueba limpia.')

# 7 ----------------------------------------------------------------- fondo kaggle
k_f = E7['e1_resnet50_kaggle__vs__e7_resnet50_solo_fondo']
k_p = E7['e1_resnet50_kaggle__vs__e7_resnet50_solo_pez']
s = nueva()
titulo(s, 'Riesgo 2 · atajo', 'En Kaggle, el fondo basta para saber la especie')
for i, (f, et, v) in enumerate([('kaggle_original.jpg', 'Foto completa', k_f['macro_f1_a']),
                                ('kaggle_sin_pez.jpg', 'Sin el pez', k_f['macro_f1_b']),
                                ('kaggle_solo_pez.jpg', 'Solo el pez', k_p['macro_f1_b'])]):
    x = 120 + i * 570
    imagen(s, os.path.join(PRIV, f), x, 300, w=530)
    texto(s, x, 720, 530, 50, et, 32, INK, bold=True)
    texto(s, x, 770, 530, 90, num(v), 72, WARM if i == 1 else TEAL, bold=True)
texto(s, 120, 880, 1680, 90,
      'macro-F1 del ResNet-50 en la prueba de Kaggle (1 = todas las especies perfectas). '
      'Por eso Kaggle no decide la comparación: es demasiado fácil.', 27, MUTED, line=1.35)
pie(s, 'Foto: Ulucan et al. (2020), CC BY 4.0 · gris = zona borrada con la silueta anotada')
notas(s, 'Borramos el pez con la silueta que trae Kaggle y reentrenamos. Sin ver el pez, el '
         'ResNet sigue en 0,971: la mesa y la luz delatan la especie. El ViT da lo mismo (0,970). '
         'El reparto por foto original no arregla esto, porque el fondo es de la especie, no de la '
         'foto. Por eso la comparación se decide en FishNet.')

# 8 ----------------------------------------------------------------- protocolo
s = nueva()
titulo(s, 'Protocolo', 'Todos los modelos reciben exactamente lo mismo')
for i, (v, e) in enumerate([('9.104', 'pasos de entrenamiento (8 épocas, lote de 64)'),
                            ('3', 'semillas por modelo'),
                            ('224 px', 'misma resolución y mismos aumentos')]):
    cifra(s, 120 + i * 570, 320, 520, v, e, vsize=84)
caja(s, 120, 620, 1680, 320)
texto(s, 170, 652, 1580, 50, 'Cómo leer los resultados', 32, INK, bold=True)
texto(s, 170, 712, 1580, 220,
      [[('macro-F1: ', True, INK), ('promedio del F1 de las 15 especies, de 0 a 1. Cada especie pesa lo '
                                     'mismo, sea común o rara.', False, MUTED)],
       [('Margen: ', True, INK), ('intervalo de confianza al 95 %, remuestreando lances completos, no fotos '
                                   'sueltas.', False, MUTED)],
       [('Se distingue: ', True, INK), ('cuando el margen de la diferencia no incluye el cero.', False, MUTED)]],
      28, line=1.35, space_after=12)
pie(s)
notas(s, 'Misma cantidad de pasos, mismo lote efectivo, mismas semillas, mismos datos y '
         'aumentos. Lo único que cambia entre modelos es la arquitectura y su optimizador: los '
         'transformers usan tasa de aprendizaje decreciente por capa, como en sus papers. La época '
         'se elige con validación limpia, nunca con la prueba. '
         'Unidades: el macro-F1 va de 0 a 1. Ya no hablamos de "puntos". No usamos accuracy porque el '
         '68 % de la prueba son personas: un modelo que siempre diga "persona" acierta el 68 %. '
         'El margen sale de remuestrear lances completos, porque las fotos de un mismo lance no son '
         'independientes.')

# 9 ----------------------------------------------------------------- resultado principal
s = nueva()
titulo(s, 'Resultado', 'Los tres pares, en la prueba limpia de FishNet')
filas = [['Par', 'CNN', 'Transformer', 'Diferencia', 'Margen al 95 %', '¿Se distingue?']]
pares = [('ViT-B/16 - ResNet-50', 'Original', 'e4_resnet50_fishnet', 'e5_vit_b_16_fishnet'),
         ('DeiT-S - ResNet-50', 'Mismo tamaño', 'e4_resnet50_fishnet', 'e10_deit_s_fishnet'),
         ('Swin-T - ConvNeXt-T', 'Moderno', 'e11_convnext_t_fishnet', 'e12_swin_t_fishnet')]
for clave, et, cnn, tr in pares:
    d = VT['pares'].get(clave)
    if d is None or cnn not in M or tr not in M:
        filas.append([et, M[cnn]['modelo'] if cnn in M else 'pendiente',
                      M[tr]['modelo'] if tr in M else 'pendiente', '—', '—', '—'])
        continue
    filas.append([(et, True, INK),
                  '%s  %s' % (M[cnn]['modelo'], num(M[cnn]['macro_f1'])),
                  '%s  %s' % (M[tr]['modelo'], num(M[tr]['macro_f1'])),
                  (num(d['delta_medio'], 3, True), True, TEAL if d['distinguible'] else INK),
                  ic(d['ic95'], True),
                  ('Sí', True, TEAL) if d['distinguible'] else ('No', True, WARM)])
tabla(s, 120, 330, 1680, filas, [2.0, 2.6, 2.6, 1.6, 2.4, 1.7], alto_fila=92, size=28)
SEM = sorted({len(M[c]['semillas']) for c in M})
texto(s, 120, 712, 1680, 40,
      'macro-F1 de 15 especies, promedio de %s semillas. Diferencia = transformer − CNN.'
      % ' o '.join(str(n) for n in SEM), 24, MUTED)
LECT = [(clave, lectura(clave, VT['pares'].get(clave))) for clave, _, _, _ in pares]
texto(s, 120, 780, 1680, 200,
      [[('%s: ' % ETIQUETA[c], True, INK), (t or 'pendiente de entrenamiento.', False, INK)]
       for c, t in LECT], 28, line=1.3, space_after=10)
pie(s)
notas(s, 'Esta es la comparación central. Leer fila por fila: el macro-F1 de cada modelo, la '
         'diferencia y si se distingue del cero. ' +
         ' '.join('%s: %s' % (ETIQUETA[c], t) for c, t in LECT if t) +
         ' "No se distingue" no significa que sean iguales: significa que con esta prueba no se '
         'puede decir cuál es mejor.')

# 10 ---------------------------------------------------------------- robustez
s = nueva()
titulo(s, 'Resultado · qué tan firme es', 'La ventaja se achica en dos pruebas más exigentes')
imagen(s, os.path.join(FIGS, 'diferencias.png'), 120, 300, w=1060)
texto(s, 1240, 330, 560, 600,
      [[('Atunes como una clase. ', True, INK),
        ('Parte de la ventaja viene de separar un atún de otro.', False, MUTED)],
       [('Barcos no vistos. ', True, INK),
        ('El %s de las cajas de prueba viene de cámaras que no estaban en entrenamiento. Ahí los '
         'dos modelos bajan y dejan de distinguirse.' % pct(1 - A['camaras']['frac_cajas_camara_vista']),
         False, MUTED)],
       [('Verde: ', True, TEAL), ('se distingue. ', False, MUTED), ('Gris: ', True, '8A989B'),
        ('no se distingue.', False, MUTED)]],
      27, line=1.35, space_after=22)
pie(s)
nv = A['por_variante']['camara_nueva']['modelos']
nvi = A['por_variante']['camara_vista']['modelos']
notas(s, 'Cada línea es una diferencia con su margen. Si la línea cruza la raya punteada del cero, '
         'no se distingue. '
         'Atunes agrupados: si los cuatro atunes cuentan como una sola clase, la ventaja del ViT baja '
         'de +0,042 a +0,031 y el margen toca el cero. '
         'Barcos no vistos: hallazgo del profesor. En cámaras que ya estaban en entrenamiento, ResNet '
         '%s y ViT %s; en cámaras nuevas, %s y %s. Generalizar a un barco nuevo es lo más difícil, '
         'y para la caleta sería siempre un barco nuevo.'
      % (num(nvi['ResNet-50']['macro_f1']), num(nvi['ViT-B/16']['macro_f1']),
         num(nv['ResNet-50']['macro_f1']), num(nv['ViT-B/16']['macro_f1'])))

# 11 ---------------------------------------------------------------- por especie
pe_r, pe_v = A['por_especie']['ResNet-50'], A['por_especie']['ViT-B/16']
orden = sorted(pe_r, key=lambda k: -pe_r[k]['cajas'])
gana = sum(pe_v[k]['f1'] > pe_r[k]['f1'] + 1e-9 for k in orden)
pierde = sum(pe_v[k]['f1'] < pe_r[k]['f1'] - 1e-9 for k in orden)
s = nueva()
titulo(s, 'Resultado · por especie', 'El ViT-B/16 supera al ResNet-50 en %d de las %d especies' % (gana, len(orden)))
cd = CategoryChartData()
cd.categories = ['%s (%d)' % (k, round(pe_r[k]['cajas'])) for k in orden]
cd.add_series('ResNet-50', [round(pe_r[k]['f1'], 3) for k in orden])
cd.add_series('ViT-B/16', [round(pe_v[k]['f1'], 3) for k in orden])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, E(120), E(280), E(1680), E(640), cd)
ch = gf.chart
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False
ch.legend.font.size = P(24); ch.legend.font.name = FONT
for ser, col in zip(ch.plots[0].series, (RESN, VITC)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = C(col)
ch.plots[0].gap_width = 60
ca = ch.category_axis.tick_labels.font
ca.size = P(17); ca.name = FONT
va = ch.value_axis
va.maximum_scale, va.minimum_scale = 1.0, 0.0
va.tick_labels.font.size = P(20); va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = C('E4E2DA')
texto(s, 120, 930, 1680, 60, 'F1 por especie en la prueba limpia (0 a 1) · entre paréntesis, cajas de prueba',
      24, MUTED)
pie(s)
notas(s, 'Las especies van de la más común a la más rara. El ViT gana en %d, pierde en %d y empata '
         'en el resto. La diferencia más grande está en el albacora (%s contra %s), que se confunde '
         'con el atún aleta amarilla. En las especies con menos de 30 cajas de prueba, los dos modelos '
         'son poco confiables. La matriz de confusión completa está en el anexo y en '
         'docs/figs/ del repositorio.'
      % (gana, pierde, num(pe_v['Albacore']['f1'], 2), num(pe_r['Albacore']['f1'], 2)))

# 12 ---------------------------------------------------------------- dónde mira
ef = PG_F['energia_en_la_caja']
s = nueva()
titulo(s, 'Interpretabilidad', 'Los dos modelos miran el pez en una medida parecida')
imagen(s, os.path.join(PRIV, 'mapas_3.png'), 120, 290, h=660)
for j, t in enumerate(['Recorte', 'ResNet · Grad-CAM', 'ViT · Chefer']):
    texto(s, 120 + j * 245, 258, 245, 30, t, 20, MUTED, bold=True, align='c')
x0 = 1000
texto(s, x0, 300, 800, 50, 'Atención dentro del pez, por sobre el azar', 30, INK, bold=True)
for i, (k, et, col) in enumerate([('gradcam', 'ResNet-50 · Grad-CAM', RESN),
                                   ('chefer', 'ViT-B/16 · Chefer', VITC),
                                   ('rollout', 'ViT-B/16 · rollout', '8A989B')]):
    y = 380 + i * 150
    texto(s, x0, y, 500, 50, et, 28, INK)
    texto(s, x0 + 520, y - 12, 280, 80, num(ef[k]['sobre_barajado'], 3, True), 56, col, bold=True, align='r')
texto(s, x0, 840, 800, 120,
      'Fracción de la atención que cae en la caja del pez, menos la que cae en la caja de otro recorte. '
      '0 = no sigue al pez.', 24, MUTED, line=1.35)
pie(s, 'FishNet, %d recortes de la prueba limpia, pez en posición al azar · caja blanca = pez' % PG_F['n'])
notas(s, 'Calculamos los mapas sobre los mismos recortes de FishNet, como pidió el profesor. '
         'Primero, un problema de medición: con el pez siempre al centro, un mapa fijo que solo '
         'favorece el centro ya "acertaba" más que cualquier modelo. Por eso pusimos el pez en una '
         'posición al azar y restamos la atención que cae en la caja de otro recorte. '
         'Resultado: Grad-CAM y Chefer siguen al pez casi igual (%s y %s). El rollout casi no lo '
         'sigue. Respuesta al profesor: el ViT no mira menos el fondo que el ResNet. En Kaggle, '
         'Grad-CAM parecía mejor (%s contra %s sin corregir), pero ese número tiene el mismo sesgo al '
         'centro. Ejemplos: arriba los dos sobre el pez; al medio el ResNet mira el brazo; abajo el '
         'ViT se dispersa.'
      % (num(ef['gradcam']['sobre_barajado'], 3, True), num(ef['chefer']['sobre_barajado'], 3, True),
         num(PG_K['resultados']['gradcam']['media']), num(PG_K['resultados']['chefer']['media'])))

# 13 ---------------------------------------------------------------- no sé
o_r, o_v = ood('e4_resnet50_fishnet', 'id_sin_human'), ood('e5_vit_b_16_fishnet', 'id_sin_human')
ch_r, ch_v = CHILE['modelos']['ResNet-50'], CHILE['modelos']['ViT-B/16']
s = nueva()
titulo(s, 'Decir «no sé»', 'El ViT-B/16 reconoce mejor lo que nunca vio')
for i, esp in enumerate(['reineta', 'merluza', 'congrio']):
    imagen(s, os.path.join(PRIV, 'chile_%s.jpg' % esp), 120 + i * 290, 290, w=270, h=270)
    texto(s, 120 + i * 290, 572, 270, 40, esp.capitalize(), 26, INK, bold=True, align='c')
filas = [['', 'ResNet-50', 'ViT-B/16'],
         ['Rechaza las 45 fotos chilenas', pct(ch_r['rechaza_total']), pct(ch_v['rechaza_total'])],
         ['AUROC chilenas vs. FishNet', num(ch_r['auroc_vs_test_limpio_sin_personas']),
          num(ch_v['auroc_vs_test_limpio_sin_personas'])],
         ['AUROC Kaggle vs. FishNet', num(o_r['auroc']), num(o_v['auroc'])]]
tabla(s, 1020, 290, 780, filas, [3.4, 1.4, 1.4], alto_fila=80, size=24)
caja(s, 120, 660, 1680, 300)
texto(s, 170, 690, 1580, 260,
      [[('Criterio: ', True, INK), ('si la energía de la salida del modelo es más baja que la del 95 % de los '
                                     'peces de validación, responde «no sé». El umbral se fija en validación.',
                                     False, MUTED)],
       [('AUROC: ', True, INK), ('1 = separa perfecto lo conocido de lo nuevo; 0,5 = no los separa.',
                                 False, MUTED)],
       [('Ninguno está listo para la caleta: ', True, WARM),
        ('cuando aceptan una especie chilena, casi siempre dicen «atún aleta amarilla».', False, INK)]],
      27, line=1.35, space_after=14)
pie(s, 'Fotos de iNaturalist, CC BY-NC: jamillam, Benjamín Ignacio, Tomás Vargas Lavín · 45 fotos con créditos en el repositorio')
notas(s, 'Respuesta a la sugerencia 5 del profesor: el criterio y el AUROC. El criterio es la '
         'energía (Liu et al., 2020) con el umbral elegido en validación. '
         'Prueba con 45 fotos de iNaturalist de reineta, merluza y congrio, especies que no están en '
         'los datos: lo correcto es decir "no sé". El ViT lo dice más seguido (%s contra %s) y separa '
         'mejor. Lo mismo pasa con los peces de Kaggle. Limitación: son fotos de otro tipo (peces en '
         'la mano o en una mesa), así que parte de lo que detectan es el cambio de foto, no solo la '
         'especie. Es una prueba cualitativa.'
      % (pct(ch_v['rechaza_total']), pct(ch_r['rechaza_total'])))

# 14 ---------------------------------------------------------------- riesgos y auditoría
s = nueva()
titulo(s, 'Riesgos y auditoría', 'Cuatro problemas de los datos, medidos y corregidos')
filas = [['Problema', 'Medición', 'Qué hicimos'],
         ['Kaggle: 21 copias por foto', '411 fotos originales en 9.000', 'Reparto por foto original'],
         ['Kaggle: el fondo delata la especie', '0,971 sin ver el pez', 'Kaggle no decide la comparación'],
         ['FishNet: fuga por lance', '%s de la prueba oficial' % pct(fg['frac_cajas'], 1), 'Prueba limpia por lance'],
         ['Fotos de un lance no son independientes', 'Margen 4,6 veces más angosto', 'Remuestreo por lance'],
         ['FishNet: barcos repetidos', '%s de la prueba limpia' % pct(A['camaras']['frac_cajas_camara_vista'], 0),
          'Se reporta aparte (lámina 10)']]
tabla(s, 120, 300, 1680, filas, [3.4, 2.6, 2.6], alto_fila=88, size=26)
texto(s, 120, 860, 1680, 100,
      'Auditoría: ningún lance ni foto original cruza de entrenamiento a prueba en los repartos que '
      'usamos. El detalle está en el anexo.', 26, MUTED, line=1.35)
pie(s)
notas(s, 'Resumen de la auditoría, que en el Avance ocupaba cuatro láminas. Lo nuevo es la última '
         'fila: el reparto por lance limpia los casi duplicados, pero no evalúa barcos nuevos. Lo '
         'reportamos aparte y lo dejamos como próximo paso: entrenar con un reparto por cámara.')

# 15 ---------------------------------------------------------------- conclusiones
s = nueva(DARK)
titulo(s, 'Conclusiones', 'Qué podemos afirmar', oscuro=True)
conc = [
    ('En FishNet, el ViT-B/16 supera al ResNet-50: %s de macro-F1 (margen %s).'
     % (num(P_VR['delta_medio'], 3, True), ic(P_VR['ic95'], True))),
] + [t for c, t in LECT[1:] if t] + [
    'La ventaja del ViT-B/16 se achica al agrupar los atunes y no se distingue en barcos nuevos.',
    'El ViT no mira menos el fondo: los dos siguen al pez en una medida parecida.',
    'El ViT reconoce mejor lo que no conoce, pero ninguno está listo para especies chilenas.',
]
paso = min(150, int(620 / len(conc)))
for i, c in enumerate(conc):
    y = 320 + i * paso
    texto(s, 120, y, 80, 80, str(i + 1), 48, TEALL, bold=True)
    texto(s, 220, y + 6, 1560, paso, c, 30 if len(conc) > 4 else 34, ONDARK, line=1.25)
pie(s, oscuro=True)
notas(s, 'Cada afirmación tiene su evidencia en una lámina anterior: la 9 para los pares, la 10 '
         'para atunes y barcos nuevos, la 12 para dónde mira y la 13 para el "no sé".')

# 16 ---------------------------------------------------------------- próximos pasos
s = nueva()
titulo(s, 'Próximos pasos', 'Qué haría falta para usarlo en la caleta')
for i, (t, d) in enumerate([
        ('Contar', 'Agregar un detector entrenado con las cajas de FishNet, para encontrar y contar '
                   'los peces antes de clasificarlos.'),
        ('Fotos locales', 'Reunir fotos de la Caleta Portales con reineta, merluza y congrio marcados.'),
        ('Barcos nuevos', 'Entrenar y evaluar con un reparto por cámara, para medir el rendimiento en '
                          'un barco que el modelo nunca vio.')]):
    x = 120 + i * 570
    caja(s, x, 320, 530, 520)
    texto(s, x + 44, 360, 450, 60, t, 40, TEAL, bold=True)
    texto(s, x + 44, 440, 450, 380, d, 29, INK, line=1.4)
pie(s)
notas(s, 'El objetivo original incluía contar peces amontonados. No lo hicimos: el sistema '
         'clasifica peces ya marcados. El camino es un detector preentrenado ajustado con las cajas '
         'de FishNet, y después nuestro clasificador sobre cada caja. Lo segundo es tener datos '
         'locales, y lo tercero, medir con barcos nuevos, que es donde los modelos más caen.')

# 17 ---------------------------------------------------------------- referencias
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

# 18 ---------------------------------------------------------------- cierre
s = nueva(DARK)
texto(s, 120, 400, 1680, 160, 'Gracias', 120, ONDARK, bold=True)
texto(s, 120, 580, 1680, 60, '¿Preguntas?', 44, DMUTED)
texto(s, 120, 860, 1680, 50, 'github.com/Nirvesta04/CaletaScan', 30, TEALL, bold=True)
notas(s, 'Cerrar e invitar a preguntas. El repositorio tiene el código, los resultados, la data '
         'card y el informe.')

# ================================================================ ANEXOS
# A1 --------------------------------------------------------------- épocas
s = nueva()
titulo(s, 'Anexo · entrenamiento', 'Macro-F1 de validación por época')
cd = CategoryChartData()
cd.categories = [str(i) for i in range(1, 9)]
cols = {'ResNet-50': RESN, 'ViT-B/16': VITC, 'DeiT-S': '3E8EC4', 'ConvNeXt-T': 'D99A2B', 'Swin-T': '6B5BA8'}
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
ch.value_axis.major_gridlines.format.line.color.rgb = C('E4E2DA')
filas = [['Modelo', 'Mejor época', 'Minutos']]
for m, por_semilla in A['epocas'].items():
    if not por_semilla:
        continue
    ep = [[e['val_macro_f1'] for e in h] for h in por_semilla.values()]
    medias = [sum(c) / len(c) for c in zip(*ep)]
    mins = [sum(e['min'] for e in h) for h in por_semilla.values()]
    filas.append([m, str(medias.index(max(medias)) + 1), '%.0f' % (sum(mins) / len(mins))])
tabla(s, 1280, 300, 520, filas, [2.2, 1.7, 1.5], alto_fila=70, size=22)
texto(s, 1280, 760, 520, 160, 'Promedio de 3 semillas. Minutos por semilla, con la lectura de imágenes.', 22, MUTED, line=1.35)
pie(s)
notas(s, 'Validación limpia (lances que no están en entrenamiento). El ResNet-50 se estanca '
         'temprano; el ViT-B/16 sigue mejorando en algunas semillas. El tiempo incluye la lectura de '
         'imágenes, que en este equipo es el cuello de botella.')

# A2 --------------------------------------------------------------- matrices
s = nueva()
titulo(s, 'Anexo · confusión', 'Qué especie predice cada modelo, por especie real')
imagen(s, os.path.join(FIGS, 'confusion_e4.png'), 120, 270, h=700)
imagen(s, os.path.join(FIGS, 'confusion_e5.png'), 980, 270, h=700)
pie(s, 'Porcentaje de cada fila · prueba limpia de FishNet · promedio de 3 semillas')
notas(s, 'Matriz de confusión entre especies. Cada fila suma 100 %. La confusión principal es '
         'entre atunes: albacora, aleta amarilla y patudo.')

# A3 --------------------------------------------------------------- auditoría y Kaggle
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
notas(s, 'Detalle de la auditoría del Avance. El vecino más cercano muestra por qué Kaggle es '
         'trivial: sin ninguna red, comparar miniaturas acierta 0,829. En Kaggle, ResNet y ViT no se '
         'distinguen. El sobreajuste es fuerte en los dos, lo que es esperable con pocas fotos por especie.')

os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
prs.save(SALIDA)
print('guardado: %s (%d láminas)' % (SALIDA, N[0]))
