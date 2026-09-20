# -*- coding: utf-8 -*-
"""Genera el PPTX del Avance de Caleta-Scan (G3) a partir del mazo corregido."""
import copy
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

PX = 6350                      # 1 px del diseno (1920x1080) = 6350 EMU
def E(px):  return Emu(int(round(px * PX)))
def P(px):  return Pt(px / 2.0)        # 1 px = 0,5 pt
def C(h):   return RGBColor.from_string(h)

DARK, LIGHT, ALT, WHITE, BORDER = '0F2226', 'F5F4EF', 'ECEAE2', 'FFFFFF', 'DCDAD0'
TEAL, TEALL, WARM = '0C6F66', '5CC0B0', 'B3541E'
WARMBG, WARMBD, WARMTX = 'F6E9DD', 'E2C8AE', '8A4416'
INK, MUTED, FOOT = '16282C', '4F6165', '6C7B7E'
ONDARK, DMUTED, DCARD, DFOOT = 'F0EFE9', 'A9BCBD', '16343A', '8FA5A7'
TCARD, TTX, TMUT, TFOOT = '0A5A53', 'F2F7F5', 'B6E3DA', '9FD5CB'
HEAD, BODY = 'Segoe UI', 'Segoe UI'

prs = Presentation()
prs.slide_width, prs.slide_height = E(1920), E(1080)
BLANK = prs.slide_layouts[6]


def nueva(fondo):
    s = prs.slides.add_slide(BLANK)
    f = s.background.fill
    f.solid(); f.fore_color.rgb = C(fondo)
    return s


def texto(s, x, y, w, h, partes, size, color, bold=False, line=1.3,
          align='l', font=BODY, spc=None, space_after=0):
    """partes: str, o lista de (texto, bold, color) o de listas (parrafos)."""
    box = s.shapes.add_textbox(E(x), E(y), E(w), E(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(partes, str):
        partes = [partes]
    parrafos = partes if isinstance(partes[0], list) else [partes]
    for i, par in enumerate(parrafos):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {'l': PP_ALIGN.LEFT, 'r': PP_ALIGN.RIGHT, 'c': PP_ALIGN.CENTER}[align]
        p.line_spacing = line
        if space_after:
            p.space_after = P(space_after)
        trozos = par if isinstance(par, list) else [par]
        for t in trozos:
            if isinstance(t, str):
                t = (t, bold, color)
            txt, b, col = (t + (color,))[:3] if len(t) == 2 else t
            r = p.add_run(); r.text = txt
            r.font.size = P(size); r.font.bold = b
            r.font.name = font; r.font.color.rgb = C(col)
            if spc:
                r.font._rPr.set('spc', str(int(spc * 100)))
    return box


def caja(s, x, y, w, h, fill, borde=None, barra=None, radio=16):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, E(x), E(y), E(w), E(h))
    sh.adjustments[0] = radio / float(min(w, h))
    sh.fill.solid(); sh.fill.fore_color.rgb = C(fill)
    if borde:
        sh.line.color.rgb = C(borde); sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ''
    if barra:
        b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, E(x + 10), E(y), E(w - 20), E(6))
        b.fill.solid(); b.fill.fore_color.rgb = C(barra)
        b.line.fill.background(); b.shadow.inherit = False
    return sh


def barra_recta(s, x, y, w, h, fill):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, E(x), E(y), E(w), E(h))
    b.fill.solid(); b.fill.fore_color.rgb = C(fill)
    b.line.fill.background(); b.shadow.inherit = False
    return b


def cabecera(s, ojo, titulo, ojo_col, tit_col, y=128, tsize=70, tw=1660, th=170):
    texto(s, 128, y, tw, 40, ojo.upper(), 26, ojo_col, bold=True, font=HEAD, spc=2)
    texto(s, 128, y + 62, tw, th, titulo, tsize, tit_col, bold=True, font=HEAD, line=1.1)


def pie(s, izq, num, col):
    texto(s, 128, 978, 1500, 40, izq, 23, col)
    texto(s, 1552, 978, 240, 40, str(num), 24, col, align='r')


def notas(s, t):
    s.notes_slide.notes_text_frame.text = t


def fila(x, total, pesos, gap):
    libre = total - gap * (len(pesos) - 1)
    suma = float(sum(pesos))
    out, cx = [], x
    for p in pesos:
        w = libre * p / suma
        out.append((cx, w)); cx += w + gap
    return out


NOM = 'Patricio Toledo · Andrés Savedra · Sebastián Casas · Bastián Molina'

# ---------------------------------------------------------------- 1 portada
s = nueva(DARK)
texto(s, 128, 128, 1660, 40, 'PUCV · APRENDIZAJE AUTOMÁTICO II · PRESENTACIÓN DE AVANCE',
      26, TEALL, bold=True, font=HEAD, spc=2)
texto(s, 128, 380, 1660, 200, 'Caleta-Scan', 150, ONDARK, bold=True, font=HEAD, line=1.0)
texto(s, 128, 600, 1400, 180,
      '¿Puede una foto decir qué pescado se acaba de desembarcar? Y si hay dos '
      'arquitecturas capaces de hacerlo, ¿cuál conviene usar?', 44, DMUTED, line=1.3)
texto(s, 128, 880, 1400, 50, 'Grupo G3 · ' + NOM, 30, ONDARK, bold=True)
texto(s, 128, 930, 1400, 50, '28 de septiembre de 2026', 30, DMUTED)
notas(s, 'Presentarse y dar el marco en 20 segundos: somos el grupo G3, el proyecto se llama '
         'Caleta-Scan, y en los proximos 15 minutos vamos a contar que datos usamos, que '
         'encontramos en ellos, y como va la comparacion entre las dos arquitecturas que pide '
         'el curso. Avisar que al final hay tiempo para preguntas.')

# ---------------------------------------------------------------- 2 problema
s = nueva(LIGHT)
cabecera(s, 'Problema y motivación',
         'En la Caleta Portales, lo que se desembarca se anota a mano', TEAL, INK, tw=1560)
texto(s, 128, 368, 1520, 130,
      'Sernapesca registra especie y cantidad durante la descarga. Es lento, depende de quién '
      'esté ese día, y en temporada alta hay más botes que inspectores.', 34, MUTED, line=1.45)
for (cx, cw), (tit, cue) in zip(fila(128, 1664, [1, 1, 1], 28), [
        ('La foto ya existe', 'Sacar una foto de la faena toma segundos. Aprovecharla no cambia el trabajo de nadie.'),
        ('Lo que aportaría', 'Una lista de especies por foto, con su nivel de confianza, que la persona revisa y corrige.'),
        ('Apoyo, no reemplazo', 'La decisión sigue siendo del inspector. El sistema ordena y propone.')]):
    caja(s, cx, 500, cw, 260, WHITE, BORDER)
    texto(s, cx + 36, 536, cw - 72, 50, tit, 34, TEAL, bold=True, font=HEAD)
    texto(s, cx + 36, 600, cw - 72, 150, cue, 28, MUTED, line=1.4)
pie(s, 'Caleta-Scan · G3', 2, FOOT)
notas(s, 'Aqui se juega la fila "relevancia local justificada" de la rubrica. Insistir en que el '
         'problema es real y cercano: Valparaiso, Caleta Portales, registro manual de '
         'desembarque. Y dejar claro desde ya el rol: el sistema propone, la persona decide. Si '
         'alguien pregunta por que no se hace ya, la respuesta honesta es que falta el dato '
         'local, y eso lo abordamos en la lamina siguiente.')

# ---------------------------------------------------------------- 3 alcance
s = nueva(ALT)
cabecera(s, 'Alcance', 'Qué prometemos y qué no', TEAL, INK)
for (cx, cw), (tit, col, items) in zip(fila(128, 1664, [1, 1], 32), [
        ('Sí', TEAL, ['Decir la especie de un pez ya marcado en la foto',
                      'Comparar ResNet y ViT con el mismo presupuesto de entrenamiento',
                      'Contestar «no sé» cuando la imagen no se parece a lo entrenado']),
        ('Todavía no', WARM, ['Encontrar los peces solo, sin que nadie los marque',
                              'Reconocer reineta, merluza o congrio: no existen en los datos públicos',
                              'Funcionar en el muelle sin recolectar fotos locales primero'])]):
    caja(s, cx, 380, cw, 440, WHITE, BORDER, barra=col)
    texto(s, cx + 44, 420, cw - 88, 60, tit, 38, col, bold=True, font=HEAD)
    texto(s, cx + 44, 494, cw - 88, 270, [['•  ' + i] for i in items], 30, '3C4F53',
          line=1.45, space_after=18)
pie(s, 'Caleta-Scan · G3', 3, FOOT)
notas(s, 'Decir esto temprano y sin rodeos da credibilidad al resto: se nota que sabemos donde '
         'estan los limites. Dos frases clave: "no hay detector, las cajas las marca una persona '
         'o vienen anotadas" y "ninguna especie chilena aparece en los datos publicos, asi que lo '
         'que demostramos es el metodo". Si el profesor pregunta por que no usamos datos '
         'chilenos: no existen publicos y recolectarlos es un proyecto en si mismo.')

# ---------------------------------------------------------------- 4 datos
s = nueva(LIGHT)
cabecera(s, 'Datos · fuentes', 'Dos conjuntos públicos, con papeles distintos', TEAL, INK)
for (cx, cw), (tit, cifra, cue, rol) in zip(fila(128, 1664, [1, 1], 32), [
        ('Kaggle · pescado de supermercado', '9.000 fotos\n9 especies',
         'Fondo de estudio, luz pareja. Trae la silueta exacta del pez marcada, y eso nos deja '
         'hacer experimentos que en la otra base serían imposibles.',
         'Para qué lo usamos: probar y diagnosticar'),
        ('FishNet · cámaras en cubierta', '20.002 fotos\n99.019 peces',
         'Barcos pesqueros reales: de noche, mojado, peces amontonados y tapados, 77 cámaras distintas.',
         'Para qué lo usamos: decidir entre las dos arquitecturas')]):
    caja(s, cx, 370, cw, 476, WHITE, BORDER)
    texto(s, cx + 44, 406, cw - 88, 90, tit, 38, INK, bold=True, font=HEAD, line=1.15)
    texto(s, cx + 44, 512, cw - 88, 160, [[l] for l in cifra.split('\n')], 62, TEAL,
          bold=True, font=HEAD, line=1.1)
    texto(s, cx + 44, 664, cw - 88, 90, cue, 28, MUTED, line=1.4)
    texto(s, cx + 44, 782, cw - 88, 40, rol, 28, INK, bold=True)
pie(s, 'Licencias CC BY 4.0 y CC BY-SA 4.0 · las imágenes no se redistribuyen', 4, FOOT)
notas(s, 'Fila "fuentes definidas" de la rubrica. Lo importante es que los dos conjuntos no '
         'compiten: uno sirve para diagnosticar y el otro para decidir. Mencionar que las '
         'licencias son distintas y que la de FishNet es virica, por eso no redistribuimos '
         'imagenes. Si preguntan por OzFish: estaba previsto como tercera fuente, su portal exige '
         'un token y devolvio error, asi que lo reemplazamos por una prueba cruzada entre estos dos.')

# ---------------------------------------------------------------- 5 etiquetado
s = nueva(LIGHT)
cabecera(s, 'Datos · criterio de etiquetado', 'Qué clases aprende y cuáles reportamos', TEAL, INK)
for (cx, cw), (n, tit, cue) in zip(fila(128, 1664, [1, 1, 1], 26), [
        ('20', 'clases que predice', 'Las 19 especies con al menos 100 ejemplos de entrenamiento, '
         'más una clase llamada «raras» que agrupa las 14 que no llegan.'),
        ('13', 'clases que reportamos', 'Solo aquellas con 30 o más ejemplos de prueba limpios. '
         'Con menos ejemplos de prueba, el número no sería confiable.'),
        ('3', 'no son especies', 'Persona, sin pez y desconocido. Se aprenden, porque hay que '
         'distinguirlos, pero no entran en la nota de especies.')]):
    caja(s, cx, 370, cw, 345, WHITE, BORDER)
    texto(s, cx + 36, 402, cw - 72, 90, n, 64, TEAL, bold=True, font=HEAD, line=1.1)
    texto(s, cx + 36, 484, cw - 72, 50, tit, 32, INK, bold=True, font=HEAD)
    texto(s, cx + 36, 542, cw - 72, 120, cue, 26, MUTED, line=1.4)
texto(s, 128, 742, 1600, 140,
      [[('Un caso especial: ', True, INK),
        ('«atún» a secas es una etiqueta ambigua, porque el anotador no resolvió cuál era. Si el '
         'modelo dice «atún de aleta amarilla» sobre un «atún», lo contamos como acierto, porque '
         'el error está en la etiqueta y no en el modelo.', False, MUTED)]], 30, MUTED, line=1.45)
pie(s, 'Caleta-Scan · G3', 5, FOOT)
notas(s, 'Fila "criterio de etiquetado" de la rubrica. La idea que tiene que quedar es que '
         'distinguimos entre lo que el modelo predice y aquello sobre lo que nos atrevemos a '
         'reportar un numero. Si preguntan por que no bajamos el umbral de 100 ejemplos para '
         'cubrir mas especies: lo probamos y no cambia nada, porque el cuello de botella no es el '
         'entrenamiento sino los ejemplos de prueba.')

# ---------------------------------------------------------------- 6 split
s = nueva(ALT)
cabecera(s, 'Datos · split', 'Repartimos por origen, no por imagen', TEAL, INK)
caja(s, 128, 350, 1664, 250, WHITE, BORDER)
cols = [(160, 430), (620, 210), (860, 210), (1100, 210), (1360, 400)]
enc = ['Conjunto', 'Entrena', 'Valida', 'Prueba', 'Lo que nunca se parte']
barra_recta(s, 129, 351, 1662, 58, 'E3E1D7')
for (cxx, cww), t in zip(cols, enc):
    texto(s, cxx, 368, cww, 40, t, 28, INK, bold=True,
          align='r' if t in ('Entrena', 'Valida', 'Prueba') else 'l')
filas = [('Kaggle', '6.328', '1.336', '1.336', 'La foto original *', INK),
         ('    · fotos distintas', '287', '62', '62', '', FOOT),
         ('FishNet', '72.885', '12.927', '13.207', 'La secuencia de video *', INK)]
for i, (a, b, c_, d, e, col) in enumerate(filas):
    yy = 424 + i * 56
    for (cxx, cww), t in zip(cols, [a, b, c_, d, e]):
        texto(s, cxx, yy, cww, 40, t, 27, col,
              align='r' if t in (b, c_, d) and t != a else 'l')
texto(s, 128, 620, 1600, 110,
      [[('Las 1.000 imágenes por especie de Kaggle son en realidad ', False, MUTED),
        ('30 a 50 fotos giradas y espejadas', True, INK),
        ('. En FishNet, una secuencia son fotogramas seguidos del mismo lance. Repartiendo imagen '
         'por imagen, el modelo respondería en la prueba sobre fotos que ya vio al entrenar, y el '
         'resultado saldría más alto de lo real.', False, MUTED)]], 30, MUTED, line=1.4)
caja(s, 128, 770, 1596, 166, WHITE, BORDER)
barra_recta(s, 128, 780, 8, 146, TEAL)
texto(s, 172, 798, 1510, 130,
      [[('¿Y no es tramposo entrenar con copias? ', True, INK),
        ('No, por dos razones. Sobre cada copia aplicamos ', False, MUTED),
        ('recorte aleatorio, espejado y cambio de brillo y color en cada pasada', True, INK),
        (': el modelo nunca ve dos veces los mismos píxeles. Y las copias de una foto van ', False, MUTED),
        ('todas a la misma partición', True, INK), (', nunca repartidas.', False, MUTED)]],
      28, MUTED, line=1.45)
texto(s, 128, 978, 1500, 40,
      '*  El reparto oficial de FishNet no cumple esto. Lo corregimos nosotros: lámina 8', 23, WARM)
texto(s, 1552, 978, 240, 40, '6', 24, FOOT, align='r')
notas(s, 'Fila "split train/val/test" de la rubrica. Decirlo literal: el riesgo es que el modelo '
         'responda sobre fotos que ya vio. Al leer la tabla, decir "287 fotos en 6.328 variantes", '
         'nunca "6.328 imagenes" a secas. El recuadro de abajo responde por adelantado la pregunta '
         'que siempre aparece: el aumento es aleatorio en cada pasada (recorte entre 70 y 100 por '
         'ciento, espejado, brillo y color), asi que ninguna pasada repite pixeles; y el reparto es '
         'por foto, asi que las 62 fotos de prueba no tienen ninguna hermana en entrenamiento.')

# ---------------------------------------------------------------- 7 riesgo fondo
s = nueva(LIGHT)
cabecera(s, 'Riesgo 1 · el atajo del fondo',
         'Borramos el pez de la foto, y el modelo siguió acertando', WARM, INK, tw=1560)
anchos = fila(128, 1664, [1, 1, 1.3], 32)
for (cx, cw), (rot, val, bg, bd, tc, vc) in zip(anchos[:2], [
        ('Con la foto completa', '0,997', WHITE, BORDER, MUTED, INK),
        ('Con el pez borrado', '0,971', WARMBG, WARMBD, WARMTX, WARM)]):
    caja(s, cx, 420, cw, 280, bg, bd)
    texto(s, cx + 44, 456, cw - 88, 50, rot, 30, tc)
    texto(s, cx + 44, 516, cw - 88, 160, val, 110, vc, bold=True, font=HEAD, line=1.05)
cx, cw = anchos[2]
caja(s, cx, 420, cw, 280, WHITE, BORDER)
texto(s, cx + 44, 456, cw - 88, 50, 'Mitigación', 32, TEAL, bold=True, font=HEAD)
texto(s, cx + 44, 516, cw - 88, 180,
      'Kaggle deja de ser el resultado y pasa a ser el laboratorio. La comparación entre '
      'arquitecturas se decide en FishNet, donde el fondo cambia todo el tiempo.', 27, MUTED, line=1.4)
texto(s, 128, 740, 1600, 130,
      [[('Cada especie se fotografió en una sesión distinta, sobre una mesa y una luz levemente '
         'distintas. El modelo aprende la mesa, no el pescado. ', False, MUTED),
        ('Un 99,7 % que no demuestra que reconozca peces.', True, INK)]], 32, MUTED, line=1.45)
pie(s, 'Kaggle · macro-F1 de prueba · 3 semillas', 7, FOOT)
notas(s, 'Esta es la lamina mas fuerte del mazo; darle aire. Contar el experimento en orden: '
         'sospechamos del 99,7 %, usamos la silueta que trae el dataset para borrar el pez y dejar '
         'solo el fondo, reentrenamos, y siguio acertando casi igual. La frase que resume: el '
         'modelo aprendio la mesa, no el pescado. Nadie que mire solo la accuracy habria encontrado '
         'esto. IMPORTANTE: decir en voz alta que este problema NO se arregla con el split, porque '
         'el fondo identifica a la especie entera y no a una foto.')

# ---------------------------------------------------------------- 8 riesgo fuga
s = nueva(LIGHT)
cabecera(s, 'Riesgo 2 · fuga en el reparto oficial',
         'El reparto que publica FishNet deja casi los mismos fotogramas a los dos lados',
         WARM, INK, tsize=68, tw=1600)
anchos = fila(128, 1664, [1, 1, 1.3], 32)
for (cx, cw), (val, cue, bg, bd, vc, tc) in zip(anchos[:2], [
        ('44,5 %', 'de la prueba comparte secuencia con el entrenamiento', WARMBG, WARMBD, WARM, WARMTX),
        ('+80 %', 'se infla el resultado si se usa ese reparto sin corregirlo', WHITE, BORDER, INK, MUTED)]):
    caja(s, cx, 430, cw, 280, bg, bd)
    texto(s, cx + 40, 462, cw - 80, 140, val, 96, vc, bold=True, font=HEAD, line=1.05)
    texto(s, cx + 40, 604, cw - 80, 100, cue, 28, tc, line=1.4)
cx, cw = anchos[2]
caja(s, cx, 430, cw, 280, WHITE, BORDER, barra=TEAL)
texto(s, cx + 40, 470, cw - 80, 50, 'Mitigación', 32, TEAL, bold=True, font=HEAD)
texto(s, cx + 40, 530, cw - 80, 170,
      'Sacamos de la prueba toda imagen cuya secuencia esté en el entrenamiento, y reportamos '
      'siempre los dos números: el oficial y el limpio.', 27, MUTED, line=1.4)
texto(s, 128, 750, 1600, 130,
      [[('Lo comprobamos mirando los píxeles: dos fotogramas del mismo lance se parecen en ', False, MUTED),
        ('0,87', True, INK), (' de correlación, contra ', False, MUTED), ('0,03', True, INK),
        (' entre dos fotos cualquiera. Son casi la misma imagen.', False, MUTED)]], 32, MUTED, line=1.45)
pie(s, 'FishNet · 3.743 de 13.097 secuencias cruzan particiones', 8, FOOT)
notas(s, 'Subrayar que esta fuga no la fabricamos nosotros: viene en el reparto que publica el '
         'dataset, y cualquiera que use esos numeros sin revisarlos esta reportando cifras '
         'infladas. Explicar que es una secuencia con las manos: fotogramas seguidos del mismo '
         'lance, el mismo pez en la misma cubierta un segundo despues. Si preguntan como lo '
         'detectamos, la respuesta corta es que el dataset trae un identificador de secuencia que '
         'el reparto oficial ignoro.')

# ---------------------------------------------------------------- 9 riesgo n
s = nueva(LIGHT)
cabecera(s, 'Riesgo 3 · márgenes de error que engañan',
         'La prueba de Kaggle tiene 62 fotos, no 1.336 imágenes', WARM, INK, tsize=60, th=100)
texto(s, 128, 320, 1620, 140,
      [[('El margen de error responde a una sola pregunta: ', False, MUTED),
        ('si hubiéramos usado otras fotos de prueba, ¿cuánto habría cambiado el resultado?', True, INK),
        (' Contar 1.336 imágenes supone que hay 1.336 pruebas distintas. En realidad hay 62, '
         'porque veinte rotaciones de una misma foto son una sola prueba.', False, MUTED)]],
      31, MUTED, line=1.4)
anchos = fila(128, 1664, [1, 1, 1.25], 28)
for (cx, cw), (tit, val, cue, bg, bd, tc, vc, cc) in zip(anchos[:2], [
        ('Contando imágenes', '± 1,2 pp',
         'Sale 4,6 veces más angosto de lo real, y así cualquier diferencia pequeña parece un hallazgo',
         WARMBG, WARMBD, WARMTX, WARM, WARMTX),
        ('Contando fotos', '± 5,4 pp', 'Cuenta las 62 fotos que de verdad son distintas entre sí',
         WHITE, BORDER, INK, INK, MUTED)]):
    caja(s, cx, 470, cw, 310, bg, bd)
    texto(s, cx + 34, 502, cw - 68, 46, tit, 29, tc, bold=True, font=HEAD)
    texto(s, cx + 34, 560, cw - 68, 100, val, 66, vc, bold=True, font=HEAD, line=1.1)
    texto(s, cx + 34, 658, cw - 68, 100, cue, 24, cc, line=1.35)
cx, cw = anchos[2]
caja(s, cx, 470, cw, 310, WHITE, BORDER, barra=TEAL)
texto(s, cx + 34, 506, cw - 68, 46, 'Mitigación', 29, TEAL, bold=True, font=HEAD)
texto(s, cx + 34, 564, cw - 68, 190,
      'Todos nuestros márgenes se calculan sorteando grupos completos: la foto original en '
      'Kaggle, la secuencia de video en FishNet. Nunca imágenes sueltas.', 25, MUTED, line=1.4)
caja(s, 128, 800, 1596, 112, WHITE, BORDER)
barra_recta(s, 128, 810, 8, 92, TEAL)
texto(s, 172, 824, 1510, 92,
      [[('«pp» son puntos porcentuales. ', True, INK),
        ('Un resultado de 95 % con ± 5,4 pp quiere decir que el valor real podría bajar hasta '
         'cerca del 90 %. Puesto en simple: preguntarle veinte veces lo mismo a una persona no es '
         'encuestar a veinte personas.', False, MUTED)]], 27, MUTED, line=1.4)
pie(s, 'Márgenes al 95 % de confianza, calculados sobre un resultado de 0,95', 9, FOOT)
notas(s, 'Este es el riesgo menos vistoso y el que mas nos protege de sacar conclusiones falsas. '
         'El orden para contarlo: primero la pregunta que responde el margen, despues los dos '
         'numeros, y al final la frase de las veinte respuestas de una sola persona. Mencionar que '
         'por esto la diferencia que se vera entre las dos arquitecturas en Kaggle no cuenta como '
         'diferencia, aunque el ViT acierte todo. Si preguntan de donde sale el 4,6: es la raiz de '
         '1.336 dividido 62.')

# ---------------------------------------------------------------- 10 auditoria
s = nueva(ALT)
cabecera(s, 'Auditoría',
         'Buscamos errores en nuestros propios resultados. Las cuatro pruebas los confirmaron.',
         TEAL, INK, tsize=64, tw=1620, th=160)
caja(s, 128, 372, 1664, 400, WHITE, BORDER)
barra_recta(s, 129, 373, 1662, 56, 'E3E1D7')
texto(s, 160, 388, 500, 40, 'Qué intentamos', 27, INK, bold=True)
texto(s, 700, 388, 900, 40, 'Qué salió', 27, INK, bold=True)
filas = [('Buscar imágenes de prueba que también estuvieran en el entrenamiento',
          [('Imposible en lo que reportamos.', True, TEAL),
           (' El que sí filtraba era el reparto oficial de FishNet, y lo detectamos nosotros', False, MUTED)]),
         ('Desconfiar de las etiquetas y comparar píxeles',
          [('La purga funcionó:', True, TEAL),
           (' lo que sacamos tenía un casi gemelo el 38 % de las veces; lo que dejamos, solo el 2,6 %', False, MUTED)]),
         ('Quitar además ese 2,6 % restante',
          [('El resultado no se mueve:', True, TEAL),
           (' 0,003 de cambio, y la ventaja del ViT se mantiene igual', False, MUTED)]),
         ('Medir cuánto memoriza cada modelo',
          [('FishNet es difícil y las dos memorizan. El ViT memoriza más ', False, MUTED),
           ('y aun así generaliza mejor', True, TEAL),
           (': ahí está el margen que queda por ganar', False, MUTED)])]
for i, (a, b) in enumerate(filas):
    yy = 446 + i * 82
    texto(s, 160, yy, 510, 80, a, 26, INK, line=1.3)
    texto(s, 700, yy, 1060, 80, [b], 26, MUTED, line=1.3)
caja(s, 128, 800, 1596, 120, WHITE, BORDER)
barra_recta(s, 128, 810, 8, 100, TEAL)
texto(s, 172, 826, 1510, 90,
      [[('Ninguna de las cuatro cambió la conclusión. ', True, INK),
        ('La segunda es la más exigente: no le cree a ninguna etiqueta, compara cada imagen de '
         'prueba con todas las de entrenamiento, y en Kaggle también con sus ocho rotaciones y '
         'espejos.', False, MUTED)]], 27, MUTED, line=1.4)
pie(s, 'scripts/auditoria.py · resultados en el repositorio', 10, FOOT)
notas(s, 'Esta lamina responde por adelantado la pregunta incomoda: "con esos numeros, no tendran '
         'una fuga?". Decirla en tono de fortaleza, no de disculpa: fuimos nosotros los que '
         'buscamos el error, y no aparecio. Si preguntan por la cuarta fila, el sobreajuste, la '
         'respuesta honesta es que FishNet tiene 20 clases muy desbalanceadas y un presupuesto de '
         'entrenamiento corto, asi que memorizar era esperable; lo relevante es que el modelo que '
         'mas memoriza es tambien el que mejor generaliza.')

# ---------------------------------------------------------------- 11 vecino
s = nueva(TEAL)
cabecera(s, 'Auditoría · el veredicto sobre Kaggle',
         'No hay fuga. El problema es que el examen era demasiado fácil.',
         TMUT, TTX, tsize=74, tw=1620)
for (cx, cw), (val, cue) in zip(fila(128, 1664, [1, 1], 40), [
        ('82,9 %', 'acierta un método sin aprendizaje: copiar la especie de la miniatura de '
                   '32×32 píxeles más parecida'),
        ('55,9 %', 'acierta ese mismo método en FishNet, donde las redes llegan al 82-85 %')]):
    caja(s, cx, 440, cw, 300, TCARD)
    texto(s, cx + 44, 476, cw - 88, 150, val, 110, TTX, bold=True, font=HEAD, line=1.05)
    texto(s, cx + 44, 626, cw - 88, 110, cue, 29, TMUT, line=1.4)
texto(s, 128, 786, 1620, 140,
      'En Kaggle, hasta un método sin aprendizaje acierta casi siempre. En FishNet no alcanza con '
      'parecerse, y por eso la comparación entre arquitecturas la hacemos ahí.', 34, 'D8EDE8', line=1.45)
pie(s, 'Vecino más cercano sobre miniaturas en blanco y negro', 11, TFOOT)
notas(s, 'Rematar el argumento de la lamina del fondo con una segunda medicion, independiente. '
         'Explicar el metodo en palabras: achicamos cada foto a una miniatura diminuta en blanco y '
         'negro, buscamos la mas parecida del entrenamiento y copiamos su etiqueta. Sin redes, sin '
         'entrenar nada, 83 %. Es la mejor prueba de que ese dataset no mide lo que parece. Y sirve '
         'de transicion: a partir de aqui, todos los numeros son de FishNet.')

# --------------------------------------------------- 12 y 13 resultados
def lamina_modelo(num, ojo, titulo, tarjetas, ci_lo, ci_hi, txt_barra, txt_cierre,
                  pasos, pie_txt, nota):
    s = nueva(LIGHT)
    cabecera(s, ojo, titulo, TEAL, INK, tsize=68, th=100)
    anchos = fila(128, 1664, [t[4] for t in tarjetas], 24)
    for (cx, cw), (rot, val, cue, dest, _pe) in zip(anchos, tarjetas):
        caja(s, cx, 296, cw, 316, WHITE, BORDER, barra=TEAL if dest else None)
        texto(s, cx + 32, 330, cw - 64, 44, rot, 25, MUTED)
        texto(s, cx + 32, 384, cw - 64, 100, val, 70, TEAL if dest else INK,
              bold=True, font=HEAD, line=1.1)
        texto(s, cx + 32, 486, cw - 64, 120, [cue], 23, MUTED, line=1.35)
    caja(s, 128, 632, 1664, 218, WHITE, BORDER)
    barra_recta(s, 128, 642, 8, 198, TEAL)
    texto(s, 176, 664, 24, 40, '0', 25, FOOT)
    izq = int(round(1400 * ci_lo)); anc = max(14, int(round(1400 * (ci_hi - ci_lo))))
    barra_recta(s, 210, 668, izq, 20, BORDER)
    barra_recta(s, 210 + izq, 668, anc, 20, TEAL)
    barra_recta(s, 210 + izq + anc, 668, 1400 - izq - anc, 20, BORDER)
    texto(s, 1626, 664, 40, 40, '1', 25, FOOT)
    texto(s, 210 + izq + anc + 16, 706, 900, 40, txt_barra, 25, TEAL, bold=True)
    texto(s, 176, 752, 1570, 90, [txt_cierre], 27, MUTED, line=1.4)
    texto(s, 128, 874, 1620, 90, [pasos], 27, MUTED, line=1.4)
    pie(s, pie_txt, num, FOOT)
    notas(s, nota)

lamina_modelo(
    12, 'Resultados preliminares · baseline 1', 'ResNet-50, entrenado y medido',
    [('Kaggle · macro-F1', '0,997', 'El techo del conjunto: ya no queda nada que medir ahí', False, 1),
     ('FishNet limpia · macro-F1', '0,352', '20 clases, varias casi idénticas entre sí', True, 1),
     ('FishNet · aciertos totales', '85,6 %',
      [('De 7.159 recortes. Está inflado: ', False, MUTED), ('el 68 % son personas', True, WARM),
       (', así que responder «persona» a todo ya daría 68,5 %', False, MUTED)], False, 1.15)],
    0.3061, 0.3848, '← ahí cae el resultado: entre 0,306 y 0,385',
    [('Qué es ese margen. ', True, INK),
     ('Medimos 0,352 usando 612 secuencias de prueba. Con otras 612 distintas habría caído dentro '
      'de la franja verde: no es la escala de la métrica, que va de 0 a 1, sino cuánto se movería '
      'el resultado al repetir la medición.', False, MUTED)],
    [('Los 9.104 pasos: ', True, INK),
     ('cada paso ajusta el modelo con un lote de 64 imágenes, o sea unas 583.000 imágenes vistas, '
      'ocho vueltas al conjunto. Alcanzó su mejor punto en la época 3 de 8 y después empeoró: le '
      'sobró presupuesto.', False, MUTED)],
    'macro-F1: promedia el desempeño de cada especie por separado, con el mismo peso para todas. '
    'Va de 0 a 1; al azar entre 20 clases daría 0,05',
    'Fila "resultados preliminares ResNet" de la rubrica. Leer la lamina en este orden: primero '
    'las tres tarjetas, despues la barra. La barra responde la pregunta que siempre aparece: por '
    'que el resultado se escribe entre 0,306 y 0,385 si la metrica va de 0 a 1. Decirlo con estas '
    'palabras: "el margen no es el rango posible, es cuanto se moveria el numero si repitieramos '
    'la prueba con otras secuencias". Justificar el 0,35 si preguntan: son 20 clases, cuatro de '
    'ellas tipos de atun casi indistinguibles, y tres especies raras quedan en cero.')

lamina_modelo(
    13, 'Resultados preliminares · baseline 2', 'ViT-B/16, mismo presupuesto, mismo trato',
    [('Kaggle · macro-F1', '1,000',
      [('Las 1.336 imágenes, sin un solo error. ', False, MUTED),
       ('No es un logro, es un síntoma:', True, WARM),
       (' ese examen ya no distingue nada', False, MUTED)], False, 1.15),
     ('FishNet limpia · macro-F1', '0,393', 'Las mismas 612 secuencias que el ResNet', True, 1),
     ('FishNet · aciertos totales', '88,2 %', 'Misma advertencia: inflado por el 68 % de personas',
      False, 1)],
    0.3486, 0.4238, '← entre 0,349 y 0,424, algo más a la derecha que el ResNet',
    [('Las dos franjas se pisan', True, INK),
     (', y aun así el ViT gana. La comparación de la próxima lámina no enfrenta estas dos franjas '
      'sueltas: enfrenta a los dos modelos secuencia por secuencia sobre el mismo examen.', False, MUTED)],
    [('Los mismos 9.104 pasos', True, INK),
     (', el mismo lote de 64 imágenes y las mismas tres semillas. El ViT llegó a su mejor punto en '
      'la época 5 de 8, más tarde que el ResNet, y en una de las tres corridas seguía mejorando al '
      'agotarse el presupuesto.', False, MUTED)],
    'Mismo preprocesamiento, mismo tamaño de lote y pesos de partida de calidad pareja: '
    '80,9 contra 81,1 en ImageNet',
    'Fila "resultados preliminares ViT". Insistir en el "mismo trato": mismos pasos, mismo lote, '
    'mismas semillas, mismo preprocesamiento y pesos de partida equivalentes. Ese ultimo punto vale '
    'la pena explicarlo: la trampa mas comun es comparar el ViT contra la version antigua del '
    'ResNet, que arranca cinco puntos mas abajo. Sobre las franjas que se pisan: adelantarlo aqui '
    'evita que alguien lo use como objecion en la lamina siguiente, y la respuesta es que la '
    'comparacion es pareada, modelo contra modelo sobre las mismas secuencias.')

# ---------------------------------------------------------------- 14 comparacion
s = nueva(DARK)
cabecera(s, 'La comparación', 'En FishNet el ViT gana, y la diferencia se sostiene',
         TEALL, ONDARK, tsize=76, tw=1620)
for (cx, cw), (val, cue, vc) in zip(fila(128, 1664, [1.15, 1, 1.05], 32), [
        ('+0,042', 'de ventaja en macro-F1, con margen de +0,007 a +0,077. Como el margen no '
                   'llega al cero, la ventaja no es casualidad', TEALL),
        ('9 de 9', 'comparaciones entre semillas favorecen al ViT: su peor corrida supera a la '
                   'mejor del ResNet', ONDARK),
        ('+0,003', 'de diferencia en Kaggle, con margen de 0,000 a 0,011: ahí no hay diferencia '
                   'demostrable', ONDARK)]):
    caja(s, cx, 440, cw, 345, DCARD)
    texto(s, cx + 40, 476, cw - 80, 150, val, 110, vc, bold=True, font=HEAD, line=1.05)
    texto(s, cx + 40, 622, cw - 80, 120, cue, 28, DMUTED, line=1.4)
texto(s, 128, 790, 1620, 130,
      'Con más entrenamiento la ventaja sería probablemente mayor: el ResNet dejó de mejorar en la '
      'época 3 de 8, y el ViT siguió mejorando hasta la 5.', 32, 'D5DDDA', line=1.4)
pie(s, 'FishNet, prueba limpia · los dos modelos comparados sobre las mismas 612 secuencias', 14, DFOOT)
notas(s, 'El momento central de la presentacion: bajar el ritmo. Explicar que significa que el '
         'margen no llegue al cero: repetimos la comparacion miles de veces sorteando que '
         'secuencias entran en la prueba, y el ViT quedo arriba en casi todas; si el margen '
         'incluyera el cero, no podriamos afirmar que hay diferencia. El dato de las 9 de 9 '
         'semillas es el mas facil de entender: la peor corrida del ViT le gana a la mejor del '
         'ResNet. El tercer numero es honesto a proposito: en Kaggle no podemos afirmar que haya '
         'diferencia, y lo decimos.')

# ---------------------------------------------------------------- 15 extras
s = nueva(LIGHT)
cabecera(s, 'Dos hallazgos que van más allá del acierto',
         'Saber decir «no sé», y cuánto cuesta cada uno', TEAL, INK, th=100)
anchos = fila(128, 1664, [1, 1], 32)
cx, cw = anchos[0]
caja(s, cx, 330, cw, 500, WHITE, BORDER, barra=TEAL)
texto(s, cx + 44, 374, cw - 88, 60, 'Frente a una especie que nunca vio', 36, INK, bold=True, font=HEAD)
texto(s, cx + 44, 448, cw - 88, 110,
      'Le mostramos a los modelos fotos del otro conjunto, que nunca vieron. Con el mismo criterio '
      'de rechazo:', 29, MUTED, line=1.45)
texto(s, cx + 44, 580, cw - 88, 120,
      [[('el ', False, MUTED), ('ResNet acepta el 94 %', True, WARM), (' como si fueran suyas', False, MUTED)],
       [('el ', False, MUTED), ('ViT, solo el 49 %', True, TEAL)]], 29, MUTED, line=1.5)
texto(s, cx + 44, 740, cw - 88, 50, 'Es la diferencia más grande que encontramos.', 27, INK, bold=True)
cx, cw = anchos[1]
caja(s, cx, 330, cw, 500, WHITE, BORDER, barra=TEAL)
texto(s, cx + 44, 374, cw - 88, 60, 'Cuánto tarda en un PC sin GPU', 36, INK, bold=True, font=HEAD)
texto(s, cx + 44, 448, cw - 88, 110,
      [[('ResNet: ', False, MUTED), ('43 ms', True, INK), (' por pez', False, MUTED)],
       [('ViT: ', False, MUTED), ('119 ms', True, INK), (' por pez', False, MUTED)]], 29, MUTED, line=1.5)
texto(s, cx + 44, 580, cw - 88, 130,
      'Casi tres veces más lento, pero una foto entera igual se resuelve en medio segundo. Para '
      'fotos en el muelle, los dos alcanzan de sobra.', 29, MUTED, line=1.45)
texto(s, cx + 44, 740, cw - 88, 50, 'La velocidad no decide: elegimos el ViT.', 27, INK, bold=True)
pie(s, 'Medido en esta máquina: i5-12400F, 6 hilos · 200 repeticiones por escenario', 15, FOOT)
notas(s, 'Estos dos hallazgos no los pide la rubrica del avance, pero muestran que el trabajo va '
         'mas alla del minimo. El del "no se" se explica solo: un sistema en el muelle se va a '
         'topar con especies que nunca vio, y lo peligroso no es equivocarse sino equivocarse con '
         'seguridad, porque eso ensucia el registro sin que nadie se entere. El de velocidad cierra '
         'la decision de arquitectura con un criterio practico: como ninguna de las dos hace '
         'imposible el caso de uso, gana la que clasifica mejor.')

# ---------------------------------------------------------------- 16 estado
s = nueva(ALT)
cabecera(s, 'Dónde estamos', 'Lo hecho y lo que falta', TEAL, INK, th=100)
for (cx, cw), (tit, col, items) in zip(fila(128, 1664, [1.15, 1], 32), [
        ('Hecho', TEAL, ['Los dos baselines entrenados, con 3 semillas cada uno',
                         'Los cuatro problemas de los datos, medidos y mitigados',
                         'Auditoría de fuga y sobreajuste',
                         'Interpretabilidad: medimos en qué parte de la foto mira cada red',
                         'Demo que corre las dos arquitecturas en vivo']),
        ('Falta', WARM, ['Un tercer modelo, ViT-B/32, que cuesta lo mismo que el ResNet: si '
                         'mantiene la ventaja, cambia la decisión',
                         'Pulir la demo para la entrega final',
                         'Informe y data card en su versión definitiva'])]):
    caja(s, cx, 330, cw, 540, WHITE, BORDER, barra=col)
    texto(s, cx + 40, 372, cw - 80, 56, tit, 36, col, bold=True, font=HEAD)
    texto(s, cx + 40, 444, cw - 80, 320, [['•  ' + i] for i in items], 28, '3C4F53',
          line=1.45, space_after=16)
pie(s, 'Todo el código, los experimentos y los resultados están en el repositorio del grupo', 16, FOOT)
notas(s, 'Mostrar que el avance esta por delante de lo minimo pedido, sin exagerar. Si hay tiempo, '
         'mencionar que el codigo esta versionado desde el primer dia con commits incrementales, '
         'porque eso tambien se evalua en la entrega final. El experimento que falta se explica en '
         'una frase: hay una variante del ViT que cuesta lo mismo que el ResNet; si rinde parecido '
         'al ViT grande, seria la mejor opcion de las tres.')

# ---------------------------------------------------------------- 17 cierre
s = nueva(DARK)
texto(s, 128, 190, 1660, 40, 'PARA CERRAR', 26, TEALL, bold=True, font=HEAD, spc=2)
texto(s, 128, 270, 1620, 330,
      'Lo más valioso que encontramos no fue un modelo mejor: fueron los problemas de los datos',
      84, ONDARK, bold=True, font=HEAD, line=1.12)
texto(s, 128, 620, 1580, 160,
      'Un 99,7 % que no reconocía peces, un reparto oficial con fuga, y márgenes de error cinco '
      'veces más angostos de lo real. Nada de eso aparece si uno solo mira la accuracy.',
      36, DMUTED, line=1.45)
texto(s, 128, 810, 1000, 70, 'Gracias. ¿Preguntas?', 44, TEALL, bold=True)
pie(s, 'Caleta-Scan · Grupo G3 · ' + NOM, 17, DFOOT)
notas(s, 'Cerrar con la idea grande, no con un resumen de vinetas. Preguntas que conviene tener '
         'preparadas: 1) Por que el macro-F1 es tan bajo en FishNet? Son 20 clases, varias casi '
         'iguales, y el promedio parejo castiga fuerte los errores en las raras. 2) No sera que el '
         'ViT gana por tener mas parametros? Puede ser, y por eso falta el ViT-B/32, que tiene '
         'computo parecido al ResNet. 3) Sirve esto en Chile? Hoy no: hay que recolectar fotos de '
         'la caleta primero, y eso es parte del proyecto. 4) Como sabemos que no hay fuga? Esta la '
         'lamina de auditoria, con cuatro pruebas.')

import sys
destino = sys.argv[1]
prs.save(destino)
print('guardado:', destino, '|', len(prs.slides.__iter__.__self__._sldIdLst), 'laminas')
