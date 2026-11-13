"""
Como se lee cada comparacion de pares. Un solo lugar para el mazo y el informe,
para que digan exactamente lo mismo.

La regla: "se distingue" si el margen al 95 % de la diferencia no incluye el
cero. Si no se distingue, se dice eso y nada mas: no distinguirse no prueba que
los modelos sean iguales.
"""

# (gana el transformer, no se distinguen, gana la CNN)
TEXTOS = {
    'ViT-B/16 - ResNet-50': (
        'El ViT-B/16 supera al ResNet-50.',
        'El ViT-B/16 y el ResNet-50 no se distinguen.',
        'El ResNet-50 supera al ViT-B/16.'),
    'DeiT-S - ResNet-50': (
        'Un transformer del tamaño del ResNet-50 también lo supera: la ventaja no depende del tamaño.',
        'Con el mismo tamaño, transformer y ResNet-50 no se distinguen: no podemos afirmar que la '
        'ventaja del ViT-B/16 se mantenga en un transformer pequeño.',
        'Con el mismo tamaño gana el ResNet-50: la ventaja del ViT-B/16 no se mantiene en un '
        'transformer pequeño.'),
    'Swin-T - ConvNeXt-T': (
        'Entre dos modelos modernos del mismo tamaño también gana el transformer.',
        'Entre dos modelos modernos del mismo tamaño, CNN y transformer no se distinguen.',
        'Entre dos modelos modernos del mismo tamaño gana la CNN.'),
}

ETIQUETA = {'ViT-B/16 - ResNet-50': 'Par original',
            'DeiT-S - ResNet-50': 'Mismo tamaño',
            'Swin-T - ConvNeXt-T': 'Par moderno'}


def lectura(clave, d):
    """Texto de la comparacion `clave` con su resultado `d`, o None si falta."""
    if d is None:
        return None
    gana, empata, pierde = TEXTOS[clave]
    if not d['distinguible']:
        return empata
    return gana if d['delta_medio'] > 0 else pierde
