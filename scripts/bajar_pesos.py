"""
Baja los pesos de los cinco modelos para correr la demo sin entrenar.

    python -m scripts.bajar_pesos          # ~742 MB, luego: python -m src.demo

Descarga caleta-scan-pesos.zip del release v1.0 del repositorio y lo descomprime
en la raiz. Trae, por modelo, el punto de control de la semilla 0 y los logits de
validacion y prueba de las tres semillas: la demo los usa para fijar el ajuste tau
y el umbral del «no sé», igual que en el informe. No trae imagenes de FishNet: sin
data/raw/ la demo no muestra fotos de ejemplo, pero se puede subir una foto.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://github.com/Nirvesta04/CaletaScan/releases/download/v1.0/caleta-scan-pesos.zip"
SHA256 = "b34766b66f3dc65264f67fd39fe063159f72e2fe0915cbfba8681dae8e09e03c"
DESTINO = os.path.join(ROOT, "runs", "caleta-scan-pesos.zip")


_ultimo = [-1]


def progreso(bloques, tam_bloque, total):
    if total <= 0:
        return
    hecho = min(bloques * tam_bloque, total)
    pct = int(100 * hecho / total)
    if pct != _ultimo[0]:                  # una linea por punto porcentual, no por bloque
        _ultimo[0] = pct
        sys.stdout.write("\r  %3d %%  (%d de %d MB)" % (pct, hecho >> 20, total >> 20))
        sys.stdout.flush()


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for trozo in iter(lambda: fh.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def main():
    if not os.path.exists(DESTINO) or sha256(DESTINO) != SHA256:
        print("bajando", URL)
        urllib.request.urlretrieve(URL, DESTINO, progreso)
        print()
        if sha256(DESTINO) != SHA256:
            sys.exit("el archivo bajado no coincide con el esperado; borrarlo y reintentar")
    with zipfile.ZipFile(DESTINO) as z:
        for nombre in z.namelist():              # solo rutas runs/<exp>/seed<n>/...
            if not nombre.startswith("runs/") or ".." in nombre:
                sys.exit("ruta inesperada en el zip: %s" % nombre)
        z.extractall(ROOT)
        print("listo: %d archivos en runs/" % len(z.namelist()))
    os.remove(DESTINO)
    print("ahora: python -m src.demo   (http://127.0.0.1:8000)")


if __name__ == "__main__":
    main()
