"""
Descarga selectiva del Fishnet Open Images Database v1.0.0 (The Nature Conservancy).

Motivacion
----------
El ZIP completo de imagenes pesa 36,4 GB (143.818 imagenes). Sobre el enlace de
trabajo (~1,2 MB/s medidos) eso son ~8,5 horas. El S3 acepta HTTP range requests
(Accept-Ranges: bytes), asi que se descargan SOLO los miembros necesarios del ZIP
remoto, sin traer el archivo completo.

Criterio de seleccion (definido A PRIORI, antes de entrenar nada)
----------------------------------------------------------------
Se incluye toda imagen que contenga al menos una caja de una clase de COLA, es
decir, cualquier clase fuera de las 6 clases cabeza sobre-representadas:

    Human, Yellowfin tuna, Albacore, Skipjack tuna, Bigeye tuna, No fish

Mas todas las imagenes que contienen la clase "No fish" (unica que se perdia con
la regla anterior).

Consecuencias del criterio, verificadas sobre foid_labels_v100.csv:
  * se conserva el 100% de las cajas de las 28 clases de cola;
  * las clases cabeza quedan submuestreadas, no eliminadas: llegan igual
    ~10.043 aleta amarilla, ~6.851 albacora, ~3.023 barrilete, ~53.784 humano,
    por aparecer en las mismas fotos que la cola;
  * el split oficial train/val/test del CSV NO se altera: se respeta tal cual;
  * total: 20.002 imagenes (~4,84 GB) = 13,9% de los bytes, con 97.777 cajas.

Lo que se descarta son fotos redundantes de atun y tripulacion pertenecientes a
clases que ya tienen decenas de miles de ejemplos. Es submuestreo de clases
cabeza, declarado en la data card. NO se filtra el conjunto de test en funcion
del desempeno del modelo.

Licencia del dataset: CC BY-SA 4.0.
Cita: The Nature Conservancy (2022): Fishnet Open Images Dataset v1.0.0.
"""

import csv
import collections
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

ZIP_URL = "https://fishnet-data.s3-us-west-1.amazonaws.com/foid_images_v100.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "raw", "fishnet")
LABELS = os.path.join(DATA, "foid_labels_v100.csv")
OUT_DIR = os.path.join(DATA, "images")
INDEX_CACHE = os.path.join(DATA, "zip_index_subset.json")
MANIFEST = os.path.join(DATA, "subset_manifest.csv")

HEAD_CLASSES = {
    "Human", "Yellowfin tuna", "Albacore",
    "Skipjack tuna", "Bigeye tuna", "No fish",
}

WORKERS = 12
RETRIES = 5


def select_image_ids():
    """Aplica el criterio de seleccion sobre el CSV de etiquetas."""
    by_img = collections.defaultdict(list)
    with open(LABELS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_img[row["img_id"]].append(row)

    keep = set()
    for img_id, boxes in by_img.items():
        labels = {b["label_l1"] for b in boxes}
        if labels - HEAD_CLASSES:          # contiene alguna clase de cola
            keep.add(img_id)
        elif "No fish" in labels:          # rescate de la clase No fish
            keep.add(img_id)
    return keep, by_img


def fetch_range(start, length):
    """Un range request con reintentos y backoff."""
    end = start + length - 1
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                ZIP_URL, headers={"Range": "bytes=%d-%d" % (start, end)}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError("range %d-%d fallo tras %d intentos: %s" % (start, end, RETRIES, last))


class HttpFile:
    """Objeto tipo archivo sobre range requests, solo para leer el indice del ZIP."""

    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def seek(self, offset, whence=0):
        self.pos = (offset if whence == 0
                    else self.pos + offset if whence == 1
                    else self.size + offset)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        n = min(n, self.size - self.pos)
        data = fetch_range(self.pos, n)
        self.pos += len(data)
        return data


def build_index(keep_ids):
    """Lee el directorio central del ZIP remoto y cachea offsets de los miembros pedidos."""
    wanted = set("images/%s.jpg" % i for i in keep_ids)
    if os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE, encoding="utf-8") as fh:
            cached = json.load(fh)
        if set(cached) >= wanted:
            print("[index] cache reutilizada: %d miembros" % len(cached))
            return cached

    print("[index] leyendo directorio central del ZIP remoto (~10 MB)...", flush=True)
    t0 = time.time()
    zf = zipfile.ZipFile(HttpFile(ZIP_URL))
    index = {}
    for info in zf.infolist():
        if info.filename in wanted:
            index[info.filename] = {
                "header_offset": info.header_offset,
                "compress_size": info.compress_size,
                "file_size": info.file_size,
                "compress_type": info.compress_type,
            }
    missing = len(wanted) - len(index)
    print("[index] %d miembros localizados en %.0fs%s"
          % (len(index), time.time() - t0,
             " (%d no existen en el ZIP)" % missing if missing else ""), flush=True)
    with open(INDEX_CACHE, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return index


def extract_member(name, meta):
    """Descarga un miembro del ZIP con UN solo range request y lo escribe a disco."""
    out_path = os.path.join(OUT_DIR, os.path.basename(name))
    if os.path.exists(out_path) and os.path.getsize(out_path) == meta["file_size"]:
        return 0  # ya estaba: reanudable

    # cabecera local (30 B fijos + nombre + extra) + datos comprimidos.
    # No conocemos nombre/extra de antemano, se sobre-pide 1 KB.
    pad = 1024
    blob = fetch_range(meta["header_offset"], 30 + pad + meta["compress_size"])
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError("cabecera local invalida en %s" % name)
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    payload = blob[start:start + meta["compress_size"]]

    if meta["compress_type"] == zipfile.ZIP_STORED:
        data = payload
    else:
        data = zlib.decompress(payload, -15)

    if len(data) != meta["file_size"]:
        raise RuntimeError("tamano inesperado en %s" % name)
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError("%s no es un JPEG valido" % name)

    tmp = out_path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, out_path)
    return len(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    keep_ids, by_img = select_image_ids()
    print("[seleccion] %d imagenes cumplen el criterio" % len(keep_ids), flush=True)

    index = build_index(keep_ids)
    todo = sorted(index)

    # manifiesto: que se incluyo y por que
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["img_id", "split", "n_boxes", "labels_l1"])
        for name in todo:
            img_id = os.path.basename(name)[:-4]
            boxes = by_img[img_id]
            row = boxes[0]
            split = ("train" if row["train"] == "True"
                     else "val" if row["val"] == "True" else "test")
            w.writerow([img_id, split, len(boxes),
                        "|".join(sorted(set(b["label_l1"] for b in boxes)))])
    print("[manifiesto] %s" % MANIFEST, flush=True)

    total = len(todo)
    lock = threading.Lock()
    state = {"n": 0, "bytes": 0, "skipped": 0, "errors": []}
    t0 = time.time()

    def work(name):
        try:
            got = extract_member(name, index[name])
        except Exception as exc:
            with lock:
                state["errors"].append((name, str(exc)))
            return
        with lock:
            state["n"] += 1
            state["bytes"] += got
            if got == 0:
                state["skipped"] += 1
            n = state["n"]
            if n % 250 == 0 or n == total:
                el = time.time() - t0
                mb = state["bytes"] / 1e6
                rate = mb / el if el else 0
                eta = (total - n) / (n / el) / 60 if n else 0
                print("[%d/%d] %.0f MB  %.2f MB/s  ETA %.0f min  errores=%d"
                      % (n, total, mb, rate, eta, len(state["errors"])), flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(work, todo))

    el = (time.time() - t0) / 60
    print("\n[listo] %d/%d imagenes en %.1f min (%.2f GB descargados, %d ya estaban)"
          % (state["n"], total, el, state["bytes"] / 1e9, state["skipped"]), flush=True)
    if state["errors"]:
        print("[errores] %d: vuelve a correr el script, es reanudable" % len(state["errors"]))
        for name, err in state["errors"][:10]:
            print("   ", name, err)
        sys.exit(1)


if __name__ == "__main__":
    main()
