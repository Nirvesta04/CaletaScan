"""
Construye los splits train/val/test de las dos fuentes del proyecto Caleta-Scan (G3).

PARTE A - Kaggle "A Large Scale Fish Dataset"
---------------------------------------------
Problema: cada clase trae 1000 imagenes que son rotaciones de angulo arbitrario y
espejados de solo ~30-50 fotos originales. Un split aleatorio pondria rotaciones
de la MISMA foto en train y en test, inflando la accuracy hasta cerca del 100%
sin que el modelo generalice nada.

Solucion: se recupera a que foto original pertenece cada imagen aumentada y se
parte a nivel de ORIGINAL, no de imagen. Todas las rotaciones de una foto caen
siempre en la misma particion.

Como se recupera el grupo: ver el docstring de group_kaggle(). Se cruzan dos
evidencias independientes, la estructura de bloques del dataset y un histograma
de color invariante a rotacion, y su nivel de acuerdo queda reportado como
validacion del agrupamiento.

PARTE B - Fishnet Open Images v1.0.0
------------------------------------
Ya trae split oficial train/val/test en el CSV, disenado para que la mitad del
test venga de camaras no vistas en entrenamiento. Se respeta tal cual: aqui solo
se materializa el manifiesto de recortes (un recorte por bounding box) con las
cajas saneadas y la propuesta de espacio de etiquetas.

Salidas
-------
data/splits/kaggle_split.csv    path, clase, group_id, split
data/splits/kaggle_groups.csv   validacion del agrupamiento por clase
data/splits/fishnet_crops.csv   path, caja, etiquetas, split
data/splits/README.md           resumen legible de lo generado
"""

import collections
import csv
import glob
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "splits")

KAGGLE = os.path.join(RAW, "Fish_Dataset", "Fish_Dataset")
NA = os.path.join(RAW, "NA_Fish_Dataset")
FISHNET = os.path.join(RAW, "fishnet")

SEED = 20260827
VAL_FRAC, TEST_FRAC = 0.15, 0.15

# NA_Fish_Dataset y Fish_Dataset escriben distinto algunos nombres de clase
NA_ALIAS = {
    "Gilt-Head Bream": "Gilt Head Bream",
    "Hourse Mackerel": "Horse Mackerel",
}

# Fishnet: clases que no son especies de pez (utiles como distractores/negativos)
NON_FISH = {"Human", "No fish", "Water", "Oil"}
# soporte minimo en train para que una clase l1 se modele aparte
MIN_TRAIN_BOXES = 100


# --------------------------------------------------------------------------
# PARTE A
# --------------------------------------------------------------------------

def color_hist(path, thumb=160):
    """Histograma RGB 8x8x8 sobre pixeles no-negros. Invariante a rotacion/espejo."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((thumb, thumb), Image.BILINEAR)
    a = np.asarray(im).reshape(-1, 3).astype(np.int32)
    a = a[a.sum(1) > 30]                      # fuera las esquinas negras de la rotacion
    if len(a) == 0:
        return np.zeros(512)
    q = a // 32
    h = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512).astype(np.float64)
    return h / h.sum()


def chi2(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2 /
            (A[:, None, :] + B[None, :, :] + 1e-9)).sum(2) / 2


def group_kaggle():
    """
    Recupera la foto original de cada imagen aumentada.

    Se combinan dos evidencias independientes:

    1. ESTRUCTURA. El dataset esta ordenado por original en bloques contiguos de
       tamano fijo: con G originales y N imagenes, el bloque k son los indices
       [round(k*N/G), round((k+1)*N/G)). Verificado empiricamente: en las clases
       de 50 originales los bloques son exactamente de 20 imagenes.

    2. IDENTIDAD. A cada bloque se le asigna el original de NA_Fish_Dataset mas
       cercano por histograma de color, por voto de mayoria dentro del bloque.
       El voto de mayoria absorbe los errores sueltos del emparejamiento.

    El acuerdo dentro de cada bloque (fraccion de imagenes que votan al ganador)
    se reporta como validacion: es alto solo si ambas evidencias coinciden.

    Si dos bloques votan al MISMO original se fusionan en un solo grupo. Puede
    ser que NA traiga fotos duplicadas o que el histograma no logre separarlas;
    en cualquiera de los dos casos fusionar es la opcion conservadora, porque
    mantenerlos separados arriesga dejar la misma foto en train y en test.
    """
    classes = sorted(d for d in os.listdir(KAGGLE)
                     if os.path.isdir(os.path.join(KAGGLE, d)))
    rows, report = [], []
    for cls in classes:
        aug = sorted(glob.glob(os.path.join(KAGGLE, cls, cls, "*.png")))
        na_dir = os.path.join(NA, NA_ALIAS.get(cls, cls))
        orig = sorted(glob.glob(os.path.join(na_dir, "*")))
        if not orig:
            sys.exit("no encuentro originales para %s en %s" % (cls, na_dir))

        with ThreadPoolExecutor(8) as pool:
            A = np.stack(list(pool.map(color_hist, aug)))
            N = np.stack(list(pool.map(color_hist, orig)))

        best = chi2(A, N).argmin(1)
        n, g = len(aug), len(orig)
        # bloques de tamano fijo ceil(n/g) con el ultimo truncado. Verificado:
        # 50 originales -> 50 bloques de 20; 30 originales (Trout) -> 29 bloques
        # de 34 y uno final de 14.
        block = -(-n // g)
        bounds = [min(k * block, n) for k in range(g + 1)]

        agree, assigned = [], []
        for k in range(g):
            seg = best[bounds[k]:bounds[k + 1]].tolist()
            winner, votes = collections.Counter(seg).most_common(1)[0]
            agree.append(votes / len(seg))
            assigned.append(winner)

        for k in range(g):
            gid = "%s__orig%03d" % (cls, assigned[k])
            for p in aug[bounds[k]:bounds[k + 1]]:
                rows.append({
                    "dataset": "kaggle",
                    "path": os.path.relpath(p, ROOT).replace("\\", "/"),
                    "clase": cls,
                    "group_id": gid,
                })

        agree = np.array(agree)
        n_groups = len(set(assigned))
        report.append({
            "clase": cls, "imagenes": n, "originales": g, "bloques": g,
            "grupos_finales": n_groups,
            "acuerdo_medio": round(float(agree.mean()), 4),
            "acuerdo_min": round(float(agree.min()), 4),
        })
        print("  %-20s %4d imgs  %2d bloques de %d  ->  %2d grupos  "
              "acuerdo %.3f (min %.3f)%s"
              % (cls, n, g, bounds[1] - bounds[0], n_groups,
                 agree.mean(), agree.min(),
                 "  [%d bloques fusionados]" % (g - n_groups) if g > n_groups else ""),
              flush=True)
    return rows, report


def split_by_group(rows):
    """Split estratificado por clase, particionando GRUPOS (no imagenes)."""
    rng = random.Random(SEED)
    by_class = collections.defaultdict(set)
    for r in rows:
        by_class[r["clase"]].add(r["group_id"])

    assign = {}
    for cls, groups in by_class.items():
        g = sorted(groups)
        rng.shuffle(g)
        n = len(g)
        n_test = max(1, round(n * TEST_FRAC))
        n_val = max(1, round(n * VAL_FRAC))
        for i, gid in enumerate(g):
            assign[gid] = ("test" if i < n_test
                           else "val" if i < n_test + n_val
                           else "train")
    for r in rows:
        r["split"] = assign[r["group_id"]]
    return rows


# --------------------------------------------------------------------------
# PARTE B
# --------------------------------------------------------------------------

def build_fishnet_crops():
    manifest = os.path.join(FISHNET, "subset_manifest.csv")
    labels = os.path.join(FISHNET, "foid_labels_v100.csv")
    if not os.path.exists(manifest):
        print("  [aviso] no hay subset de Fishnet descargado, se omite la parte B")
        return [], {}

    have = {r["img_id"] for r in csv.DictReader(open(manifest, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(labels, newline="", encoding="utf-8"))
            if r["img_id"] in have]

    # dimensiones reales (PIL lee solo la cabecera del JPEG)
    dims = {}
    def dim(img_id):
        p = os.path.join(FISHNET, "images", img_id + ".jpg")
        with Image.open(p) as im:
            return im.size
    with ThreadPoolExecutor(16) as pool:
        for img_id, wh in zip(have, pool.map(dim, have)):
            dims[img_id] = wh

    # soporte por clase en train para decidir el espacio de etiquetas
    train_count = collections.Counter(
        r["label_l1"] for r in rows if r["train"] == "True")
    modeled = {c for c, n in train_count.items() if n >= MIN_TRAIN_BOXES}

    out, dropped = [], collections.Counter()
    for r in rows:
        w, h = dims[r["img_id"]]
        x0, x1 = sorted((int(r["x_min"]), int(r["x_max"])))
        y0, y1 = sorted((int(r["y_min"]), int(r["y_max"])))
        # el CSV trae coordenadas fuera de rango (p.ej. y_min=-1): se recortan
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 24 or y1 - y0 < 24:
            dropped["caja_muy_chica"] += 1
            continue
        split = ("train" if r["train"] == "True"
                 else "val" if r["val"] == "True" else "test")
        lab = r["label_l1"]
        out.append({
            "dataset": "fishnet",
            "path": "data/raw/fishnet/images/%s.jpg" % r["img_id"],
            "img_id": r["img_id"], "cam_id": r["cam_id"], "bbox_id": r["bbox_id"],
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "label_l1": lab,
            "label_l2": r["label_l2"],
            "label_model": lab if lab in modeled else "Other_rare",
            "is_fish": int(lab not in NON_FISH),
            "split": split,
        })
    return out, {"dropped": dict(dropped), "modeled": sorted(modeled),
                 "train_count": train_count}


# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)

    print("[A] Kaggle: recuperando grupos de aumentacion...", flush=True)
    krows, kreport = group_kaggle()
    krows = split_by_group(krows)

    with open(os.path.join(OUT, "kaggle_split.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["dataset", "path", "clase", "group_id", "split"])
        w.writeheader()
        w.writerows(krows)

    with open(os.path.join(OUT, "kaggle_groups.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["clase", "imagenes", "originales", "bloques",
                                "grupos_finales", "acuerdo_medio", "acuerdo_min"])
        w.writeheader()
        w.writerows(kreport)

    ks = collections.Counter(r["split"] for r in krows)
    kg = {s: len({r["group_id"] for r in krows if r["split"] == s})
          for s in ("train", "val", "test")}
    print("\n  imagenes: %s" % dict(ks))
    print("  grupos:   %s" % kg)

    # verificacion dura: ningun grupo puede cruzar particiones
    splits_of = collections.defaultdict(set)
    for r in krows:
        splits_of[r["group_id"]].add(r["split"])
    cross = [g for g, ss in splits_of.items() if len(ss) > 1]
    print("  grupos que cruzan particiones: %d %s"
          % (len(cross), "(correcto)" if not cross else "<-- ERROR"))

    print("\n[B] Fishnet: manifiesto de recortes...", flush=True)
    frows, finfo = build_fishnet_crops()
    if frows:
        cols = ["dataset", "path", "img_id", "cam_id", "bbox_id", "x0", "y0", "x1", "y1",
                "label_l1", "label_l2", "label_model", "is_fish", "split"]
        with open(os.path.join(OUT, "fishnet_crops.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(frows)
        fs = collections.Counter(r["split"] for r in frows)
        print("  recortes: %s" % dict(fs))
        print("  descartados: %s" % finfo["dropped"])
        print("  clases modeladas aparte (>=%d cajas en train): %d"
              % (MIN_TRAIN_BOXES, len(finfo["modeled"])))
        print("  resto colapsado en 'Other_rare'")

    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(render_readme(krows, kreport, frows, finfo))
    print("\n[listo] %s" % OUT)


def render_readme(krows, kreport, frows, finfo):
    L = ["# Splits generados", "",
         "Generado por `scripts/build_splits.py` (semilla %d)." % SEED, "",
         "## Kaggle - A Large Scale Fish Dataset", "",
         "Split a nivel de **foto original**, no de imagen: las ~34 rotaciones y",
         "espejados de una misma foto caen siempre en la misma particion.", "",
         "| clase | imgs | bloques | grupos | acuerdo medio | acuerdo min |",
         "|---|---|---|---|---|---|"]
    for r in kreport:
        L.append("| %s | %d | %d | %d | %.3f | %.3f |" % (
            r["clase"], r["imagenes"], r["bloques"], r["grupos_finales"],
            r["acuerdo_medio"], r["acuerdo_min"]))
    ks = collections.Counter(r["split"] for r in krows)
    L += ["", "Imagenes por particion: %s" % dict(ks), "",
          "`acuerdo` = fraccion de imagenes del bloque que votan al mismo original",
          "por histograma de color. Es una verificacion independiente: la estructura",
          "de bloques y el histograma no comparten informacion, asi que solo",
          "coinciden si ambas identifican la misma foto de origen.", ""]
    if frows:
        fs = collections.Counter(r["split"] for r in frows)
        L += ["## Fishnet Open Images v1.0.0", "",
              "Split **oficial** del CSV, respetado sin cambios (mitad del test",
              "proviene de camaras no vistas en entrenamiento).", "",
              "Recortes por particion: %s" % dict(fs), "",
              "Columnas de etiqueta: `label_l1` (especie), `label_l2` (grupo grueso),",
              "`label_model` (l1 con la cola colapsada en `Other_rare`), `is_fish`",
              "(0 para Human/No fish/Water/Oil).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
