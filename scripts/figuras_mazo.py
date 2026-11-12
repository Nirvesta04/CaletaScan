"""
Imagenes de los datos para la presentacion final.

    python -m scripts.figuras_mazo

Escribe en docs/figs_privadas/, que NO se versiona: son fotos de Kaggle, FishNet
e iNaturalist, y las de FishNet muestran tripulantes. FishNet ya publica las
caras pixeladas, pero no siempre: aqui se difumina ademas la parte de arriba de
cada caja `Human`. Cada imagen que va al mazo se revisa a mano igual.
"""

from __future__ import annotations

import csv
import os
import uuid

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figs_privadas")
SPLITS = os.path.join(ROOT, "data", "splits")


def fuente(n):
    for f in ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def difuminar_cabezas(img, filas, frac=0.40):
    """Difumina el tercio superior (40 %) de cada caja Human, donde va la
    cabeza. FishNet ya pixela las caras, pero no siempre; tapar la caja entera
    borraria tambien los peces que la persona tiene en las manos."""
    for _, r in filas[filas["label_model"] == "Human"].iterrows():
        w, h = r.x1 - r.x0, r.y1 - r.y0
        b = (max(0, int(r.x0 - .10 * w)), max(0, int(r.y0 - .10 * h)),
             min(img.width, int(r.x1 + .10 * w)), min(img.height, int(r.y0 + frac * h)))
        if b[2] > b[0] and b[3] > b[1]:
            img.paste(img.crop(b).filter(ImageFilter.GaussianBlur(22)), b[:2])
    return img


def cajas(img, filas, color=(92, 192, 176), grosor=5):
    d = ImageDraw.Draw(img)
    for _, r in filas[filas["is_fish"] == 1].iterrows():
        d.rectangle([r.x0, r.y0, r.x1, r.y1], outline=color, width=grosor)
    return img


def foto_fishnet(df, img_id, ancho=900, con_cajas=True):
    f = df[df["img_id"] == img_id]
    img = Image.open(os.path.join(ROOT, f["path"].iloc[0])).convert("RGB")
    img = difuminar_cabezas(img, f)
    if con_cajas:
        img = cajas(img, f)
    return img.resize((ancho, int(img.height * ancho / img.width)), Image.LANCZOS)


def kaggle_grilla():
    k = pd.read_csv(os.path.join(SPLITS, "kaggle_split.csv"))
    clases = sorted(k["clase"].unique())
    t, pad, cab = 360, 16, 54
    lienzo = Image.new("RGB", (5 * t + 6 * pad, 2 * (t + cab) + 3 * pad), "white")
    d = ImageDraw.Draw(lienzo)
    for i, c in enumerate(clases):
        p = k[k["clase"] == c]["path"].iloc[0]
        im = Image.open(os.path.join(ROOT, p)).convert("RGB")
        im = im.resize((t, int(t * im.height / im.width)))
        x = pad + (i % 5) * (t + pad)
        y = pad + (i // 5) * (t + cab + pad)
        lienzo.paste(im, (x, y))
        d.text((x, y + im.height + 6), c, fill=(22, 40, 44), font=fuente(26))
    lienzo.save(os.path.join(OUT, "kaggle_especies.jpg"), quality=90)


def kaggle_borrado():
    """Original, sin pez y solo pez: el experimento E7 en una imagen."""
    from src.data import FILL, gt_path
    k = pd.read_csv(os.path.join(SPLITS, "kaggle_split.csv"))
    p = k[(k["clase"] == "Red Mullet") & (k["split"] == "test")]["path"].iloc[0]
    im = Image.open(os.path.join(ROOT, p)).convert("RGB")
    m = np.asarray(Image.open(os.path.join(ROOT, gt_path(p))).convert("L").resize(im.size)) > 127
    a = np.asarray(im)
    sin_pez, solo_pez = a.copy(), a.copy()
    sin_pez[m] = FILL
    solo_pez[~m] = FILL
    for nombre, arr in (("kaggle_original", a), ("kaggle_sin_pez", sin_pez),
                        ("kaggle_solo_pez", solo_pez)):
        Image.fromarray(arr).save(os.path.join(OUT, nombre + ".jpg"), quality=90)


def tiempo(img_id):
    try:
        return uuid.UUID(img_id).time
    except ValueError:
        return 0


def fishnet_secuencia(df, seq=185, n=4):
    """Fotogramas consecutivos de una secuencia que el split oficial reparte
    entre train y test: el riesgo de fuga, visto."""
    f = df[df["seq_id"] == seq].drop_duplicates("img_id").copy()
    f["t"] = f["img_id"].map(tiempo)
    f = f.sort_values("t")
    # cuatro fotogramas seguidos que incluyan train y test
    ids, splits = f["img_id"].tolist(), f["split"].tolist()
    ini = 0
    for i in range(len(ids) - n + 1):
        if {"train", "test"} <= set(splits[i:i + n]):
            ini = i
            break
    for j, (iid, sp) in enumerate(zip(ids[ini:ini + n], splits[ini:ini + n])):
        foto_fishnet(df, iid, 800).save(
            os.path.join(OUT, "fishnet_seq_%d_%s.jpg" % (j + 1, sp)), quality=88)
    print("secuencia %d: %s" % (seq, splits[ini:ini + n]))


def fishnet_ejemplos(df, n=4):
    """Fotos del test limpio con muchos peces y oclusion: la dificultad real."""
    g = df[df["split"] == "test"].groupby("img_id").agg(
        peces=("is_fish", "sum"), occ=("occ_union", "mean"), cam=("cam_id", "first"))
    g = g[g["peces"] >= 6].sort_values("occ", ascending=False)
    elegidas = g.drop_duplicates("cam").head(n).index
    for j, iid in enumerate(elegidas):
        foto_fishnet(df, iid, 900).save(os.path.join(OUT, "fishnet_ejemplo_%d.jpg" % (j + 1)),
                                        quality=88)


def chile():
    with open(os.path.join(ROOT, "data", "chile_creditos.csv"), encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    creditos = []
    for esp in ("reineta", "merluza", "congrio"):
        r = [f for f in filas if f["especie"] == esp][0]
        im = Image.open(os.path.join(ROOT, r["archivo"])).convert("RGB")
        lado = min(im.size)   # cuadrado central, para que las tres midan lo mismo
        im = im.crop(((im.width - lado) // 2, (im.height - lado) // 2,
                      (im.width + lado) // 2, (im.height + lado) // 2))
        im.save(os.path.join(OUT, "chile_%s.jpg" % esp), quality=90)
        creditos.append("%s: %s" % (esp, r["atribucion"]))
    with open(os.path.join(OUT, "chile_creditos.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(creditos))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(SPLITS, "fishnet_crops.csv"))
    kaggle_grilla()
    kaggle_borrado()
    fishnet_secuencia(df)
    fishnet_ejemplos(df)
    chile()
    print("listo: %s" % OUT)


if __name__ == "__main__":
    main()
