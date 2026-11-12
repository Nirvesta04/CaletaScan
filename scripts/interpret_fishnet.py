"""
Interpretabilidad en FishNet: ¿cuanto de lo que mira cada red cae sobre el pez?

    python -m scripts.interpret_fishnet

El pointing game de src/interpret.py usa las mascaras de Kaggle, y Kaggle es
justo donde el fondo es un atajo. FishNet no trae mascaras, pero si la caja de
cada pez. Se usa la caja como region del pez:

1. Se recorta una ventana del doble de ancho y alto que la caja (el
   entrenamiento usa 10 % de margen), con el pez en una POSICION AL AZAR dentro
   de la ventana. La caja ocupa alrededor de un cuarto del recorte y queda fondo
   real alrededor: cubierta, redes, otros peces. Con el pez siempre al centro,
   una gaussiana fija centrada dejaba 0,76 de su energia en la caja, mas que
   cualquier metodo real: la medida no distinguia nada.
2. Sobre los MISMOS recortes se calculan Grad-CAM (ResNet-50) y attention
   rollout y relevancia de Chefer (ViT-B/16).
3. Se mide que fraccion de la energia de cada mapa cae dentro de la caja.
   Tres referencias: el area de la caja (lo que deja un mapa uniforme), una
   gaussiana fija centrada, y el CONTROL BARAJADO: la energia del mapa de un
   recorte dentro de la caja de OTRO recorte. El barajado conserva la tendencia
   espacial propia de cada metodo (Grad-CAM es borroso y central; el rollout es
   casi plano) y solo quita la relacion con el pez. Lo que un metodo supera a su
   propio barajado es lo que realmente sigue al pez.

Como el margen ancho no es el que vieron en el entrenamiento, tambien se reporta
cuantas veces cada modelo predice lo mismo que con el recorte normal.

Las figuras de ejemplo se guardan en docs/figs_privadas/, que no se versiona:
son fotos de FishNet y pueden mostrar tripulantes.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

from src import data
from src.interpret import (energy_inside, gradcam, load_ckpt, verificar_atencion,
                           vit_maps)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARGEN = 0.50
POR_ESPECIE = 40
SEMILLA = 0


def caja_con_margen(box, size, margen, u=(0.5, 0.5)):
    """Ventana de (1 + 2*margen) veces la caja. `u` en [0,1]^2 fija donde cae el
    pez dentro de ella: (0,5, 0,5) es centrado, como en el entrenamiento. La
    ventana se desplaza (no se recorta) para no salirse de la imagen."""
    W, H = size
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    ww, wh = min(W, bw * (1 + 2 * margen)), min(H, bh * (1 + 2 * margen))
    wx0 = x0 - u[0] * (ww - bw)
    wy0 = y0 - u[1] * (wh - bh)
    wx0 = min(max(0, wx0), W - ww)
    wy0 = min(max(0, wy0), H - wh)
    return (int(round(wx0)), int(round(wy0)), int(round(wx0 + ww)), int(round(wy0 + wh)))


def mascara_caja(box, recorte, n=data.IMG_SIZE):
    """La caja del pez, en coordenadas del recorte ya deformado a n x n."""
    cx0, cy0, cx1, cy1 = recorte
    sx, sy = n / (cx1 - cx0), n / (cy1 - cy0)
    x0, y0, x1, y1 = box
    m = np.zeros((n, n), dtype=bool)
    m[max(0, int(round((y0 - cy0) * sy))):int(round((y1 - cy0) * sy)),
      max(0, int(round((x0 - cx0) * sx))):int(round((x1 - cx0) * sx))] = True
    return m


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dss = data.fishnet(label_col="label_model", purge=False)
    tr, te = dss["train"], dss["test"]
    keep = tr.species_mask()
    limpio = data.clean_mask(tr, te)

    # muestra: hasta POR_ESPECIE cajas por especie del test limpio, al azar fijo
    rng = np.random.default_rng(0)
    filas = []
    for k in np.flatnonzero(keep):
        cand = np.flatnonzero(limpio & (te.labels == k))
        if len(cand):
            filas += list(rng.choice(cand, min(POR_ESPECIE, len(cand)), replace=False))
    filas = np.array(sorted(filas))
    print("muestra: %d cajas de %d especies" % (len(filas), len(set(te.labels[filas]))))

    K = len(tr.classes)
    rn = load_ckpt("e4_resnet50_fishnet", SEMILLA, "resnet50", K, dev)
    vt = load_ckpt("e5_vit_b_16_fishnet", SEMILLA, "vit_b_16", K, dev)
    tf = data.build_transform(train=False)

    # posicion al azar del pez en la ventana, fija por caja
    pos = {int(j): tuple(rng.uniform(0, 1, 2)) for j in filas}

    def cargar(j, margen, centrado=False):
        img = Image.open(os.path.join(ROOT, te.paths[j])).convert("RGB")
        rec = caja_con_margen(te.boxes[j], img.size, margen,
                              (0.5, 0.5) if centrado else pos[int(j)])
        return img.crop(rec), rec

    x0 = torch.stack([tf(cargar(j, MARGEN)[0]) for j in filas[:8]]).to(dev)
    v = verificar_atencion(vt, x0)
    print("forward parcheado del ViT: max|dif| %.1e -> %s"
          % (v["max_dif_logits"], "OK" if v["ok"] else "FALLA"))
    if not v["ok"]:
        raise SystemExit("el forward parcheado no reproduce el original")

    n = data.IMG_SIZE
    yy, xx = np.mgrid[0:n, 0:n]
    # gaussiana del ancho tipico de un mapa de Grad-CAM a 7x7 (sigma = n/6)
    gauss = np.exp(-(((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * (n / 6) ** 2)))
    res = {k: [] for k in ("gradcam", "rollout", "chefer", "gauss")}
    chicos = {k: [] for k in ("gradcam", "rollout", "chefer")}   # mapas a 56x56
    mchicas = []
    base, igual_r, igual_v, ok_r, ok_v = [], [], [], [], []
    ejemplos = []
    bs = 8
    for i0 in range(0, len(filas), bs):
        idx = filas[i0:i0 + bs]
        anchos = [cargar(j, MARGEN) for j in idx]
        xs = torch.stack([tf(a) for a, _ in anchos]).to(dev)
        xn = torch.stack([tf(cargar(j, data.CROP_MARGIN, centrado=True)[0]) for j in idx]).to(dev)
        masks = [mascara_caja(te.boxes[j], rec) for j, (_, rec) in zip(idx, anchos)]
        base += [float(m.mean()) for m in masks]
        y = torch.as_tensor(te.labels[idx], device=dev)

        with torch.no_grad():
            pr, pv = rn(xs).argmax(1), vt(xs).argmax(1)
            igual_r += (pr == rn(xn).argmax(1)).tolist()
            igual_v += (pv == vt(xn).argmax(1)).tolist()
        ok_r += (pr == y).tolist()
        ok_v += (pv == y).tolist()

        cams = gradcam(rn, xs, pr)
        roll, che = vit_maps(vt, xs, pv)
        for k, arr in (("gradcam", cams), ("rollout", roll), ("chefer", che),
                       ("gauss", [gauss] * len(masks))):
            res[k] += [energy_inside(a, m) for a, m in zip(arr, masks)]
        for k, arr in (("gradcam", cams), ("rollout", roll), ("chefer", che)):
            chicos[k] += [a.reshape(56, 4, 56, 4).mean((1, 3)).astype(np.float32) for a in arr]
        mchicas += [m.reshape(56, 4, 56, 4).mean((1, 3)) > 0.5 for m in masks]
        if len(ejemplos) < 12 and i0 % 48 == 0:
            ejemplos.append((idx[0], anchos[0][0], masks[0], cams[0], che[0],
                             tr.classes[te.labels[idx[0]]]))
        if (i0 // bs) % 10 == 0:
            print("  %d/%d" % (i0, len(filas)), flush=True)

    b = float(np.mean(base))
    out = {"n": int(len(filas)), "margen": MARGEN, "semilla": SEMILLA,
           "linea_base_area_caja": b,
           "misma_prediccion_que_recorte_normal": {"resnet": float(np.mean(igual_r)),
                                                   "vit": float(np.mean(igual_v))},
           "acierto_en_recorte_ancho": {"resnet": float(np.mean(ok_r)),
                                        "vit": float(np.mean(ok_v))},
           "energia_en_la_caja": {}}
    print("\nlinea base (area de la caja): %.3f" % b)
    out["linea_base_gaussiana_centrada"] = float(np.nanmean(res["gauss"]))
    print("linea base gaussiana centrada: %.3f" % out["linea_base_gaussiana_centrada"])
    estable = {"gradcam": np.array(igual_r), "rollout": np.array(igual_v),
               "chefer": np.array(igual_v)}
    for k, et in (("gradcam", "ResNet-50 Grad-CAM"), ("rollout", "ViT-B/16 rollout"),
                  ("chefer", "ViT-B/16 Chefer")):
        todo = np.array(res[k])
        # solo las cajas donde el margen ancho NO cambio la prediccion del modelo:
        # ahi el mapa explica la misma decision que el modelo toma normalmente
        est = todo[estable[k]]
        est = est[~np.isnan(est)]
        a = todo[~np.isnan(todo)]
        # control barajado: mapa i contra la caja de otro recorte, 20 barajados
        bar = []
        for _ in range(20):
            perm = rng.permutation(len(mchicas))
            bar.append(np.nanmean([energy_inside(chicos[k][i], mchicas[perm[i]])
                                   for i in range(len(mchicas)) if perm[i] != i]))
        barajado = float(np.mean(bar))
        # IC por bootstrap simple sobre cajas (la muestra esta estratificada por
        # especie y casi no repite secuencias; es orientativo)
        bt = [rng.choice(a, len(a)).mean() for _ in range(2000)]
        out["energia_en_la_caja"][k] = {"media": float(a.mean()), "veces_linea_base": float(a.mean() / b),
                                        "ic95": [float(np.percentile(bt, 2.5)),
                                                 float(np.percentile(bt, 97.5))],
                                        "control_barajado": barajado,
                                        "sobre_barajado": float(a.mean() - barajado),
                                        "media_prediccion_estable": float(est.mean()),
                                        "n_prediccion_estable": int(len(est))}
        print("  %-20s %.3f | barajado %.3f | sobre barajado %+.3f | prediccion estable %.3f (n=%d)"
              % (et, a.mean(), barajado, a.mean() - barajado, est.mean(), len(est)))
    print("misma prediccion que con el recorte normal: ResNet %.0f %%, ViT %.0f %%"
          % (100 * np.mean(igual_r), 100 * np.mean(igual_v)))

    with open(os.path.join(ROOT, "runs", "pointing_game_fishnet.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    figura(ejemplos)


def figura(ejemplos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = os.path.join(ROOT, "docs", "figs_privadas")
    os.makedirs(d, exist_ok=True)
    fig, ax = plt.subplots(len(ejemplos), 3, figsize=(7.5, 2.5 * len(ejemplos)), dpi=110)
    for r, (j, img, m, cam, che, nombre) in enumerate(ejemplos):
        im = np.asarray(img.resize((data.IMG_SIZE, data.IMG_SIZE)))
        for c, (mapa, titulo) in enumerate(((None, nombre), (cam, "ResNet Grad-CAM"),
                                            (che, "ViT Chefer"))):
            ax[r, c].imshow(im)
            if mapa is not None:
                ax[r, c].imshow(mapa, cmap="jet", alpha=0.45)
            ys, xs = np.nonzero(m)
            ax[r, c].add_patch(plt.Rectangle((xs.min(), ys.min()), xs.max() - xs.min(),
                                             ys.max() - ys.min(), fill=False, ec="white", lw=1.5))
            ax[r, c].set_title("%s  [fila %d]" % (titulo, j) if c == 0 else titulo, fontsize=8)
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "mapas_fishnet.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
