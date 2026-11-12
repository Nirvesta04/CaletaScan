"""
Brecha chilena: ¿que hacen los modelos de FishNet frente a reineta, merluza y
congrio, especies que nunca vieron?

    python -m scripts.chile_ood

Fotos: 45 observaciones de iNaturalist con licencia CC0, CC-BY o CC-BY-NC
(15 por especie), listadas con su autor en data/chile_creditos.csv. Las
imagenes viven en data/raw/chile/ y no se versionan.

Es una prueba CUALITATIVA. Lo correcto frente a estas fotos es decir "no se":
ninguna de las tres especies esta en FishNet. Se mide:

- cuantas rechaza cada modelo con el criterio de energia de la demo,
  con el umbral elegido en VALIDACION (no en test): la energia que deja pasar
  el 95 % de los recortes de val limpios que no son personas;
- el AUROC entre esas fotos y el test limpio sin personas, que no depende de
  ningun umbral;
- que especie dice cada modelo cuando NO rechaza.

Limitacion que hay que decir siempre: son fotos completas de un pez, muchas
sobre una mesa o en la mano, no recortes de una camara a bordo. Parte de lo que
los modelos detectan puede ser el cambio de tipo de foto y no solo la especie.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image

from src import calibrate as C
from src import data, models
from src import evaluate as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPR = 0.95
CORRIDAS = [("ResNet-50", "e4_resnet50_fishnet"), ("ViT-B/16", "e5_vit_b_16_fishnet")]


def umbral_val(run: E.Run, seed: int) -> float:
    iv = run.rows("val", True)
    iv = iv[run.y("val")[iv] != run.classes.index("Human")]
    s = C.score_energy(run.z[seed]["val"][iv])
    k = int(np.ceil(TPR * len(s)))
    return float(np.sort(s)[len(s) - k])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(ROOT, "data", "chile_creditos.csv"), encoding="utf-8") as fh:
        fotos = list(csv.DictReader(fh))
    tf = data.build_transform(train=False)
    x = torch.stack([tf(Image.open(os.path.join(ROOT, f["archivo"])).convert("RGB"))
                     for f in fotos])
    especie = np.array([f["especie"] for f in fotos])
    print("%d fotos: %s" % (len(fotos), dict(Counter(especie))))

    out = {"n": len(fotos), "por_especie": dict(Counter(especie)), "tpr_val": TPR,
           "modelos": {}}
    for etiqueta, nombre in CORRIDAS:
        run = E.Run(nombre)
        tau = C._tau(run, limpio=True)
        it = run.rows("test", True)
        it = it[run.y("test")[it] != run.classes.index("Human")]
        rech, aucs, dichas = [], [], Counter()
        for s in run.seeds:
            m = models.build_model(run.cfg["model"], run.K)
            ck = os.path.join(E.RUNS, nombre, "seed%d" % s, "checkpoint.pt")
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
            m.to(dev).eval()
            with torch.inference_mode():
                z = torch.cat([m(x[i:i + 16].to(dev)).float().cpu()
                               for i in range(0, len(x), 16)]).numpy()
            del m
            torch.cuda.empty_cache()

            e = C.score_energy(z)
            u = umbral_val(run, s)
            rech.append(e < u)
            aucs.append(C.auroc(C.score_energy(run.z[s]["test"][it]), e))
            pred = E.logit_adjust(z, run.prior, tau).argmax(1)
            dichas.update(run.classes[p] for p in pred[e >= u])

        rech = np.array(rech)                     # semillas x fotos
        out["modelos"][etiqueta] = {
            "rechaza_total": float(rech.mean()),
            "rechaza_por_especie": {sp: float(rech[:, especie == sp].mean())
                                    for sp in sorted(set(especie))},
            "auroc_vs_test_limpio_sin_personas": float(np.mean(aucs)),
            "auroc_por_semilla": [float(a) for a in aucs],
            "cuando_acepta_dice": dict(dichas.most_common(6)),
        }
        r = out["modelos"][etiqueta]
        print("\n%s  rechaza %.0f %%  (%s)  AUROC %.3f"
              % (etiqueta, 100 * r["rechaza_total"],
                 ", ".join("%s %.0f %%" % (k, 100 * v) for k, v in r["rechaza_por_especie"].items()),
                 r["auroc_vs_test_limpio_sin_personas"]))
        print("  cuando acepta dice: %s" % r["cuando_acepta_dice"])

    with open(os.path.join(E.RUNS, "chile_ood.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
