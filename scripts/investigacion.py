"""
Que significan los resultados: analisis sobre las predicciones guardadas.

    python -m scripts.investigacion      # -> runs/investigacion.json

Preguntas, todas sobre la prueba limpia de FishNet y sin reentrenar:
1. Tamano del pez en la foto: ¿donde gana cada diseno?
2. Luz (brillo del recorte): ¿de dia y de noche igual?
3. Datos por especie: ¿el F1 lo explica el modelo o la cantidad de ejemplos?
4. Errores compartidos: ¿fallan los cinco en los mismos peces?
5. Combinar modelos: ¿promediar sus probabilidades mejora?
6. Derivar al inspector: si el sistema solo responde cuando esta seguro,
   ¿cuanto acierta en lo que responde?
7. Camaras: ¿que tan distinto rinde de una camara a otra?
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

from src import data
from src.evaluate import Run, confusion, logit_adjust, macro_f1, report, hierarchical_credit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
MOD = {"ResNet-50": "e4_resnet50_fishnet", "DeiT-S": "e10_deit_s_fishnet",
       "ViT-B/16": "e5_vit_b_16_fishnet", "ConvNeXt-T": "e11_convnext_t_fishnet",
       "Swin-T": "e12_swin_t_fishnet"}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    runs = {m: Run(r) for m, r in MOD.items()}
    reps = {m: report(runs[m], "test", True, True, n_boot=1, solo_limpio=True) for m in MOD}
    base = runs["ResNet-50"]
    K, keep, cls = base.K, base.keep, base.classes
    idx = reps["ResNet-50"]["_idx"]
    y = reps["ResNet-50"]["_y"]
    df = base.ds["test"].df.iloc[idx].reset_index(drop=True)
    es_especie = keep[y]
    out = {}

    def f1(rows, preds):
        return float(np.mean([macro_f1(confusion(y[rows], p[rows], K), keep) for p in preds]))

    def acc(rows, preds):
        return float(np.mean([(p[rows] == y[rows]).mean() for p in preds]))

    P = {m: list(reps[m]["_preds"].values()) for m in MOD}

    # ---------------------------------------------------------- 1. tamano
    lado = np.minimum(df["x1"] - df["x0"], df["y1"] - df["y0"]).to_numpy()
    cortes = [0, 80, 130, 220, 10_000]
    nombres = ["< 80 px", "80-130 px", "130-220 px", "> 220 px"]
    out["tamano"] = {}
    for a, b, n in zip(cortes[:-1], cortes[1:], nombres):
        rows = np.flatnonzero((lado >= a) & (lado < b) & es_especie)
        out["tamano"][n] = {"cajas_especie": int(len(rows)),
                            **{m: {"acierto": acc(rows, P[m])} for m in MOD}}
    # macro-F1 por tamano con todas las filas del estrato (incluye no-especies como contexto)
    for a, b, n in zip(cortes[:-1], cortes[1:], nombres):
        rows = np.flatnonzero((lado >= a) & (lado < b))
        for m in MOD:
            out["tamano"][n][m]["macro_f1"] = f1(rows, P[m])

    # ---------------------------------------------------------- 2. luz
    brillo = np.zeros(len(df))
    ds = base.ds["test"]
    for i, j in enumerate(idx):
        if not es_especie[i]:
            continue
        with Image.open(os.path.join(ROOT, ds.paths[j])) as im:
            brillo[i] = np.asarray(im.convert("L").crop(ds._margin_box(j, im.size)).resize((32, 32))).mean()
    q = np.quantile(brillo[es_especie], [1 / 3, 2 / 3])
    out["luz"] = {"cortes_brillo": [float(x) for x in q]}
    for n, cond in (("oscuro", brillo < q[0]), ("medio", (brillo >= q[0]) & (brillo < q[1])),
                    ("claro", brillo >= q[1])):
        rows = np.flatnonzero(cond & es_especie)
        out["luz"][n] = {"cajas": int(len(rows)), **{m: acc(rows, P[m]) for m in MOD}}

    # ---------------------------------------------------------- 3. datos por especie
    ntrain = base.ds["train"].class_counts()
    out["datos_por_especie"] = {}
    sp = [k for k in range(K) if keep[k] and (y == k).any()]
    for m in MOD:
        cm = sum(confusion(y, p, K) for p in P[m]) / len(P[m])
        tp = np.diag(cm); f1k = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1e-9)
        rho, pv = spearmanr([ntrain[k] for k in sp], [f1k[k] for k in sp])
        out["datos_por_especie"][m] = {"spearman_f1_vs_ejemplos_train": float(rho), "p": float(pv),
                                       "f1": {cls[k]: float(f1k[k]) for k in sp}}
    out["datos_por_especie"]["ejemplos_train"] = {cls[k]: int(ntrain[k]) for k in sp}
    out["datos_por_especie"]["cajas_test"] = {cls[k]: int((y == k).sum()) for k in sp}

    # ---------------------------------------------------------- 4. errores compartidos
    rows = np.flatnonzero(es_especie)
    # semilla 0 de cada modelo, para no mezclar semillas
    p0 = {m: P[m][0] for m in MOD}
    ok = np.array([p0[m][rows] == y[rows] for m in MOD])        # modelos x cajas
    out["errores"] = {"cajas_especie": int(len(rows)),
                      "todos_aciertan": float(ok.all(0).mean()),
                      "todos_fallan": float((~ok).all(0).mean()),
                      "al_menos_uno_acierta": float(ok.any(0).mean())}
    pares = {}
    nm = list(MOD)
    for i in range(len(nm)):
        for j in range(i + 1, len(nm)):
            fi, fj = ~ok[i], ~ok[j]
            pares["%s | %s" % (nm[i], nm[j])] = float((fi & fj).sum() / max(1, (fi | fj).sum()))
    out["errores"]["solapamiento_errores_jaccard"] = pares
    # en que fallan todos
    todos = rows[(~ok).all(0)]
    out["errores"]["todos_fallan_por_especie"] = {cls[k]: int((y[todos] == k).sum()) for k in sp
                                                  if (y[todos] == k).sum() > 0}

    # ---------------------------------------------------------- 5. combinar modelos
    def probs(m, s):
        r = runs[m]
        z = r.z[s]["test"][idx]
        return softmax(logit_adjust(z, r.prior, reps[m]["tau"]))

    seeds = sorted(set.intersection(*[set(runs[m].seeds) for m in MOD]))
    out["combinar"] = {}
    for nombre, grupo in (("ConvNeXt-T + Swin-T", ["ConvNeXt-T", "Swin-T"]),
                          ("los cinco", list(MOD)),
                          ("ResNet-50 + ViT-B/16", ["ResNet-50", "ViT-B/16"])):
        pr = [hierarchical_credit(np.mean([probs(m, s) for m in grupo], 0).argmax(1), y, cls) for s in seeds]
        out["combinar"][nombre] = {"macro_f1": f1(np.arange(len(y)), pr)}
    for m in MOD:
        out["combinar"][m] = {"macro_f1": f1(np.arange(len(y)), P[m])}

    # ---------------------------------------------------------- 6. derivar al inspector
    out["derivar"] = {}
    peces = np.flatnonzero(df["is_fish"].to_numpy() == 1)
    for m in ("ResNet-50", "ViT-B/16", "ConvNeXt-T", "Swin-T"):
        curva = []
        for s in seeds:
            p = probs(m, s)[peces]
            pred = hierarchical_credit(p.argmax(1), y[peces], cls)
            conf = p.max(1)
            orden = np.argsort(-conf)
            fila = []
            for cob in (1.0, 0.8, 0.6, 0.4, 0.2):
                sel = orden[:int(round(cob * len(orden)))]
                fila.append(float((pred[sel] == y[peces][sel]).mean()))
            curva.append(fila)
        out["derivar"][m] = dict(zip(["100 %", "80 %", "60 %", "40 %", "20 %"],
                                     [float(x) for x in np.mean(curva, 0)]))
    out["derivar"]["nota"] = ("acierto sobre los peces (is_fish=1) que el modelo responde, ordenando por "
                              "confianza; la cobertura es la fraccion de peces que responde solo")

    # ---------------------------------------------------------- 7. camaras
    out["camaras"] = {}
    cam = df["cam_id"].to_numpy()
    for m in ("ResNet-50", "ConvNeXt-T"):
        accs = {}
        for c in np.unique(cam):
            rows = np.flatnonzero((cam == c) & es_especie)
            if len(rows) >= 40:
                accs[int(c)] = acc(rows, P[m])
        v = np.array(list(accs.values()))
        out["camaras"][m] = {"n_camaras": len(v), "min": float(v.min()), "mediana": float(np.median(v)),
                             "max": float(v.max()), "por_camara": accs}

    with open(os.path.join(RUNS, "investigacion.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    imprimir(out)


def imprimir(o):
    print("\n1. TAMANO (acierto en cajas de especie)")
    for n, d in o["tamano"].items():
        print("  %-11s n=%4d  " % (n, d["cajas_especie"]) + "  ".join("%s %.2f" % (m[:6], d[m]["acierto"]) for m in MOD))
    print("   macro-F1:")
    for n, d in o["tamano"].items():
        print("  %-11s       " % n + "  ".join("%s %.2f" % (m[:6], d[m]["macro_f1"]) for m in MOD))
    print("\n2. LUZ (acierto en cajas de especie)")
    for n in ("oscuro", "medio", "claro"):
        d = o["luz"][n]
        print("  %-7s n=%4d  " % (n, d["cajas"]) + "  ".join("%s %.2f" % (m[:6], d[m]) for m in MOD))
    print("\n3. F1 POR ESPECIE vs EJEMPLOS DE ENTRENAMIENTO (Spearman)")
    for m in MOD:
        d = o["datos_por_especie"][m]
        print("  %-11s rho %.2f (p %.3f)" % (m, d["spearman_f1_vs_ejemplos_train"], d["p"]))
    print("\n4. ERRORES", {k: v for k, v in o["errores"].items() if k not in ("solapamiento_errores_jaccard", "todos_fallan_por_especie")})
    print("   todos fallan, por especie:", o["errores"]["todos_fallan_por_especie"])
    print("   solapamiento:", {k: round(v, 2) for k, v in o["errores"]["solapamiento_errores_jaccard"].items()})
    print("\n5. COMBINAR (macro-F1)", {k: round(v["macro_f1"], 3) for k, v in o["combinar"].items()})
    print("\n6. DERIVAR (acierto segun cobertura, peces)")
    for m, d in o["derivar"].items():
        if m != "nota":
            print("  %-11s " % m + "  ".join("%s: %.2f" % kv for kv in d.items()))
    print("\n7. CAMARAS (acierto en especies, camaras con >= 40 cajas)")
    for m, d in o["camaras"].items():
        print("  %-11s %d camaras: min %.2f mediana %.2f max %.2f" % (m, d["n_camaras"], d["min"], d["mediana"], d["max"]))


if __name__ == "__main__":
    main()
