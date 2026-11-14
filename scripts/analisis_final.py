"""
Analisis para la entrega final. Responde las observaciones del Avance sin
reentrenar: todo sale de los logits guardados en runs/.

    python -m scripts.analisis_final

Escribe runs/analisis_final.json y las figuras en docs/figs/. Calcula:

1. Tabla de modelos: macro-F1 de especies en el test limpio de FishNet, con IC,
   y las comparaciones pareadas de cada par (ResNet-50 vs ViT-B/16, ResNet-50
   vs DeiT-S, ConvNeXt-T vs Swin-T). Las corridas que no existan se saltan.
2. F1 por especie y matrices de confusion entre especies.
3. Atunes agrupados: los cuatro atunes y `Tuna` cuentan como una sola clase.
4. Camaras nuevas: el test limpio separado en camaras que estan en train y
   camaras que no. Lo segundo mide si el modelo sirve en un barco nuevo.
5. Evolucion por epoca (val) de cada corrida.
6. La cifra de fuga del split oficial, en cajas y en imagenes.

Todas las cifras de FishNet usan lo mismo que el informe del Avance: test
limpio, ajuste de logits con tau elegido en val y credito jerarquico de Tuna.
"""

from __future__ import annotations

import json
import os

import numpy as np

from src import data
from src.evaluate import (Run, bootstrap, cluster_index, confusion, macro_f1,
                          paired_bootstrap, per_class, report)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "docs", "figs")

MODELOS = {
    "e4_resnet50_fishnet": "ResNet-50",
    "e5_vit_b_16_fishnet": "ViT-B/16",
    "e10_deit_s_fishnet": "DeiT-S",
    "e11_convnext_t_fishnet": "ConvNeXt-T",
    "e12_swin_t_fishnet": "Swin-T",
}
# (a, b): la diferencia se reporta como a - b. Los tres primeros son transformer - CNN.
PARES = [
    ("e5_vit_b_16_fishnet", "e4_resnet50_fishnet", "Par original"),
    ("e10_deit_s_fishnet", "e4_resnet50_fishnet", "Mismo tamano"),
    ("e12_swin_t_fishnet", "e11_convnext_t_fishnet", "Par moderno"),
    # moderno - antiguo, dentro de cada familia. OJO: el ResNet-50 usa otra receta
    # (dos niveles de lr) que ConvNeXt-T (decaimiento por nivel): la diferencia
    # mide modelo y receta juntos.
    ("e11_convnext_t_fishnet", "e4_resnet50_fishnet", "CNN moderna contra antigua"),
    ("e12_swin_t_fishnet", "e5_vit_b_16_fishnet", "Transformer moderno contra antiguo"),
]
N_BOOT = 2000


def cargar():
    runs, reps = {}, {}
    for nombre in MODELOS:
        try:
            r = Run(nombre)
        except FileNotFoundError:
            print("  (sin corrida completa: %s)" % nombre)
            continue
        if len(r.seeds) < 3:
            print("  (%s tiene %d semillas; se usa igual)" % (nombre, len(r.seeds)))
        runs[nombre] = r
        reps[nombre] = report(r, "test", True, True, n_boot=1, solo_limpio=True)
    return runs, reps


def f_macro(run, rep, filas_sel=None, mapa=None, keep=None):
    """Funcion de filas -> macro-F1 promediado sobre semillas, con remapeo opcional."""
    y = rep["_y"] if mapa is None else mapa[rep["_y"]]
    preds = [p if mapa is None else mapa[p] for p in rep["_preds"].values()]
    keep = run.keep if keep is None else keep

    def f(rows):
        return float(np.mean([macro_f1(confusion(y[rows], p[rows], run.K), keep)
                              for p in preds]))
    return f


def resumen(run, rep, rows, mapa=None, keep=None):
    f = f_macro(run, rep, mapa=mapa, keep=keep)
    grupos = cluster_index(run.clusters("test")[rep["_idx"]][rows])
    # los grupos indexan dentro de `rows`; se traducen a filas de rep
    grupos = [rows[g] for g in grupos]
    return {"macro_f1": f(rows), "ic95": list(bootstrap(f, grupos, N_BOOT)),
            "n_cajas": int(len(rows)), "n_secuencias": len(grupos)}, grupos


def delta(ra, pa, rb, pb, rows, grupos, mapa=None, keep=None):
    fa = f_macro(ra, pa, mapa=mapa, keep=keep)
    fb = f_macro(rb, pb, mapa=mapa, keep=keep)
    out = paired_bootstrap(fa, fb, grupos, N_BOOT)
    out["distinguible"] = not (out["ic95"][0] <= 0 <= out["ic95"][1])
    return out


def atunes(run):
    """Mapa de clases que junta los 4 atunes y `Tuna` en el indice de `Tuna`."""
    idx = {c: i for i, c in enumerate(run.classes)}
    mapa = np.arange(run.K)
    for c in data.TUNA_SPECIES:
        mapa[idx[c]] = idx["Tuna"]
    keep = run.keep.copy()
    for c in data.TUNA_SPECIES:
        keep[idx[c]] = False
    keep[idx["Tuna"]] = True
    return mapa, keep


def camaras(run, rep):
    """Filas (dentro del test limpio) de camaras vistas y no vistas en train."""
    cams_train = set(run.ds["train"].df["cam_id"])
    cam = run.ds["test"].df["cam_id"].to_numpy()[rep["_idx"]]
    nueva = ~np.isin(cam, list(cams_train))
    return np.flatnonzero(~nueva), np.flatnonzero(nueva), len(set(cam)), len(set(cam[nueva]))


def fuga_oficial(run):
    tr, te = run.ds["train"].df, run.ds["test"].df
    cruza = te["seq_id"].isin(set(tr["seq_id"]))
    img = te.groupby("img_id")["seq_id"].first().isin(set(tr["seq_id"]))
    return {"cajas_test": int(len(te)), "cajas_que_comparten_secuencia": int(cruza.sum()),
            "frac_cajas": float(cruza.mean()),
            "imagenes_test": int(len(img)), "imagenes_que_comparten_secuencia": int(img.sum()),
            "frac_imagenes": float(img.mean())}


def epocas(nombre):
    out = {}
    d = os.path.join(RUNS, nombre)
    for s in sorted(os.listdir(d)):
        f = os.path.join(d, s, "metrics.json")
        if s.startswith("seed") and os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                h = json.load(fh)["hist"]
            out[s] = [{"epoca": i + 1, "loss": e["loss"], "val_macro_f1": e["val_macro_f1"],
                       "min": e["sec"] / 60} for i, e in enumerate(h)]
    return out


def figura_confusion(run, rep, nombre, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = rep["_y"]
    cm = sum(confusion(y, p, run.K) for p in rep["_preds"].values()) / len(rep["_preds"])
    sp = [i for i in range(run.K) if run.keep[i] and cm[i].sum() > 0]
    sub = cm[np.ix_(sp, sp)]
    fila = sub / np.maximum(sub.sum(1, keepdims=True), 1e-12)
    etiquetas = ["%s (%d)" % (run.classes[i], round(cm[i].sum())) for i in sp]

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=160)
    ax.imshow(fila, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(sp)), [run.classes[i] for i in sp], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(sp)), etiquetas, fontsize=8)
    for i in range(len(sp)):
        for j in range(len(sp)):
            if fila[i, j] >= 0.05:
                ax.text(j, i, "%.0f" % (100 * fila[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if fila[i, j] > 0.5 else "black")
    ax.set_xlabel("especie predicha")
    ax.set_ylabel("especie real (cajas en el test limpio)")
    ax.set_title("%s: %% de cada especie real, test limpio de FishNet" % nombre, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    runs, reps = cargar()
    base = runs["e4_resnet50_fishnet"]
    out = {"nota": "test limpio de FishNet; macro-F1 sobre especies; IC95 por "
                   "bootstrap de secuencias (%d replicas)" % N_BOOT}

    out["fuga_split_oficial"] = fuga_oficial(base)

    # --- 1. tabla de modelos y pares -----------------------------------------
    todas = np.arange(len(reps["e4_resnet50_fishnet"]["_y"]))
    tabla, grupos_todas = {}, None
    for n, r in runs.items():
        res, grupos_todas = resumen(r, reps[n], todas)
        res.update({"modelo": MODELOS[n], "tau": reps[n]["tau"], "semillas": r.seeds,
                    "por_semilla": [macro_f1(confusion(reps[n]["_y"], p, r.K), r.keep)
                                    for p in reps[n]["_preds"].values()]})
        tabla[n] = res
    out["modelos"] = tabla

    # --- 3. atunes agrupados y 4. camaras -------------------------------------
    mapa, keep_at = atunes(base)
    vistas, nuevas, n_cam, n_cam_nuevas = camaras(base, reps["e4_resnet50_fishnet"])
    out["camaras"] = {"camaras_en_test_limpio": n_cam, "camaras_nuevas": n_cam_nuevas,
                      "cajas_camara_vista": int(len(vistas)),
                      "cajas_camara_nueva": int(len(nuevas)),
                      "frac_cajas_camara_vista": float(len(vistas) / len(todas))}

    variantes = {"todas": (todas, None, None), "atunes_agrupados": (todas, mapa, keep_at),
                 "camara_vista": (vistas, None, None), "camara_nueva": (nuevas, None, None)}
    out["por_variante"] = {}
    for v, (rows, m, k) in variantes.items():
        fila = {}
        gr = None
        for n, r in runs.items():
            res, gr = resumen(r, reps[n], rows, mapa=m, keep=k)
            fila[MODELOS[n]] = res
        pares = {}
        for a, b, etiqueta in PARES:
            if a in runs and b in runs:
                pares["%s - %s" % (MODELOS[a], MODELOS[b])] = dict(
                    etiqueta=etiqueta,
                    **delta(runs[a], reps[a], runs[b], reps[b], rows, gr, mapa=m, keep=k))
        out["por_variante"][v] = {"modelos": fila, "pares": pares}

    # --- 2. por especie y matrices --------------------------------------------
    out["por_especie"] = {}
    for n, r in runs.items():
        rep = reps[n]
        cm = sum(confusion(rep["_y"], p, r.K) for p in rep["_preds"].values()) / len(rep["_preds"])
        pc = per_class(cm)
        out["por_especie"][MODELOS[n]] = {
            r.classes[i]: {"f1": float(pc["f1"][i]), "precision": float(pc["precision"][i]),
                           "recall": float(pc["recall"][i]), "cajas": float(pc["support"][i])}
            for i in range(r.K) if r.keep[i] and pc["support"][i] > 0}
        figura_confusion(r, rep, MODELOS[n],
                         os.path.join(FIGS, "confusion_%s.png" % n.split("_")[0]))

    # --- 5. epocas -------------------------------------------------------------
    out["epocas"] = {MODELOS[n]: epocas(n) for n in runs}

    ruta = os.path.join(RUNS, "analisis_final.json")
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    imprimir(out)
    print("\nguardado: %s" % ruta)


def imprimir(out):
    f = out["fuga_split_oficial"]
    print("\nFuga del split oficial (nuestro subconjunto): %.1f %% de las cajas de test, "
          "%.1f %% de las imagenes" % (100 * f["frac_cajas"], 100 * f["frac_imagenes"]))
    c = out["camaras"]
    print("Camaras: %d en el test limpio, %d nuevas; %.1f %% de las cajas vienen de camaras vistas"
          % (c["camaras_en_test_limpio"], c["camaras_nuevas"], 100 * c["frac_cajas_camara_vista"]))
    for v, d in out["por_variante"].items():
        print("\n== %s" % v)
        for m, r in d["modelos"].items():
            print("  %-11s %.3f  [%.3f, %.3f]  cajas %d, secuencias %d"
                  % (m, r["macro_f1"], r["ic95"][0], r["ic95"][1], r["n_cajas"], r["n_secuencias"]))
        for p, r in d["pares"].items():
            print("  %-24s %+.3f  [%+.3f, %+.3f]  %s"
                  % (p, r["delta_medio"], r["ic95"][0], r["ic95"][1],
                     "distinguible" if r["distinguible"] else "no se distingue"))


if __name__ == "__main__":
    main()
