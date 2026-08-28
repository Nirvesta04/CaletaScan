"""
Metricas, intervalos de confianza y comparacion entre arquitecturas.

    python -m src.evaluate --run e1_resnet50_kaggle
    python -m src.evaluate --run e2_vit_b_16_kaggle --compare e1_resnet50_kaggle

Trabaja SOLO sobre los logits guardados por train.py, asi que no toca la GPU y
corre en segundos. Todo lo que sigue -- ajuste de logits, bootstrap, estratos de
oclusion, credito jerarquico -- se recalcula sin reentrenar nada.

Las dos ideas que hacen que los numeros signifiquen algo:

1. La unidad independiente NO es la fila. En Kaggle son 62 fotos originales
   detras de 1.336 imagenes; en Fishnet, secuencias de video detras de los
   recortes. Los IC se calculan remuestreando CLUSTERES COMPLETOS.

2. Comparar dos modelos exige una prueba pareada sobre esos mismos clusteres.
   McNemar seria lo habitual, pero supone muestras independientes y aqui no lo
   son: con 21,5 rotaciones por foto declararia significativo casi cualquier
   cosa.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from . import data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
N_BOOT = 2000

# Unidad de agrupamiento para el bootstrap, por dataset.
CLUSTER_COL = {"kaggle": "group_id", "kaggle_random": "group_id",
               "fishnet": "seq_id"}


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def confusion(y: np.ndarray, pred: np.ndarray, K: int) -> np.ndarray:
    """Matriz de confusion KxK por bincount. Mucho mas rapida que sklearn
    dentro del bucle del bootstrap, que la llama 2.000 veces."""
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    sup = cm.sum(1).astype(np.float64)          # verdaderos de cada clase
    pred_n = cm.sum(0).astype(np.float64)       # predicciones de cada clase
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pred_n > 0, tp / np.maximum(pred_n, 1e-12), 0.0)
        rec = np.where(sup > 0, tp / np.maximum(sup, 1e-12), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"precision": prec, "recall": rec, "f1": f1, "support": sup}


def macro_f1(cm: np.ndarray, keep: np.ndarray) -> float:
    """
    Macro-F1 promediado sobre las clases de `keep` QUE TENGAN SOPORTE.

    Una clase ausente de la muestra no aporta un F1 de 0: aporta nada. Contarla
    como 0 hundiria las replicas del bootstrap donde una clase rara no salio
    sorteada, y el IC mediria eso en vez de medir el modelo.
    """
    m = per_class(cm)
    sel = keep & (m["support"] > 0)
    return float(m["f1"][sel].mean()) if sel.any() else float("nan")


def balanced_acc(cm: np.ndarray, keep: np.ndarray) -> float:
    m = per_class(cm)
    sel = keep & (m["support"] > 0)
    return float(m["recall"][sel].mean()) if sel.any() else float("nan")


# ---------------------------------------------------------------------------
# Ajuste de logits y credito jerarquico
# ---------------------------------------------------------------------------

def logit_adjust(z: np.ndarray, prior: np.ndarray, tau: float) -> np.ndarray:
    """
    Ajuste post-hoc de Menon et al. (2021): argmax_c (z_c - tau * log pi_c).

    Con tau=1 el clasificador resultante es el optimo de Bayes para el ERROR
    BALANCEADO. Como aqui se reporta macro-F1, el tau optimo no tiene por que
    ser 1: se barre en VALIDACION sobre la metrica que efectivamente se reporta.
    """
    return z - tau * np.log(np.maximum(prior, 1e-12))[None, :]


def sweep_tau(zv, yv, prior, keep, K, taus=None) -> tuple[float, float]:
    """Devuelve (mejor tau, macro-F1 de val con ese tau). Se ajusta SOLO en val."""
    taus = np.arange(0.0, 2.01, 0.05) if taus is None else taus
    best = (0.0, -1.0)
    for t in taus:
        f = macro_f1(confusion(yv, logit_adjust(zv, prior, t).argmax(1), K), keep)
        if f > best[1]:
            best = (float(t), f)
    return best


def hierarchical_credit(pred: np.ndarray, y: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    Credito jerarquico para la superclase `Tuna`.

    Un recorte etiquetado `Tuna` ES alguno de los atunes especificos; el
    anotador solo no resolvio cual. Si el modelo dice "Yellowfin tuna" sobre una
    verdad `Tuna`, hoy eso cuenta como falso positivo de Yellowfin y castiga su
    precision por un ruido de etiqueta que el modelo no causo.

    Se remapea esa prediccion a `Tuna`: deja de ser FP del atun especifico y
    pasa a ser acierto de la superclase. No toca ningun otro caso.
    """
    if "Tuna" not in classes:
        return pred
    idx = {c: i for i, c in enumerate(classes)}
    tuna = idx["Tuna"]
    especificos = np.array([idx[c] for c in data.TUNA_SPECIES if c in idx])
    out = pred.copy()
    m = (y == tuna) & np.isin(pred, especificos)
    out[m] = tuna
    return out


# ---------------------------------------------------------------------------
# Bootstrap de clusteres
# ---------------------------------------------------------------------------

def cluster_index(clusters: np.ndarray) -> list[np.ndarray]:
    """Filas agrupadas por cluster, listas para remuestrear."""
    order = np.argsort(clusters, kind="stable")
    c = clusters[order]
    corte = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    return np.split(order, corte[1:])


def bootstrap(fn, grupos: list[np.ndarray], n_boot: int = N_BOOT,
              seed: int = 0) -> tuple[float, float]:
    """
    IC 95% remuestreando CLUSTERES COMPLETOS con reemplazo.

    `fn` recibe un array de indices de fila y devuelve un escalar. Remuestrear
    filas sueltas en vez de clusteres da intervalos ~4,6x mas angostos de lo
    real en Kaggle (1.336 filas pero solo 62 unidades independientes).
    """
    rng = np.random.default_rng(seed)
    G = len(grupos)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        vals[b] = fn(np.concatenate([grupos[i] for i in pick]))
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(fn_a, fn_b, grupos, n_boot=N_BOOT, seed=0) -> dict:
    """
    Bootstrap pareado sobre la diferencia A - B.

    En cada replica se remuestrean LOS MISMOS clusteres para ambos modelos. El
    pareo importa: los dos se evaluan sobre las mismas muestras y sus errores
    estan correlacionados, asi que la varianza de la diferencia es mucho menor
    que la de cada metrica por separado.
    """
    rng = np.random.default_rng(seed)
    G = len(grupos)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        rows = np.concatenate([grupos[i] for i in pick])
        d[b] = fn_a(rows) - fn_b(rows)
    d = d[~np.isnan(d)]
    return {"delta_medio": float(d.mean()),
            "ic95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_delta_positivo": float((d > 0).mean())}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

class Run:
    """Una corrida cargada: config, datasets y logits de todas sus semillas."""

    def __init__(self, name: str):
        d = os.path.join(RUNS, name)
        if not os.path.isdir(d):
            raise FileNotFoundError("no existe runs/%s" % name)
        self.name = name
        self.seeds = sorted(int(s[4:]) for s in os.listdir(d)
                            if s.startswith("seed")
                            and os.path.exists(os.path.join(d, s, "metrics.json")))
        if not self.seeds:
            raise FileNotFoundError("runs/%s no tiene semillas completas" % name)

        with open(os.path.join(d, "seed%d" % self.seeds[0], "config.yaml"),
                  encoding="utf-8") as fh:
            self.cfg = yaml.safe_load(fh)

        kind = self.cfg["dataset"]
        if kind == "kaggle":
            dss = data.kaggle()
        elif kind == "kaggle_random":
            dss = data.kaggle_random_split(seed=0)
        else:
            dss = data.fishnet(label_col=self.cfg.get("label_col") or "label_model",
                               purge=self.cfg.get("purge", False))
        self.ds = dss
        self.classes = dss["train"].classes
        self.K = len(self.classes)
        self.keep = (dss["train"].species_mask() if kind == "fishnet"
                     else np.ones(self.K, dtype=bool))
        self.prior = dss["train"].class_counts() / len(dss["train"])
        self.cluster_col = CLUSTER_COL[kind]

        self.z = {s: {p: np.load(os.path.join(d, "seed%d" % s, "%s_logits.npy" % p))
                      for p in ("val", "test")} for s in self.seeds}

    def y(self, part):
        return self.ds[part].labels

    def clusters(self, part):
        return self.ds[part].df[self.cluster_col].to_numpy()


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def predictions(run: Run, seed: int, part: str, tau: float, jerarquico: bool):
    z = logit_adjust(run.z[seed][part], run.prior, tau) if tau else run.z[seed][part]
    pred = z.argmax(1)
    if jerarquico:
        pred = hierarchical_credit(pred, run.y(part), run.classes)
    return pred


def report(run: Run, part: str = "test", ajustar: bool = True,
           jerarquico: bool = True, n_boot: int = N_BOOT) -> dict:
    y = run.y(part)
    grupos = cluster_index(run.clusters(part))

    # tau se elige en VALIDACION, sobre la metrica que se reporta
    tau = 0.0
    if ajustar:
        taus = [sweep_tau(run.z[s]["val"], run.y("val"), run.prior, run.keep, run.K)
                for s in run.seeds]
        tau = float(np.mean([t for t, _ in taus]))

    preds = {s: predictions(run, s, part, tau, jerarquico) for s in run.seeds}

    f1s = np.array([macro_f1(confusion(y, preds[s], run.K), run.keep) for s in run.seeds])
    bas = np.array([balanced_acc(confusion(y, preds[s], run.K), run.keep) for s in run.seeds])
    accs = np.array([float((preds[s] == y).mean()) for s in run.seeds])

    # el IC de bootstrap promedia sobre semillas dentro de cada replica: mezcla
    # la variabilidad del test con la del entrenamiento en un solo intervalo
    def f_macro(rows):
        return float(np.mean([macro_f1(confusion(y[rows], preds[s][rows], run.K), run.keep)
                              for s in run.seeds]))

    ic = bootstrap(f_macro, grupos, n_boot)

    cm = sum(confusion(y, preds[s], run.K) for s in run.seeds)
    pc = per_class(cm)

    return {"run": run.name, "particion": part, "tau": tau,
            "jerarquico": jerarquico, "semillas": run.seeds,
            "n_filas": len(y), "n_clusteres": len(grupos),
            "macro_f1": {"media": float(f1s.mean()),
                         "std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
                         "ic95_bootstrap": list(ic)},
            "balanced_acc": {"media": float(bas.mean())},
            "accuracy": {"media": float(accs.mean()),
                         "nota": "no informativa con este desbalance; se reporta por completitud"},
            "por_clase": {c: {k: float(v[i]) for k, v in pc.items()}
                          for i, c in enumerate(run.classes)},
            "_preds": preds, "_y": y, "_grupos": grupos}


def por_estrato(run: Run, rep: dict, col: str = "occ_union",
                cortes=(-1e-9, 1e-9, 0.10, 0.30, 1.01)) -> pd.DataFrame:
    """Macro-F1 por estrato de oclusion. Es la prueba de la hipotesis del
    enunciado: si el ViT resiste mejor la oclusion, su ventaja sobre el ResNet
    tiene que CRECER de forma monotona con el estrato."""
    df = run.ds[rep["particion"]].df
    if col not in df.columns:
        return pd.DataFrame()
    nombres = ["0 aislado", "(0,0.10]", "(0.10,0.30]", ">0.30"]
    est = pd.cut(df[col], bins=list(cortes), labels=nombres)
    y, preds = rep["_y"], rep["_preds"]
    filas = []
    for n in nombres:
        rows = np.flatnonzero((est == n).to_numpy())
        if len(rows) == 0:
            continue
        f1 = np.mean([macro_f1(confusion(y[rows], preds[s][rows], run.K), run.keep)
                      for s in preds])
        filas.append({"estrato": n, "n": len(rows), "macro_f1": f1})
    return pd.DataFrame(filas)


def imprimir(rep: dict, run: Run):
    m = rep["macro_f1"]
    print("\n" + "=" * 74)
    print("%s  ·  %s  ·  semillas %s" % (rep["run"], rep["particion"], rep["semillas"]))
    print("=" * 74)
    print("  filas %d  |  clusteres independientes %d  (%s)"
          % (rep["n_filas"], rep["n_clusteres"], run.cluster_col))
    if rep["tau"]:
        print("  ajuste de logits tau=%.2f (elegido en validacion)" % rep["tau"])
    if rep["jerarquico"] and "Tuna" in run.classes:
        print("  credito jerarquico para la superclase Tuna: activo")
    print()
    print("  macro-F1 (especies) : %.4f  ± %.4f entre semillas" % (m["media"], m["std"]))
    print("                        IC95 bootstrap de clusteres [%.4f, %.4f]  (±%.1f pp)"
          % (m["ic95_bootstrap"][0], m["ic95_bootstrap"][1],
             100 * (m["ic95_bootstrap"][1] - m["ic95_bootstrap"][0]) / 2))
    print("  balanced accuracy   : %.4f" % rep["balanced_acc"]["media"])
    print("  accuracy global     : %.4f   <- %s"
          % (rep["accuracy"]["media"], rep["accuracy"]["nota"]))

    print("\n  %-26s %7s %7s %7s %8s" % ("clase", "prec", "recall", "F1", "soporte"))
    print("  " + "-" * 60)
    for c, v in sorted(rep["por_clase"].items(), key=lambda kv: -kv[1]["support"]):
        marca = " " if c in run.classes and run.keep[run.classes.index(c)] else "*"
        print("  %-26s %7.3f %7.3f %7.3f %8.0f %s"
              % (c[:26], v["precision"], v["recall"], v["f1"], v["support"], marca))
    print("  (* = fuera del macro-F1 de especies)")


def comparar(a: Run, b: Run, part="test", ajustar=True, jerarquico=True,
             n_boot=N_BOOT) -> dict:
    """Bootstrap de clusteres PAREADO sobre delta = macro-F1(a) - macro-F1(b)."""
    if a.classes != b.classes:
        raise ValueError("las dos corridas no comparten espacio de clases")
    ra = report(a, part, ajustar, jerarquico, n_boot=1)
    rb = report(b, part, ajustar, jerarquico, n_boot=1)
    y, grupos = ra["_y"], ra["_grupos"]

    def mk(rep, run):
        def f(rows):
            return float(np.mean([macro_f1(confusion(y[rows], p[rows], run.K), run.keep)
                                  for p in rep["_preds"].values()]))
        return f

    out = paired_bootstrap(mk(ra, a), mk(rb, b), grupos, n_boot)
    out.update({"a": a.name, "b": b.name,
                "macro_f1_a": ra["macro_f1"]["media"],
                "macro_f1_b": rb["macro_f1"]["media"]})

    print("\n" + "=" * 74)
    print("COMPARACION PAREADA  ·  %s  vs  %s  ·  %s" % (a.name, b.name, part))
    print("=" * 74)
    print("  macro-F1 %-28s %.4f" % (a.name, out["macro_f1_a"]))
    print("  macro-F1 %-28s %.4f" % (b.name, out["macro_f1_b"]))
    print("  delta (a - b)                        %+.4f" % out["delta_medio"])
    print("  IC95 de la diferencia                [%+.4f, %+.4f]"
          % tuple(out["ic95"]))
    print("  P(delta > 0)                         %.3f" % out["p_delta_positivo"])
    cruza = out["ic95"][0] <= 0 <= out["ic95"][1]
    print("\n  -> %s" % ("El IC de la diferencia CRUZA CERO: con este test no se "
                         "distinguen." if cruza else
                         "El IC no cruza cero: la diferencia es distinguible."))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--compare", default=None, help="segunda corrida a comparar")
    ap.add_argument("--part", default="test", choices=["val", "test"])
    ap.add_argument("--sin-ajuste", action="store_true",
                    help="no aplicar ajuste de logits")
    ap.add_argument("--sin-jerarquia", action="store_true",
                    help="no dar credito jerarquico a Tuna")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--json", default=None, help="ruta donde guardar el informe")
    args = ap.parse_args()

    run = Run(args.run)
    rep = report(run, args.part, not args.sin_ajuste, not args.sin_jerarquia, args.boot)
    imprimir(rep, run)

    est = por_estrato(run, rep)
    if not est.empty:
        print("\n  macro-F1 por estrato de oclusion (occ_union)")
        print("  " + est.to_string(index=False).replace("\n", "\n  "))

    salida = {k: v for k, v in rep.items() if not k.startswith("_")}
    if args.compare:
        salida["comparacion"] = comparar(run, Run(args.compare), args.part,
                                         not args.sin_ajuste,
                                         not args.sin_jerarquia, args.boot)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(salida, fh, indent=2, ensure_ascii=False)
        print("\n  informe guardado en %s" % args.json)


if __name__ == "__main__":
    main()
