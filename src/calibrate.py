"""
Calibracion y deteccion fuera de distribucion (§10 del protocolo).

    python -m src.calibrate --inferir     # logits OOD cruzados (GPU, reanudable)
    python -m src.calibrate               # tablas, sobre logits guardados (CPU)

Calibracion. Se ajusta una temperatura T por semilla sobre VALIDACION y se
reporta ECE, Brier y NLL en test antes y despues. Con ajuste de logits el orden
es el de §6.3/§10.1: primero tau (el mismo de evaluate.py), despues T. En
Fishnet ambos se eligen sobre la val LIMPIA y se evaluan sobre el test purgado.

La comparacion entre arquitecturas se hace sobre los logits CRUDOS: tau es una
regla de decision elegida por modelo (0,18 ResNet, 0,05 ViT) y descalibra por
construccion, asi que compararlas despues de tau mezclaria la red con la regla.

En Fishnet la T de validacion no transfiere al test, porque las dos
particiones limpias son poblaciones distintas (diagnostico()). El control
"crudo+T test2f" ajusta T dentro del test con validacion cruzada por secuencia
para separar eso de un fallo del metodo.

OOD sin OzFish. Los espacios de etiquetas de Kaggle y Fishnet son disjuntos, asi
que cada uno es un sondeo fuera de distribucion para el modelo del otro:

    modelo Kaggle  ·  ID = test Kaggle         ·  OOD = recortes de pez del test Fishnet
    modelo Fishnet ·  ID = test Fishnet limpio ·  OOD = imagenes del test Kaggle

Se trata como deteccion binaria con una puntuacion de confianza y se reporta
AUROC y FPR@95TPR, con IC de bootstrap de clusteres.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata

from . import data
from . import evaluate as E

ROOT = E.ROOT
RUNS = E.RUNS
N_BINS = 15            # ECE depende del numero de bins: se declara
T_MIN, T_MAX = 0.05, 20.0

KAGGLE_RUNS = ("e1_resnet50_kaggle", "e2_vit_b_16_kaggle")
FISHNET_RUNS = ("e4_resnet50_fishnet", "e5_vit_b_16_fishnet")


# ---------------------------------------------------------------------------
# Probabilidades y metricas de calibracion
# ---------------------------------------------------------------------------

def lse(z: np.ndarray) -> np.ndarray:
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def nll_rows(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    return lse(z) - z[np.arange(len(y)), y]


def brier_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brier multiclase: sum_c (p_c - 1[c=y])^2. Regla propia, sin bins."""
    q = p.copy()
    q[np.arange(len(y)), y] -= 1.0
    return (q * q).sum(1)


def ece_from(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    ECE con bins de ancho igual sobre la confianza top-1.

    Recibe confianza y acierto por fila, no probabilidades, para que el
    bootstrap pueda recalcularlo sobre cualquier subconjunto sin volver a
    pasar por la softmax.
    """
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    n = np.bincount(b, minlength=n_bins)
    s_conf = np.bincount(b, weights=conf, minlength=n_bins)
    s_acc = np.bincount(b, weights=correct, minlength=n_bins)
    return float(np.abs(s_acc - s_conf).sum() / len(conf))


def reliability(conf, correct, n_bins: int = N_BINS) -> dict:
    """Datos del diagrama de fiabilidad: por bin, confianza media, accuracy y n."""
    b = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    n = np.bincount(b, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        c = np.bincount(b, weights=conf, minlength=n_bins) / n
        a = np.bincount(b, weights=correct, minlength=n_bins) / n
    return {"n": n.tolist(), "confianza": np.nan_to_num(c).tolist(),
            "accuracy": np.nan_to_num(a).tolist()}


def fit_temperature(z: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
    """
    T que minimiza la NLL de softmax(z/T), por biseccion sobre beta = 1/T.

    La NLL es convexa en beta: su segunda derivada es la varianza de z bajo
    softmax(beta z), que no puede ser negativa. Entonces su derivada,
    E_p[z] - z_y, es monotona y el optimo es su unico cero: la biseccion lo
    encuentra sin supuestos de curvatura. Newton no sirve aqui: cuando los
    margenes son grandes la NLL decae como e^{-beta * margen} y Newton avanza
    a pasos de 1/margen, sin llegar nunca al borde.

    Devuelve (T, en_limite). Si la derivada no cambia de signo en
    [T_MIN, T_MAX] el optimo esta fuera: pasa con una validacion perfecta, en
    que la NLL baja sin limite al subir beta. Entonces T no esta identificada,
    se devuelve la cota y se avisa en vez de reportarla como una estimacion.
    """
    z = z.astype(np.float64)
    zy = z[np.arange(len(y)), y]

    def grad(b):
        p = softmax(b * z)
        return float(np.mean((p * z).sum(1) - zy))

    lo, hi = np.log(1.0 / T_MAX), np.log(1.0 / T_MIN)
    if grad(np.exp(hi)) <= 0:
        return T_MIN, True
    if grad(np.exp(lo)) >= 0:
        return T_MAX, True
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if grad(np.exp(mid)) > 0:
            hi = mid
        else:
            lo = mid
    return float(np.exp(-0.5 * (lo + hi))), False


# ---------------------------------------------------------------------------
# Puntuaciones OOD
# ---------------------------------------------------------------------------

def score_msp(z: np.ndarray) -> np.ndarray:
    """
    log de la max-softmax. Mismo orden que la MSP, sin saturar.

    En float32 la MSP vale exactamente 1,0 para margenes grandes, y miles de
    empates en 1,0 degradan el AUROC de forma artificial. En log y en float64,
    -log1p(sum_{j != max} e^{z_j - z_max}) distingue margenes de 50 y de 200.
    """
    z = z.astype(np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    e[np.arange(len(z)), z.argmax(1)] = 0.0   # sumar SIN el maximo: 1 + e^-50 == 1
    return -np.log1p(e.sum(1))


def score_energy(z: np.ndarray) -> np.ndarray:
    """Energia libre negada con T=1 (Liu et al. 2020): logsumexp(z). Alto = ID."""
    return lse(z.astype(np.float64))


def auroc(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    """AUROC con ID como clase positiva: U de Mann-Whitney / (n1 n2), empates a 1/2."""
    r = rankdata(np.concatenate([s_id, s_ood]))
    n1, n2 = len(s_id), len(s_ood)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def fpr_at_tpr(s_id: np.ndarray, s_ood: np.ndarray, tpr: float = 0.95) -> float:
    """
    Fraccion de OOD aceptada con el umbral que conserva el 95 % del ID.

    El umbral es el mayor t con mean(s_id >= t) >= tpr: el k-esimo mayor
    valor de ID, k = ceil(tpr * n). Los empates en el umbral cuentan como
    aceptados, que es el lado conservador: nunca hace parecer mejor al detector.
    """
    k = int(np.ceil(tpr * len(s_id)))
    thr = np.sort(s_id)[len(s_id) - k]
    return float((s_ood >= thr).mean())


def bootstrap2(fn, g_id, g_ood, n_boot=E.N_BOOT, seed=0) -> np.ndarray:
    """
    Bootstrap de clusteres en DOS muestras: ID y OOD se remuestrean por
    separado, cada una por sus propios clusteres. `fn(rows_id, rows_ood)`
    devuelve un escalar o un vector. Devuelve las replicas.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        ri = np.concatenate([g_id[i] for i in rng.integers(0, len(g_id), len(g_id))])
        ro = np.concatenate([g_ood[i] for i in rng.integers(0, len(g_ood), len(g_ood))])
        out.append(fn(ri, ro))
    return np.asarray(out)


def ic(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------------------------------------------------------------------------
# Calibracion
# ---------------------------------------------------------------------------

def _tau(run: E.Run, limpio: bool) -> float:
    """El mismo tau que usa evaluate.report(): media de los barridos por semilla."""
    iv = run.rows("val", limpio)
    return float(np.mean([E.sweep_tau(run.z[s]["val"][iv], run.y("val")[iv],
                                      run.prior, run.keep, run.K)[0]
                          for s in run.seeds]))


def _t_cruzada(z: np.ndarray, y: np.ndarray, grupos: list[np.ndarray],
               seed: int = 0) -> tuple[np.ndarray, list[float]]:
    """
    Control: temperatura ajustada DENTRO del test, en 2 pliegues por cluster.

    La mitad de las secuencias ajusta T y la otra mitad se evalua con ella, y
    al reves. Ninguna fila se usa a la vez para ajustar y para evaluar. No es
    un numero reportable como modelo (usa etiquetas de test); es el
    diagnostico que separa "la temperatura no sirve en estos modelos" de "la
    validacion no se parece al test".
    """
    orden = np.random.default_rng(seed).permutation(len(grupos))
    pliegues = [np.concatenate([grupos[i] for i in orden[k::2]]) for k in (0, 1)]
    out, Ts = z.astype(np.float64).copy(), []
    for a, b in ((0, 1), (1, 0)):
        T, _ = fit_temperature(z[pliegues[a]], y[pliegues[a]])
        out[pliegues[b]] = z[pliegues[b]] / T
        Ts.append(T)
    return out, Ts


def diagnostico(run: E.Run) -> dict:
    """
    Por que la temperatura de val no transfiere al test en Fishnet: las dos
    particiones limpias son poblaciones distintas (mezcla de clases y de
    camaras), y la T optima de cada una apunta en direccion opuesta.
    """
    hum = run.classes.index("Human")
    tr_cam = set(run.ds["train"].df["cam_id"])
    out = {}
    for part in ("val", "test"):
        it = run.rows(part, True)
        y = run.y(part)[it]
        cam = run.ds[part].df["cam_id"].to_numpy()[it]
        per = []
        for s in run.seeds:
            z = run.z[s][part][it]
            p = softmax(z.astype(np.float64))
            conf, ok = p.max(1), (p.argmax(1) == y).astype(float)
            per.append((ok.mean(), conf.mean(), ece_from(conf, ok), fit_temperature(z, y)[0]))
        m = np.mean(per, axis=0)
        out[part] = {"n": int(len(y)), "frac_human": float((y == hum).mean()),
                     "frac_camaras_no_vistas": float(np.mean([c not in tr_cam for c in cam])),
                     "accuracy": float(m[0]), "confianza_media": float(m[1]),
                     "ece_crudo": float(m[2]), "T_optima_en_esta_particion": float(m[3])}
    return out


def calibracion(run: E.Run, n_boot: int) -> dict:
    limpio = run.cfg["dataset"] == "fishnet"
    iv, it = run.rows("val", limpio), run.rows("test", limpio)
    yv, yt = run.y("val")[iv], run.y("test")[it]
    tau = _tau(run, limpio)
    grupos = E.cluster_index(run.clusters("test")[it])

    variantes = {}          # nombre -> {seed: (conf, correct, nll, brier)}
    temps = {}
    for s in run.seeds:
        zv, zt = run.z[s]["val"][iv], run.z[s]["test"][it]
        za_v, za_t = (E.logit_adjust(zv, run.prior, tau),
                      E.logit_adjust(zt, run.prior, tau))
        T_raw, lim_raw = fit_temperature(zv, yv)
        T_adj, lim_adj = fit_temperature(za_v, yv)
        z_cruz, T_cruz = _t_cruzada(zt, yt, grupos)
        temps[s] = {"T_crudo": T_raw, "T_ajustado": T_adj,
                    "T_test_cruzada": T_cruz, "en_limite": lim_raw or lim_adj}
        for nombre, z in (("crudo", zt), ("crudo+T", zt / T_raw),
                          ("crudo+T test2f", z_cruz),
                          ("ajustado", za_t), ("ajustado+T", za_t / T_adj)):
            p = softmax(z.astype(np.float64))
            conf, pred = p.max(1), p.argmax(1)
            variantes.setdefault(nombre, {})[s] = (
                conf, (pred == yt).astype(float), nll_rows(z, yt), brier_rows(p, yt))

    def metricas(v, rows):
        per = [(ece_from(c[rows], a[rows]), float(nl[rows].mean()), float(br[rows].mean()),
                float(c[rows].mean()), float(a[rows].mean()))
               for c, a, nl, br in v.values()]
        return np.mean(per, axis=0)            # promedio sobre semillas

    todas = np.arange(len(yt))
    out = {"run": run.name, "tau": tau, "solo_limpio": limpio,
           "n_filas": len(yt), "n_clusteres": len(grupos), "n_bins": N_BINS,
           "temperaturas": temps, "variantes": {}}
    for nombre, v in variantes.items():
        m = metricas(v, todas)
        reps = np.array([metricas(v, np.concatenate([grupos[i] for i in
                         np.random.default_rng(b).integers(0, len(grupos), len(grupos))]))
                         for b in range(n_boot)])
        c = np.concatenate([x[0] for x in v.values()])
        a = np.concatenate([x[1] for x in v.values()])
        out["variantes"][nombre] = {
            "ece": float(m[0]), "nll": float(m[1]), "brier": float(m[2]),
            "confianza_media": float(m[3]), "accuracy": float(m[4]),
            "ic95": {"ece": ic(reps[:, 0]), "nll": ic(reps[:, 1]), "brier": ic(reps[:, 2])},
            "fiabilidad": reliability(c, a)}

    if limpio:
        # Human es el 68 % del test limpio y se descalibra en sentido CONTRARIO
        # al resto: el label smoothing topa su confianza en 1 - 0,1 + 0,1/K
        # mientras acierta casi siempre. En el agregado los dos errores se
        # compensan y lo esconden; por eso se separa.
        hum = yt == run.classes.index("Human")
        v = variantes["crudo"]
        out["crudo_por_grupo"] = {
            g: {k: float(np.mean([f(c[mk], a[mk]) for c, a, _, _ in v.values()]))
                for k, f in (("ece", ece_from),
                             ("confianza_media", lambda c, a: c.mean()),
                             ("accuracy", lambda c, a: a.mean()))}
            for g, mk in (("Human", hum), ("sin Human", ~hum))}
        out["diagnostico_val_test"] = diagnostico(run)
        # clusteres restringidos a las filas sin Human, para la comparacion
        # pareada donde de verdad se juega la pregunta del proyecto
        idx = np.flatnonzero(~hum)
        out["_grupos_sin_human"] = [idx[g] for g in
                                    E.cluster_index(run.clusters("test")[it][idx])]
    out["_v"], out["_grupos"] = variantes, grupos
    return out


def comparar_calibracion(a: dict, b: dict, variante: str, n_boot: int,
                         grupos: str = "_grupos") -> dict:
    """Bootstrap pareado de B - A (ViT - ResNet) sobre los mismos clusteres."""
    va, vb = a["_v"][variante], b["_v"][variante]
    g = a[grupos]

    def m(v, rows):
        return np.mean([(ece_from(c[rows], ac[rows]), nl[rows].mean(), br[rows].mean())
                        for c, ac, nl, br in v.values()], axis=0)

    d = []
    for bi in range(n_boot):
        rows = np.concatenate([g[i] for i in
                               np.random.default_rng(bi).integers(0, len(g), len(g))])
        d.append(m(vb, rows) - m(va, rows))
    d = np.array(d)
    todas = np.concatenate(g)
    d0 = m(vb, todas) - m(va, todas)
    return {k: {"delta": float(d0[j]), "ic95": ic(d[:, j]),
                "p_delta_negativo": float((d[:, j] < 0).mean())}
            for j, k in enumerate(("ece", "nll", "brier"))}


# ---------------------------------------------------------------------------
# OOD: inferencia cruzada
# ---------------------------------------------------------------------------

def ood_dataset(run: E.Run):
    """El test del OTRO dataset, con su propio pipeline de eval."""
    if run.cfg["dataset"] == "kaggle":
        return data.fishnet()["test"]
    return data.kaggle()["test"]


def ood_path(run_name: str, seed: int) -> str:
    return os.path.join(RUNS, run_name, "seed%d" % seed, "ood_logits.npy")


def inferir(nombres, batch_size=64, num_workers=4):
    import torch

    from . import models
    from .train import infer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for nombre in nombres:
        run = E.Run(nombre)
        pend = [s for s in run.seeds if not os.path.exists(ood_path(nombre, s))]
        if not pend:
            print("  %s: ya inferido, se omite" % nombre)
            continue
        ds = ood_dataset(run)
        ld = data.make_loader(ds, batch_size, train=False, num_workers=num_workers)
        try:
            for s in pend:
                ck = os.path.join(RUNS, nombre, "seed%d" % s, "checkpoint.pt")
                m = models.build_model(run.cfg["model"], run.K)
                m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
                m.to(dev)
                z = infer(m, ld, len(ds), run.K, dev)
                np.save(ood_path(nombre, s), z)
                print("  %s seed%d: %d filas OOD -> ood_logits.npy" % (nombre, s, len(ds)),
                      flush=True)
                del m
                torch.cuda.empty_cache()
        finally:
            data.shutdown_loaders(ld)


# ---------------------------------------------------------------------------
# OOD: metricas
# ---------------------------------------------------------------------------

SCORES = {"msp": score_msp, "energia": score_energy}


def ood(run: E.Run, subconjunto: str = "peces") -> dict:
    """
    Puntuaciones ID/OOD por semilla y grupos para el bootstrap.

    Modelo Kaggle: el OOD principal son los recortes de PEZ de Fishnet
    (is_fish=1): la pregunta es si dice "trucha" con 0,98 frente a un atun.
    `subconjunto='todos'` agrega tambien personas y cubierta vacia.

    Modelo Fishnet: `subconjunto='id_sin_human'` saca a Human del lado ID.
    Human es el 68 % del test limpio y el modelo lo reconoce con confianza
    alta, asi que con el ID completo el AUROC mide en buena parte "persona vs
    pez de Kaggle". Sin Human la pregunta es la del muelle: distinguir un pez
    de su dominio de uno que no lo es.
    """
    faltan = [s for s in run.seeds if not os.path.exists(ood_path(run.name, s))]
    if faltan:
        raise SystemExit("faltan logits OOD de %s (semillas %s): correr con --inferir"
                         % (run.name, faltan))
    limpio = run.cfg["dataset"] == "fishnet"
    it = run.rows("test", limpio)
    if subconjunto == "id_sin_human":
        it = it[run.y("test")[it] != run.classes.index("Human")]
    ds_ood = ood_dataset(run)
    if run.cfg["dataset"] == "kaggle":
        io = (np.flatnonzero(ds_ood.df["is_fish"].to_numpy() == 1)
              if subconjunto == "peces" else np.arange(len(ds_ood)))
        col_ood = "seq_id"
    else:
        io, col_ood = np.arange(len(ds_ood)), "group_id"

    s = {k: {sd: (f(run.z[sd]["test"][it]), f(np.load(ood_path(run.name, sd))[io]))
             for sd in run.seeds} for k, f in SCORES.items()}
    return {"run": run.name, "s": s,
            "g_id": E.cluster_index(run.clusters("test")[it]),
            "g_ood": E.cluster_index(ds_ood.df[col_ood].to_numpy()[io]),
            "n_id": len(it), "n_ood": len(io)}


def _ood_metricas(s_por_semilla, ri=None, ro=None):
    """[AUROC, FPR@95] promediados sobre semillas, en las filas dadas."""
    v = []
    for a, b in s_por_semilla.values():
        a_, b_ = (a, b) if ri is None else (a[ri], b[ro])
        v.append((auroc(a_, b_), fpr_at_tpr(a_, b_)))
    return np.mean(v, axis=0)


def ood_informe(o: dict, n_boot: int) -> dict:
    out = {"run": o["run"], "n_id": o["n_id"], "n_ood": o["n_ood"],
           "n_clusteres_id": len(o["g_id"]), "n_clusteres_ood": len(o["g_ood"])}
    for k, sp in o["s"].items():
        m = _ood_metricas(sp)
        per = [(auroc(a, b), fpr_at_tpr(a, b)) for a, b in sp.values()]
        reps = bootstrap2(lambda ri, ro: _ood_metricas(sp, ri, ro),
                          o["g_id"], o["g_ood"], n_boot)
        out[k] = {"auroc": float(m[0]), "auroc_std_semillas": float(np.std([p[0] for p in per], ddof=1)),
                  "auroc_ic95": ic(reps[:, 0]),
                  "fpr95": float(m[1]), "fpr95_ic95": ic(reps[:, 1])}
    # confianza media ID vs OOD, en probabilidad: la cifra que se entiende sola
    msp = o["s"]["msp"]
    out["msp_media_id"] = float(np.mean([np.exp(a).mean() for a, _ in msp.values()]))
    out["msp_media_ood"] = float(np.mean([np.exp(b).mean() for _, b in msp.values()]))
    out["ood_con_msp_mayor_0.9"] = float(np.mean([(np.exp(b) > 0.9).mean()
                                                  for _, b in msp.values()]))
    return out


def ood_comparar(a: dict, b: dict, n_boot: int) -> dict:
    """Pareado B - A: mismos clusteres ID y OOD para las dos arquitecturas."""
    out = {}
    for k in SCORES:
        d = bootstrap2(lambda ri, ro: _ood_metricas(b["s"][k], ri, ro)
                       - _ood_metricas(a["s"][k], ri, ro),
                       a["g_id"], a["g_ood"], n_boot)
        d0 = _ood_metricas(b["s"][k]) - _ood_metricas(a["s"][k])
        out[k] = {"delta_auroc": float(d0[0]), "ic95_auroc": ic(d[:, 0]),
                  "p_delta_auroc_positivo": float((d[:, 0] > 0).mean()),
                  "delta_fpr95": float(d0[1]), "ic95_fpr95": ic(d[:, 1])}
    return out


# ---------------------------------------------------------------------------

def _limpiar(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _fmt_ic(v):
    return "[%.4f, %.4f]" % tuple(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inferir", action="store_true",
                    help="genera los logits OOD cruzados (GPU); reanudable por semilla")
    ap.add_argument("--boot", type=int, default=E.N_BOOT)
    ap.add_argument("--json", default="runs/calibracion_ood.json")
    args = ap.parse_args()

    if args.inferir:
        inferir(KAGGLE_RUNS + FISHNET_RUNS)
        return

    res = {"calibracion": {}, "ood": {}, "comparaciones": {}}

    print("=" * 78)
    print("CALIBRACION  ·  ECE (%d bins), NLL y Brier en test  ·  T ajustada en val" % N_BINS)
    print("=" * 78)
    for par in (KAGGLE_RUNS, FISHNET_RUNS):
        cal = {}
        for nombre in par:
            run = E.Run(nombre)
            c = calibracion(run, args.boot)
            cal[nombre] = c
            res["calibracion"][nombre] = _limpiar(c)
            Ts = [t["T_crudo"] for t in c["temperaturas"].values()]
            lim = any(t["en_limite"] for t in c["temperaturas"].values())
            print("\n%s  ·  tau %.2f  ·  %d filas, %d clusteres  ·  T = %s%s"
                  % (nombre, c["tau"], c["n_filas"], c["n_clusteres"],
                     " / ".join("%.3f" % t for t in Ts),
                     "  (EN EL LIMITE: no identificada)" if lim else ""))
            print("  %-15s %8s %8s %8s %10s %9s" % ("variante", "ECE", "NLL", "Brier",
                                                   "conf.media", "accuracy"))
            for k, v in c["variantes"].items():
                print("  %-15s %8.4f %8.4f %8.4f %10.4f %9.4f   ECE %s"
                      % (k, v["ece"], v["nll"], v["brier"], v["confianza_media"],
                         v["accuracy"], _fmt_ic(v["ic95"]["ece"])))
            if "diagnostico_val_test" in c:
                for part, d in c["diagnostico_val_test"].items():
                    print("  %-5s limpia: Human %.2f, camaras no vistas %.2f, acc %.3f, "
                          "conf %.3f, T optima %.3f"
                          % (part, d["frac_human"], d["frac_camaras_no_vistas"],
                             d["accuracy"], d["confianza_media"],
                             d["T_optima_en_esta_particion"]))
                for g, d in c["crudo_por_grupo"].items():
                    print("  crudo · %-9s ECE %.4f  conf %.3f  acc %.3f  (conf - acc %+.3f)"
                          % (g, d["ece"], d["confianza_media"], d["accuracy"],
                             d["confianza_media"] - d["accuracy"]))
        a, b = (cal[n] for n in par)
        casos = [(v, "_grupos") for v in ("crudo", "crudo+T test2f", "ajustado")]
        if "_grupos_sin_human" in a:
            casos.append(("crudo", "_grupos_sin_human"))
        for var, gr in casos:
            cmp_ = comparar_calibracion(a, b, var, args.boot, gr)
            etiq = var + (" · sin Human" if gr != "_grupos" else "")
            res["comparaciones"]["cal · %s vs %s · %s" % (par[1], par[0], etiq)] = cmp_
            print("\n  ViT - ResNet (%s):  " % etiq + "   ".join(
                "%s %+.4f %s" % (k, v["delta"], _fmt_ic(v["ic95"])) for k, v in cmp_.items()))

    print("\n" + "=" * 78)
    print("OOD CRUZADO  ·  AUROC (ID positivo) y FPR@95TPR  ·  puntuaciones con T=1")
    print("=" * 78)
    for par, subs in ((KAGGLE_RUNS, ("peces", "todos")),
                      (FISHNET_RUNS, ("todos", "id_sin_human"))):
        for sub in subs:
            oo = {}
            for nombre in par:
                o = ood(E.Run(nombre), sub)
                oo[nombre] = o
                r = ood_informe(o, args.boot)
                res["ood"]["%s · %s" % (nombre, sub)] = r
                print("\n%s  ·  OOD=%s  ·  ID %d filas (%d cl.)  OOD %d filas (%d cl.)"
                      % (nombre, sub, r["n_id"], r["n_clusteres_id"], r["n_ood"],
                         r["n_clusteres_ood"]))
                for k in SCORES:
                    print("  %-8s AUROC %.4f %s   FPR@95 %.4f %s"
                          % (k, r[k]["auroc"], _fmt_ic(r[k]["auroc_ic95"]),
                             r[k]["fpr95"], _fmt_ic(r[k]["fpr95_ic95"])))
                print("  MSP media: ID %.3f  OOD %.3f  ·  OOD con MSP > 0,9: %.1f %%"
                      % (r["msp_media_id"], r["msp_media_ood"], 100 * r["ood_con_msp_mayor_0.9"]))
            cmp_ = ood_comparar(oo[par[0]], oo[par[1]], args.boot)
            res["comparaciones"]["ood · %s vs %s · %s" % (par[1], par[0], sub)] = cmp_
            for k, v in cmp_.items():
                print("  ViT - ResNet  %-8s dAUROC %+.4f %s  P>0 %.3f   dFPR95 %+.4f %s"
                      % (k, v["delta_auroc"], _fmt_ic(v["ic95_auroc"]),
                         v["p_delta_auroc_positivo"], v["delta_fpr95"], _fmt_ic(v["ic95_fpr95"])))

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\nguardado en %s" % args.json)


if __name__ == "__main__":
    main()
