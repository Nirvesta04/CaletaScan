"""
Auditoria de los resultados: fuga, sobreajuste y veracidad de la comparacion.

    python scripts/auditoria.py            # ~10 min; usa GPU en la parte 3
    python scripts/auditoria.py --rapido   # muestrea; para iterar

La pregunta que responde: los numeros de Kaggle rozan el 100 %, asi que hay que
descartar que vengan de una fuga y no del problema. Y hay que comprobar que la
comparacion entre arquitecturas no este inclinada por un detalle del montaje.

Cuatro pruebas independientes:

1. INTEGRIDAD DE LOS SPLITS. Ningun grupo, imagen ni secuencia puede aparecer
   en dos particiones. Se verifica contra los CSV, no contra lo que dice el
   codigo que hace.

2. CASI DUPLICADOS. La prueba anterior confia en que la clave de agrupamiento
   (la foto original, la secuencia) capture todas las repeticiones. Esta no
   confia en nada: compara los PIXELES de cada imagen de test contra todas las
   de train. En Kaggle se comparan ademas las 8 variantes de rotacion y espejo,
   porque una rotacion de 90 grados destruye la correlacion y un duplicado
   rotado pasaria desapercibido.

3. SOBREAJUSTE. Se mide el rendimiento sobre el propio train, con transformada
   de evaluacion, y se compara con el de test. Una brecha enorme significa
   memorizacion; una brecha pequena con test alto significa que la tarea es
   facil o que hay un atajo.

4. VERACIDAD DE LA COMPARACION. Las metricas se recalculan desde los logits
   guardados y se contrastan contra metrics.json, y se comprueba que las dos
   arquitecturas recibieron el mismo presupuesto.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data, models                      # noqa: E402
from src import evaluate as E                     # noqa: E402

MINI = 32                    # lado de la miniatura para comparar pixeles
CASI_DUP = (0.90, 0.95)      # umbrales de correlacion, los mismos del analisis de secuencias


def titulo(t):
    print("\n" + "=" * 78 + "\n%s\n" % t + "=" * 78)


# ---------------------------------------------------------------------------
# 1. Integridad de los splits
# ---------------------------------------------------------------------------

def integridad():
    titulo("1 · INTEGRIDAD DE LOS SPLITS")
    ok = []
    k = pd.read_csv(os.path.join(data.SPLITS, "kaggle_split.csv"))
    por = {s: set(k.loc[k.split == s, "group_id"]) for s in ("train", "val", "test")}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        n = len(por[a] & por[b])
        ok.append(("kaggle · fotos originales compartidas %s/%s" % (a, b), n == 0, n))
    n = len(set(k.loc[k.split == "train", "path"]) & set(k.loc[k.split == "test", "path"]))
    ok.append(("kaggle · rutas de imagen compartidas train/test", n == 0, n))

    f = pd.read_csv(os.path.join(data.SPLITS, "fishnet_crops.csv"))
    for col in ("img_id", "path"):
        n = len(set(f.loc[f.split == "train", col]) & set(f.loc[f.split == "test", col]))
        ok.append(("fishnet · %s compartidos train/test" % col, n == 0, n))
    tr = set(f.loc[f.split == "train", "seq_id"])
    n = int(f.loc[f.split == "test", "seq_id"].isin(tr).sum())
    ok.append(("fishnet · cajas de test con secuencia de train (SPLIT OFICIAL)", False, n))
    limpio = data.purge_sequence_leak(f)
    n = int(limpio.loc[limpio.split == "test", "seq_id"].isin(tr).sum())
    ok.append(("fishnet · lo mismo tras purgar (el test que reportamos)", n == 0, n))
    tr_img = set(f.loc[f.split == "train", "img_id"])
    n = int(limpio.loc[limpio.split == "test", "img_id"].isin(tr_img).sum())
    ok.append(("fishnet · cajas del test purgado cuya IMAGEN esta en train", n == 0, n))
    n = len(set(f.loc[f.split == "train", "cam_id"]) & set(f.loc[f.split == "test", "cam_id"]))
    ok.append(("fishnet · camaras compartidas train/test (NO es fuga, es dominio)", None, n))

    for nombre, bien, n in ok:
        marca = {True: "OK  ", False: "OJO ", None: "nota"}[bien]
        print("  [%s] %-62s %6d" % (marca, nombre[:62], n))
    return ok


# ---------------------------------------------------------------------------
# 2. Casi duplicados entre particiones, por pixeles
# ---------------------------------------------------------------------------

class Miniaturas(Dataset):
    """Una imagen por item; devuelve una miniatura por caja (o una si no hay cajas)."""

    def __init__(self, df: pd.DataFrame, crop: bool):
        self.crop = crop
        self.grupos = list(df.groupby("path").indices.items())
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.grupos)

    def __getitem__(self, i):
        path, filas = self.grupos[i]
        img = Image.open(os.path.join(data.ROOT, path)).convert("L")
        out = []
        for j in filas:
            im = img
            if self.crop:
                r = self.df.iloc[j]
                im = img.crop((int(r.x0), int(r.y0), int(r.x1), int(r.y1)))
            v = np.asarray(im.resize((MINI, MINI), Image.BILINEAR), dtype=np.float32).ravel()
            v -= v.mean()
            n = np.linalg.norm(v)
            out.append(v / n if n > 1e-6 else v)
        return np.array(filas), np.stack(out)


def _tal_cual(x):
    return x          # en Windows el collate no puede ser una lambda: spawn no la serializa


CACHE = os.path.join(E.RUNS, "_cache_miniaturas")


def miniaturas(df, crop, workers=4, cache=None):
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        f = os.path.join(CACHE, "%s_%d.npy" % (cache, len(df)))
        if os.path.exists(f):
            return np.load(f)
    ds = Miniaturas(df, crop)
    ld = DataLoader(ds, batch_size=None, num_workers=workers, collate_fn=_tal_cual)
    idx, vecs = [], []
    for k, (ii, vv) in enumerate(ld):
        idx.append(ii)
        vecs.append(vv)
        if k % 2000 == 0:
            print("    %d/%d imagenes" % (k, len(ds)), flush=True)
    orden = np.concatenate(idx)
    M = np.concatenate(vecs)[np.argsort(orden)]
    if cache:
        np.save(f, M)
    return M


def variantes(M):
    """Las 8 simetrias del cuadrado: rotar destruye la correlacion pixel a pixel."""
    A = M.reshape(-1, MINI, MINI)
    out = []
    for esp in (A, A[:, :, ::-1]):
        for r in range(4):
            out.append(np.rot90(esp, r, axes=(1, 2)).reshape(len(A), -1))
    return out


def max_corr(test_vars, train_M, bloque=4096):
    """Maxima correlacion de cada fila de test contra TODO train, sobre las variantes."""
    best = np.full(len(test_vars[0]), -1.0, dtype=np.float32)
    quien = np.zeros(len(test_vars[0]), dtype=np.int64)
    for i in range(0, len(train_M), bloque):
        B = np.ascontiguousarray(train_M[i:i + bloque].T)
        for V in test_vars:
            S = np.ascontiguousarray(V) @ B
            m = S.max(1)
            arg = S.argmax(1) + i
            mejor = m > best
            quien[mejor] = arg[mejor]
            best = np.maximum(best, m)
    return best, quien


def duplicados(rapido: bool):
    titulo("2 · CASI DUPLICADOS ENTRE PARTICIONES (correlacion de pixeles)")
    res = {}

    # --- Kaggle: test contra train, con las 8 simetrias ---
    k = pd.read_csv(os.path.join(data.SPLITS, "kaggle_split.csv"))
    tr, te = k[k.split == "train"], k[k.split == "test"]
    if rapido:
        tr, te = tr.sample(1500, random_state=0), te.sample(300, random_state=0)
    print("  kaggle: %d test contra %d train (x8 variantes)" % (len(te), len(tr)))
    Mtr = miniaturas(tr, crop=False, cache="k_train")
    Mte = miniaturas(te, crop=False, cache="k_test")
    best, quien = max_corr(variantes(Mte), Mtr)
    # clasificador trivial: la clase del vecino mas parecido en train
    vecino_clase = tr.reset_index(drop=True).clase.to_numpy()[quien]
    nn_acc = float((vecino_clase == te.clase.to_numpy()).mean())
    print("    1-NN sobre miniaturas de %dx%d px: %.4f de accuracy en el test de Kaggle" % (MINI, MINI, nn_acc))
    res["kaggle_1nn_accuracy"] = nn_acc

    # referencia: cuanto se parecen entre si las rotaciones de UNA MISMA foto
    g = te.reset_index(drop=True).groupby("group_id").indices
    ref = []
    for filas in list(g.values())[:40]:
        if len(filas) > 1:
            S = Mte[filas] @ Mte[filas].T
            np.fill_diagonal(S, -1)
            ref.append(S.max(1).mean())
    res["kaggle"] = {"max_medio": float(best.mean()), "max_maximo": float(best.max()),
                     "sobre_0.90": int((best > 0.90).sum()), "sobre_0.95": int((best > 0.95).sum()),
                     "n_test": int(len(te)), "referencia_misma_foto": float(np.mean(ref))}
    print("    correlacion maxima contra train: media %.3f · maxima %.3f" % (best.mean(), best.max()))
    print("    referencia (rotaciones de la MISMA foto, dentro de test): %.3f" % np.mean(ref))
    print("    pares > 0,90: %d · > 0,95: %d  de %d" % ((best > 0.90).sum(), (best > 0.95).sum(), len(te)))

    # --- FishNet: el test LIMPIO y el test PURGADO, los dos contra train ---
    # Las cajas purgadas comparten secuencia con train: son la referencia de
    # "esto SI es un casi duplicado". Si las limpias se parecen a train tanto
    # como ellas, la purga no sirvio de nada.
    f = pd.read_csv(os.path.join(data.SPLITS, "fishnet_crops.csv"))
    tr_seq = set(f.loc[f.split == "train", "seq_id"])
    test = f[f.split == "test"]
    te_limpio = test[~test.seq_id.isin(tr_seq)]
    te_fuga = test[test.seq_id.isin(tr_seq)]
    tr = f[f.split == "train"]
    if rapido:
        tr = tr.sample(6000, random_state=0)
        te_limpio = te_limpio.sample(800, random_state=0)
        te_fuga = te_fuga.sample(800, random_state=0)
    print("  fishnet: %d limpias y %d purgadas contra %d de train"
          % (len(te_limpio), len(te_fuga), len(tr)))
    Mtr = miniaturas(tr, crop=True, cache="f_train")
    res["fishnet"] = {}
    guardar = {}
    for nombre, te in (("limpio", te_limpio), ("purgado_por_fuga", te_fuga)):
        M = miniaturas(te, crop=True, cache="f_" + nombre)
        best_f, quien_f = max_corr([M], Mtr)
        guardar[nombre] = (te.reset_index(drop=True), best_f, quien_f)
        res["fishnet"][nombre] = {
            "max_medio": float(best_f.mean()), "max_maximo": float(best_f.max()),
            "frac_sobre_0.90": float((best_f > 0.90).mean()),
            "frac_sobre_0.95": float((best_f > 0.95).mean()), "n": int(len(te))}
        print("    %-18s correlacion maxima contra train: media %.3f · >0,90 %.1f %% · >0,95 %.1f %%"
              % (nombre, best_f.mean(), 100 * (best_f > 0.90).mean(), 100 * (best_f > 0.95).mean()))
    # que son las pocas cajas limpias que se parecen mucho: pez o fondo?
    te_r, best_f, quien_f = guardar["limpio"]
    altos = np.flatnonzero(best_f > 0.90)
    if len(altos):
        sub = te_r.iloc[altos]
        tr_r = tr.reset_index(drop=True)
        vecino = tr_r.iloc[quien_f[altos]]
        misma = int((sub.label_model.to_numpy() == vecino.label_model.to_numpy()).sum())
        print("    de las %d cajas limpias con correlacion > 0,90: %d son pez, %d comparten clase con su vecino"
              % (len(altos), int(sub.is_fish.sum()), misma))
        res["fishnet"]["limpio_altos"] = {"n": int(len(altos)), "son_pez": int(sub.is_fish.sum()),
                                          "misma_clase_que_el_vecino": misma,
                                          "clases": sub.label_model.value_counts().to_dict(),
                                          "filas_del_test_limpio": altos.tolist()}
    vecino_clase = tr.reset_index(drop=True).label_model.to_numpy()[quien_f]
    nn_f = float((vecino_clase == te_r.label_model.to_numpy()).mean())
    print("    1-NN sobre miniaturas: %.4f de accuracy en el test limpio de FishNet" % nn_f)
    res["fishnet"]["1nn_accuracy_limpio"] = nn_f

    # sensibilidad: el resultado sin esas cajas parecidas
    res["sensibilidad_sin_casi_duplicados"] = sensibilidad(altos)
    return res


def sensibilidad(filas_fuera: np.ndarray) -> dict:
    """
    Recalcula la comparacion QUITANDO las cajas del test limpio que mas se
    parecen a alguna de train. Si la conclusion aguanta sin ellas, el residuo
    de casi duplicados no la sostiene.
    """
    a, b = E.Run("e4_resnet50_fishnet"), E.Run("e5_vit_b_16_fishnet")
    it = a.rows("test", True)
    keep = np.setdiff1d(np.arange(len(it)), filas_fuera)
    y = a.y("test")[it]
    taus = {}
    for r, nombre in ((a, "ResNet"), (b, "ViT")):
        iv = r.rows("val", True)
        taus[nombre] = float(np.mean([E.sweep_tau(r.z[s]["val"][iv], r.y("val")[iv],
                                                  r.prior, r.keep, r.K)[0] for s in r.seeds]))
    out = {"n_fuera": int(len(filas_fuera)), "n_queda": int(len(keep))}
    preds = {}
    for r, nombre in ((a, "ResNet"), (b, "ViT")):
        preds[nombre] = {s: E.predictions(r, s, "test", taus[nombre], True)[it] for s in r.seeds}
        con = float(np.mean([E.macro_f1(E.confusion(y, preds[nombre][s], r.K), r.keep) for s in r.seeds]))
        sin = float(np.mean([E.macro_f1(E.confusion(y[keep], preds[nombre][s][keep], r.K), r.keep)
                             for s in r.seeds]))
        out[nombre] = {"con_todo": con, "sin_casi_duplicados": sin, "cambio": sin - con}
        print("    %-7s macro-F1 con todo %.4f · sin los casi duplicados %.4f (%+.4f)"
              % (nombre, con, sin, sin - con))
    def f(rows, nombre, r):
        return float(np.mean([E.macro_f1(E.confusion(y[rows], preds[nombre][s][rows], r.K), r.keep)
                              for s in r.seeds]))

    # los clusteres se arman sobre las filas que quedan y se devuelven a indices
    # del test limpio, que es como estan indexados y y preds
    gru = [keep[g] for g in E.cluster_index(a.clusters("test")[it][keep])]
    d = E.paired_bootstrap(lambda rows: f(rows, "ViT", b), lambda rows: f(rows, "ResNet", a), gru, n_boot=1000)
    print("    delta sin los casi duplicados: %+.4f  IC95 [%+.4f, %+.4f]  P(delta>0) %.3f"
          % (d["delta_medio"], d["ic95"][0], d["ic95"][1], d["p_delta_positivo"]))
    out["delta_pareado"] = d
    return out


# ---------------------------------------------------------------------------
# 3. Sobreajuste: train contra test, mismos modelos
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _logits(modelo, ds, filas, dev, bs=64):
    out = np.zeros((len(filas), len(ds.classes)), dtype=np.float32)
    for i in range(0, len(filas), bs):
        lote = filas[i:i + bs]
        x = torch.stack([ds[j][0] for j in lote]).to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out[i:i + bs] = modelo(x).float().cpu().numpy()
    return out


def sobreajuste(rapido: bool, n=3000):
    titulo("3 · SOBREAJUSTE: RENDIMIENTO EN TRAIN CONTRA TEST")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = 800 if rapido else n
    res = {}
    casos = [("e1_resnet50_kaggle", "resnet50"), ("e2_vit_b_16_kaggle", "vit_b_16"),
             ("e4_resnet50_fishnet", "resnet50"), ("e5_vit_b_16_fishnet", "vit_b_16")]
    for run, arch in casos:
        r = E.Run(run)
        ds_tr = r.ds["train"]
        ds_tr.transform = data.build_transform(train=False)     # sin aumentacion: medicion, no entrenamiento
        filas = np.random.default_rng(0).choice(len(ds_tr), min(n, len(ds_tr)), replace=False)
        m = models.build_model(arch, r.K)
        ck = os.path.join(E.RUNS, run, "seed0", "checkpoint.pt")
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
        m = m.to(dev).eval()
        z = _logits(m, ds_tr, filas, dev)
        y = ds_tr.labels[filas]
        f1_tr = E.macro_f1(E.confusion(y, z.argmax(1), r.K), r.keep)
        acc_tr = float((z.argmax(1) == y).mean())

        limpio = r.cfg["dataset"] == "fishnet"
        it = r.rows("test", limpio)
        yt = r.y("test")[it]
        zt = r.z[0]["test"][it]
        f1_te = E.macro_f1(E.confusion(yt, zt.argmax(1), r.K), r.keep)
        acc_te = float((zt.argmax(1) == yt).mean())
        res[run] = {"n_train": int(len(filas)), "macro_f1_train": f1_tr, "macro_f1_test": f1_te,
                    "acc_train": acc_tr, "acc_test": acc_te,
                    "brecha_f1": f1_tr - f1_te, "brecha_acc": acc_tr - acc_te}
        print("  %-24s train F1 %.4f acc %.4f | test F1 %.4f acc %.4f | brecha F1 %+.4f acc %+.4f"
              % (run, f1_tr, acc_tr, f1_te, acc_te, f1_tr - f1_te, acc_tr - acc_te))
        del m
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------
# 4. Veracidad de la comparacion
# ---------------------------------------------------------------------------

def veracidad():
    titulo("4 · VERACIDAD DE LA COMPARACION")
    res = {}

    print("  4.1 metricas recalculadas desde los logits contra metrics.json")
    for run in ("e1_resnet50_kaggle", "e2_vit_b_16_kaggle",
                "e4_resnet50_fishnet", "e5_vit_b_16_fishnet"):
        r = E.Run(run)
        limpio = r.cfg["dataset"] == "fishnet"
        it = r.rows("test", limpio)
        difs = []
        for s in r.seeds:
            with open(os.path.join(E.RUNS, run, "seed%d" % s, "metrics.json"), encoding="utf-8") as fh:
                mj = json.load(fh)
            guardado = mj.get("test", {}).get("macro_f1_limpio", mj.get("test", {}).get("macro_f1"))
            recalc = E.macro_f1(E.confusion(r.y("test")[it], r.z[s]["test"][it].argmax(1), r.K), r.keep)
            difs.append(abs(recalc - guardado))
        res.setdefault("recalculo", {})[run] = float(max(difs))
        print("    %-24s maxima diferencia entre lo guardado y lo recalculado: %.2e"
              % (run, max(difs)))

    print("  4.2 presupuesto emparejado")
    for par in (("e1_resnet50_kaggle", "e2_vit_b_16_kaggle"),
                ("e4_resnet50_fishnet", "e5_vit_b_16_fishnet")):
        fichas = []
        for run in par:
            with open(os.path.join(E.RUNS, run, "seed0", "config.yaml"), encoding="utf-8") as fh:
                c = yaml.safe_load(fh)
            with open(os.path.join(E.RUNS, run, "seed0", "metrics.json"), encoding="utf-8") as fh:
                mj = json.load(fh)
            fichas.append({"run": run, "pasos": mj["steps"]["total"],
                           "epocas": c["epochs"], "batch": c["batch_size"] * c["accum_steps"],
                           "semillas": c["seeds"], "label_smoothing": c["label_smoothing"],
                           "amp": c["amp"], "dataset": c["dataset"]})
            print("    %-24s pasos %s · epocas %s · batch efectivo %s · semillas %s · ls %s"
                  % (fichas[-1]["run"], fichas[-1]["pasos"], fichas[-1]["epocas"],
                     fichas[-1]["batch"], fichas[-1]["semillas"], fichas[-1]["label_smoothing"]))
        igual = all(fichas[0][k] == fichas[1][k] for k in ("pasos", "batch", "semillas",
                                                          "label_smoothing", "amp", "dataset"))
        print("      -> presupuesto identico: %s" % ("SI" if igual else "NO"))
        res.setdefault("presupuesto", {})["%s vs %s" % par] = igual

    print("  4.3 estabilidad de la diferencia (por semilla y con otras semillas de bootstrap)")
    a, b = E.Run("e4_resnet50_fishnet"), E.Run("e5_vit_b_16_fishnet")
    it_a, it_b = a.rows("test", True), b.rows("test", True)
    tau_a = float(np.mean([E.sweep_tau(a.z[s]["val"][a.rows("val", True)],
                                       a.y("val")[a.rows("val", True)], a.prior, a.keep, a.K)[0]
                           for s in a.seeds]))
    tau_b = float(np.mean([E.sweep_tau(b.z[s]["val"][b.rows("val", True)],
                                       b.y("val")[b.rows("val", True)], b.prior, b.keep, b.K)[0]
                           for s in b.seeds]))
    f1 = {}
    for r, it, tau, nombre in ((a, it_a, tau_a, "ResNet"), (b, it_b, tau_b, "ViT")):
        y = r.y("test")[it]
        f1[nombre] = [E.macro_f1(E.confusion(y, E.predictions(r, s, "test", tau, True)[it], r.K), r.keep)
                      for s in r.seeds]
        print("    %-7s por semilla: %s" % (nombre, " ".join("%.4f" % v for v in f1[nombre])))
    peor_vit, mejor_resnet = min(f1["ViT"]), max(f1["ResNet"])
    print("      -> la PEOR semilla del ViT (%.4f) contra la MEJOR del ResNet (%.4f): %s"
          % (peor_vit, mejor_resnet, "ViT gana" if peor_vit > mejor_resnet else "se cruzan"))
    res["por_semilla"] = f1
    res["vit_peor_gana_a_resnet_mejor"] = bool(peor_vit > mejor_resnet)

    print("  4.4 la epoca elegida es la mejor de VALIDACION (el test no participa)")
    for run in ("e1_resnet50_kaggle", "e2_vit_b_16_kaggle",
                "e4_resnet50_fishnet", "e5_vit_b_16_fishnet"):
        filas = []
        for s_ in E.Run(run).seeds:
            with open(os.path.join(E.RUNS, run, "seed%d" % s_, "metrics.json"), encoding="utf-8") as fh:
                mj = json.load(fh)
            curva = [h["val_macro_f1"] for h in mj["hist"]]
            filas.append((mj["best_epoch"], int(np.argmax(curva)), len(curva), curva))
        coincide = all(be == am for be, am, _, _ in filas)
        c0 = filas[0][3]
        print("    %-24s epoca elegida %s · mejor de val %s · coincide: %s"
              % (run, [f[0] for f in filas], [f[1] for f in filas], "SI" if coincide else "NO"))
        print("      curva de val (semilla 0): %s" % " ".join("%.3f" % v for v in c0))
        res.setdefault("seleccion_en_val", {})[run] = {
            "coincide": bool(coincide), "epoca_elegida": [int(f[0]) for f in filas],
            "curva_val_semilla0": [float(v) for v in c0]}

    grupos = E.cluster_index(a.clusters("test")[it_a])
    ya = a.y("test")[it_a]
    pa = {s: E.predictions(a, s, "test", tau_a, True)[it_a] for s in a.seeds}
    pb = {s: E.predictions(b, s, "test", tau_b, True)[it_b] for s in b.seeds}

    def fa(rows):
        return float(np.mean([E.macro_f1(E.confusion(ya[rows], pa[s][rows], a.K), a.keep) for s in a.seeds]))

    def fb(rows):
        return float(np.mean([E.macro_f1(E.confusion(ya[rows], pb[s][rows], b.K), b.keep) for s in b.seeds]))

    for semilla in (0, 1, 2):
        d = E.paired_bootstrap(fb, fa, grupos, n_boot=1000, seed=semilla)
        print("    bootstrap semilla %d: delta %+.4f  IC95 [%+.4f, %+.4f]  P(delta>0) %.3f"
              % (semilla, d["delta_medio"], d["ic95"][0], d["ic95"][1], d["p_delta_positivo"]))
        res.setdefault("bootstrap", {})[semilla] = d
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--json", default="runs/auditoria.json")
    args = ap.parse_args()

    out = {"integridad": [(n, bool(o) if o is not None else None, v) for n, o, v in integridad()],
           "duplicados": duplicados(args.rapido),
           "sobreajuste": sobreajuste(args.rapido),
           "veracidad": veracidad()}
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("\nguardado en %s" % args.json)


if __name__ == "__main__":
    main()
