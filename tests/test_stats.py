"""
Validacion de la maquinaria estadistica de evaluate.py.

    python -m tests.test_stats

No necesita GPU ni corridas entrenadas: contrasta contra sklearn y contra el
IC binomial teorico. La comprobacion que mas importa es la del factor de
diseno: con 100 clusteres de 20 filas identicas el bootstrap tiene que
devolver el IC de n=100, no el de n=2000. Si esa falla, todos los intervalos
del informe estan mal.
"""
import os
import sys


import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import evaluate as E


def main():
    rng = np.random.default_rng(0)
    ok = []

    # ---- 1. confusion vs sklearn ----
    K = 7
    y = rng.integers(0, K, 3000)
    p = np.where(rng.random(3000) < 0.75, y, rng.integers(0, K, 3000))
    cm_mio = E.confusion(y, p, K)
    cm_sk = confusion_matrix(y, p, labels=range(K))
    ok.append(("confusion == sklearn", np.array_equal(cm_mio, cm_sk)))

    # ---- 2. macro_f1 vs sklearn (todas las clases con soporte) ----
    keep = np.ones(K, bool)
    mio = E.macro_f1(cm_mio, keep)
    sk = f1_score(y, p, average="macro", labels=range(K), zero_division=0)
    ok.append(("macro_f1 == sklearn (%.6f vs %.6f)" % (mio, sk), abs(mio - sk) < 1e-9))

    # subconjunto de clases
    keep2 = np.zeros(K, bool); keep2[[1, 3, 5]] = True
    mio2 = E.macro_f1(cm_mio, keep2)
    sk2 = f1_score(y, p, average="macro", labels=[1, 3, 5], zero_division=0)
    ok.append(("macro_f1 sobre subconjunto (%.6f vs %.6f)" % (mio2, sk2), abs(mio2 - sk2) < 1e-9))

    # ---- 3. balanced_acc vs sklearn ----
    mio3 = E.balanced_acc(cm_mio, keep)
    sk3 = balanced_accuracy_score(y, p)
    ok.append(("balanced_acc == sklearn (%.6f vs %.6f)" % (mio3, sk3), abs(mio3 - sk3) < 1e-9))

    # ---- 4. cluster_index parte bien ----
    cl = np.array([5, 1, 5, 3, 1, 1])
    g = E.cluster_index(cl)
    recon = sorted(sorted(x.tolist()) for x in g)
    ok.append(("cluster_index agrupa correcto", recon == [[0, 2], [1, 4, 5], [3]]))
    ok.append(("cluster_index cubre todas las filas",
               sorted(np.concatenate(g).tolist()) == list(range(6))))

    # ---- 5. bootstrap SIN clustering reproduce el IC binomial ----
    n, acc_true = 2000, 0.80
    y2 = np.zeros(n, int)
    p2 = (rng.random(n) > acc_true).astype(int)      # 0 = acierto
    grupos_ind = E.cluster_index(np.arange(n))       # 1 fila por cluster = independiente
    f = lambda rows: float((p2[rows] == y2[rows]).mean())
    lo, hi = E.bootstrap(f, grupos_ind, n_boot=3000, seed=1)
    acc = f(np.arange(n))
    teo = 1.96 * np.sqrt(acc * (1 - acc) / n)
    ancho = (hi - lo) / 2
    ok.append(("bootstrap sin clustering ~ IC binomial (%.4f vs %.4f teorico)" % (ancho, teo),
               abs(ancho - teo) < 0.15 * teo))

    # ---- 6. el clustering ENSANCHA el IC por el factor de diseno ----
    # 100 clusteres de 20 filas identicas -> n efectivo = 100, no 2000
    m_cl, tam = 100, 20
    base = (rng.random(m_cl) > acc_true).astype(int)
    p3 = np.repeat(base, tam)
    y3 = np.zeros(m_cl * tam, int)
    cl3 = np.repeat(np.arange(m_cl), tam)
    f3 = lambda rows: float((p3[rows] == y3[rows]).mean())
    lo3, hi3 = E.bootstrap(f3, E.cluster_index(cl3), n_boot=3000, seed=1)
    ancho_cl = (hi3 - lo3) / 2
    a3 = f3(np.arange(m_cl * tam))
    teo_cl = 1.96 * np.sqrt(a3 * (1 - a3) / m_cl)      # n efectivo = 100 clusteres
    teo_ing = 1.96 * np.sqrt(a3 * (1 - a3) / (m_cl * tam))   # el ingenuo, mal
    ok.append(("IC con clusteres ~ n efectivo=%d (%.4f vs %.4f)" % (m_cl, ancho_cl, teo_cl),
               abs(ancho_cl - teo_cl) < 0.20 * teo_cl))
    ok.append(("...y NO el ingenuo n=%d (%.4f), factor %.1fx"
               % (m_cl * tam, teo_ing, ancho_cl / teo_ing), ancho_cl > 3 * teo_ing))

    # ---- 7. bootstrap pareado: modelos identicos -> delta 0, IC degenerado ----
    r = E.paired_bootstrap(f3, f3, E.cluster_index(cl3), n_boot=500, seed=2)
    ok.append(("pareado con modelos identicos da delta=0 exacto",
               r["delta_medio"] == 0.0 and r["ic95"] == [0.0, 0.0]))

    # ---- 8. credito jerarquico ----
    clases = ["Albacore", "Human", "Tuna", "Yellowfin tuna"]
    idx = {c: i for i, c in enumerate(clases)}
    yh = np.array([idx["Tuna"], idx["Tuna"], idx["Albacore"], idx["Tuna"]])
    ph = np.array([idx["Yellowfin tuna"], idx["Human"], idx["Yellowfin tuna"], idx["Tuna"]])
    out = E.hierarchical_credit(ph, yh, clases)
    esperado = np.array([idx["Tuna"],            # atun especifico sobre verdad Tuna -> credito
                         idx["Human"],           # no es atun -> sin cambio
                         idx["Yellowfin tuna"],  # verdad no es Tuna -> sin cambio
                         idx["Tuna"]])           # ya acertaba
    ok.append(("credito jerarquico solo toca el caso correcto", np.array_equal(out, esperado)))

    # ---- 9. ajuste de logits: tau=0 no cambia nada; prior uniforme tampoco ----
    z = rng.normal(size=(500, 4))
    prior = np.array([0.7, 0.2, 0.06, 0.04])
    ok.append(("logit_adjust con tau=0 es identidad",
               np.array_equal(E.logit_adjust(z, prior, 0.0).argmax(1), z.argmax(1))))
    uni = np.full(4, 0.25)
    ok.append(("prior uniforme no cambia el argmax",
               np.array_equal(E.logit_adjust(z, uni, 1.0).argmax(1), z.argmax(1))))
    # tau grande favorece a la clase rara
    frac0 = (E.logit_adjust(z, prior, 0.0).argmax(1) == 3).mean()
    frac2 = (E.logit_adjust(z, prior, 2.0).argmax(1) == 3).mean()
    ok.append(("tau alto sube la prediccion de la clase rara (%.2f -> %.2f)" % (frac0, frac2),
               frac2 > frac0))

    # ---- 10. macro_f1 ignora clases sin soporte ----
    cm = np.zeros((3, 3), int); cm[0, 0] = 10; cm[1, 1] = 10   # clase 2 sin soporte
    keep3 = np.ones(3, bool)
    ok.append(("macro_f1 ignora la clase ausente (=1.0, no 0.667)",
               abs(E.macro_f1(cm, keep3) - 1.0) < 1e-9))

    print("\n%-62s %s" % ("COMPROBACION", "OK"))
    print("-" * 70)
    for nombre, res in ok:
        print("%-62s %s" % (nombre[:62], "si" if res else "NO <-- FALLA"))
    print("-" * 70)
    print("%d/%d correctas" % (sum(r for _, r in ok), len(ok)))


if __name__ == "__main__":
    main()
