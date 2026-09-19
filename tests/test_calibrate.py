"""
Validacion de calibrate.py.

    python -m tests.test_calibrate

Sin GPU ni corridas: contrasta contra sklearn y contra casos con respuesta
conocida. La que mas importa es la recuperacion de T: si las etiquetas se
muestrean de softmax(z/T0), el estimador de maxima verosimilitud tiene que
devolver T0.
"""
import os
import sys

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import calibrate as C
from src import evaluate as E


def muestrear(p, rng):
    u = rng.random(len(p))[:, None]
    return (p.cumsum(1) < u).sum(1)


def main():
    rng = np.random.default_rng(0)
    ok = []

    # ---- 1. recupera la temperatura verdadera ----
    z = rng.normal(0, 3, (40000, 6))
    for T0 in (0.5, 2.0):
        y = muestrear(C.softmax(z / T0), rng)
        T, lim = C.fit_temperature(z, y)
        ok.append(("fit_temperature recupera T0=%.1f (%.3f)" % (T0, T),
                   abs(T / T0 - 1) < 0.03 and not lim))

    # ---- 2. validacion perfecta: T no identificada, se avisa ----
    zz = rng.normal(0, 1, (500, 4)); yy = zz.argmax(1); zz[np.arange(500), yy] += 20
    T, lim = C.fit_temperature(zz, yy)
    ok.append(("val perfecta -> T en el limite y marcada", lim and np.isclose(T, C.T_MIN)))

    # ---- 3. NLL == sklearn log_loss ----
    y = muestrear(C.softmax(z), rng)
    mio = C.nll_rows(z, y).mean()
    sk = log_loss(y, C.softmax(z), labels=range(6))
    ok.append(("nll == sklearn log_loss (%.6f vs %.6f)" % (mio, sk), abs(mio - sk) < 1e-9))

    # ---- 4. Brier en casos a mano ----
    p = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]); yb = np.array([0, 0, 0])
    ok.append(("brier en casos a mano (0 · 0,5 · 2)",
               np.allclose(C.brier_rows(p, yb), [0.0, 0.5, 2.0])))

    # ---- 5. ECE: calibrado ~0, descalibrado conocido exacto ----
    pc = C.softmax(z); yc = muestrear(pc, rng)
    e_cal = C.ece_from(pc.max(1), (pc.argmax(1) == yc).astype(float))
    ok.append(("ECE de un modelo calibrado ~ 0 (%.4f)" % e_cal, e_cal < 0.01))
    conf = np.full(1000, 0.9); cor = np.r_[np.ones(600), np.zeros(400)]
    ok.append(("ECE conf 0,9 / acc 0,6 = 0,30 exacto", abs(C.ece_from(conf, cor) - 0.3) < 1e-12))
    ok.append(("conf = 1,0 cae en el ultimo bin, no fuera",
               abs(C.ece_from(np.ones(10), np.ones(10))) < 1e-12))

    # ---- 6. AUROC == sklearn, con empates ----
    a = np.round(rng.normal(1, 1, 700), 1); b = np.round(rng.normal(0, 1, 900), 1)
    sk = roc_auc_score(np.r_[np.ones(700), np.zeros(900)], np.r_[a, b])
    ok.append(("auroc == sklearn con empates (%.6f vs %.6f)" % (C.auroc(a, b), sk),
               abs(C.auroc(a, b) - sk) < 1e-12))

    # ---- 7. FPR@95 en un caso a mano ----
    # ID 0..99: el umbral que conserva 95 de 100 es 5. El OOD en 5,0 empata y
    # cuenta como aceptado; 4,5 no pasa. FPR = 2/4.
    s_id = np.arange(100, dtype=float)
    s_ood = np.array([3.5, 4.5, 5.0, 100.0])
    ok.append(("fpr_at_tpr: umbral 5, empate aceptado (%.2f)" % C.fpr_at_tpr(s_id, s_ood),
               abs(C.fpr_at_tpr(s_id, s_ood) - 0.5) < 1e-12))

    # ---- 8. la MSP en log no satura ----
    zs = np.array([[50.0, 0, 0], [200.0, 0, 0]])
    msp32 = C.softmax(zs.astype(np.float32)).max(1)
    ok.append(("MSP float32 satura (empata 50 y 200)", msp32[0] == msp32[1]))
    ls = C.score_msp(zs)
    ok.append(("score_msp distingue margen 50 de 200", ls[1] > ls[0]))
    zr = rng.normal(0, 2, (2000, 5))
    ok.append(("score_msp mismo orden que la MSP",
               np.array_equal(np.argsort(C.score_msp(zr)), np.argsort(C.softmax(zr).max(1)))))

    # ---- 9. bootstrap en dos muestras respeta los clusteres ----
    # 50 clusteres de 20 copias identicas a cada lado: el IC tiene que ser el de
    # n=50, bastante mas ancho que el de tratar las 1.000 filas como independientes
    sa = np.repeat(rng.normal(0.5, 1, 50), 20); sb = np.repeat(rng.normal(0, 1, 50), 20)
    ga = E.cluster_index(np.repeat(np.arange(50), 20))
    gb = E.cluster_index(np.repeat(np.arange(50), 20))
    g1 = [np.array([i]) for i in range(1000)]
    f = lambda ri, ro: C.auroc(sa[ri], sb[ro])
    w_cl = np.subtract(*C.ic(C.bootstrap2(f, ga, gb, 400))[::-1])
    w_fl = np.subtract(*C.ic(C.bootstrap2(f, g1, g1, 400))[::-1])
    ok.append(("bootstrap2 por clusteres ~sqrt(20)x mas ancho (%.1fx)" % (w_cl / w_fl),
               w_cl / w_fl > 3))

    print("\n%-62s %s" % ("COMPROBACION", "OK"))
    print("-" * 70)
    for nombre, res in ok:
        print("%-62s %s" % (nombre[:62], "si" if res else "NO <-- FALLA"))
    print("-" * 70)
    print("%d/%d correctas" % (sum(r for _, r in ok), len(ok)))


if __name__ == "__main__":
    main()
