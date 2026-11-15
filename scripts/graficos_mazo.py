"""
Graficos de resultados para la presentacion final. Sin fotos: se versionan.

    python -m scripts.graficos_mazo        # -> docs/figs/g_*.png

Lee runs/analisis_final.json y los logits guardados (para la prueba oficial y
la confusion entre atunes). Colores fijos en todo el mazo: las CNN en naranjo,
los transformers en azul; el tono oscuro es el modelo antiguo.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import data
from src.evaluate import Run, confusion, macro_f1, report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
FIGS = os.path.join(ROOT, "docs", "figs")

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13, "axes.edgecolor": "#9AA5AE",
                     "axes.labelcolor": "#2B3A45", "xtick.color": "#2B3A45",
                     "ytick.color": "#2B3A45", "savefig.dpi": 200, "savefig.transparent": True})

INK, MUT, GRID = "#16282C", "#6B7A80", "#E3E6E8"
COL = {"ResNet-50": "#B3541E", "ConvNeXt-T": "#E8913A",
       "ViT-B/16": "#1B3F73", "DeiT-S": "#7FA7D6", "Swin-T": "#2F7FC1"}
TIPO = {"ResNet-50": "CNN · 2015", "ConvNeXt-T": "CNN · 2022", "ViT-B/16": "Transformer · 2020",
        "DeiT-S": "Transformer · 2021", "Swin-T": "Transformer · 2021"}
RUN = {"ResNet-50": "e4_resnet50_fishnet", "ViT-B/16": "e5_vit_b_16_fishnet",
       "DeiT-S": "e10_deit_s_fishnet", "ConvNeXt-T": "e11_convnext_t_fishnet",
       "Swin-T": "e12_swin_t_fishnet"}

with open(os.path.join(RUNS, "analisis_final.json"), encoding="utf-8") as fh:
    A = json.load(fh)
MOD = {v["modelo"]: v for v in A["modelos"].values()}
ORDEN = ["ResNet-50", "DeiT-S", "ViT-B/16", "ConvNeXt-T", "Swin-T"]


def coma(x, d=3, s=False):
    return (("%+.*f" if s else "%.*f") % (d, x)).replace(".", ",").replace("-", "−")


def limpio(ax, ejes=("top", "right")):
    for e in ejes:
        ax.spines[e].set_visible(False)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, nombre))
    plt.close(fig)
    print("  ", nombre)


# ---------------------------------------------------------------- 1. modelos
def g_modelos():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, m in enumerate(ORDEN):
        r = MOD[m]
        ax.barh(i, r["macro_f1"], color=COL[m], height=0.62)
        ax.errorbar(r["macro_f1"], i, xerr=[[r["macro_f1"] - r["ic95"][0]], [r["ic95"][1] - r["macro_f1"]]],
                    color=INK, capsize=5, lw=1.6)
        ax.text(r["ic95"][1] + 0.008, i, coma(r["macro_f1"]), va="center", fontsize=15,
                fontweight="bold", color=INK)
    ax.set_yticks(range(len(ORDEN)), ["%s\n%s" % (m, TIPO[m]) for m in ORDEN], fontsize=13)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("macro-F1 en la prueba limpia de FishNet (0 a 1) · línea = margen al 95 %")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    limpio(ax)
    guardar(fig, "g_modelos.png")


# ---------------------------------------------------------------- 2. comparaciones
def g_pares():
    filas = [("DeiT-S − ResNet-50", "DeiT-S - ResNet-50", "mismo tamaño"),
             ("Swin-T − ConvNeXt-T", "Swin-T - ConvNeXt-T", "misma época"),
             ("ViT-B/16 − ResNet-50", "ViT-B/16 - ResNet-50", "par original"),
             ("Swin-T − ViT-B/16", "Swin-T - ViT-B/16", "transformer moderno − antiguo"),
             ("ConvNeXt-T − ResNet-50", "ConvNeXt-T - ResNet-50", "CNN moderna − antigua")]
    P = A["por_variante"]["todas"]["pares"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, (et, k, sub) in enumerate(filas[::-1]):
        d = P[k]
        c = "#0C6F66" if d["distinguible"] else "#9AA5AE"
        ax.plot(d["ic95"], [i, i], color=c, lw=5, solid_capstyle="round")
        ax.plot(d["delta_medio"], i, "o", color=c, ms=13)
        ax.text(d["ic95"][1] + 0.006, i, coma(d["delta_medio"], 3, True), va="center",
                fontsize=15, fontweight="bold", color=c)
    ax.axvline(0, color="#B3541E", ls="--", lw=1.6)
    ax.set_yticks(range(len(filas)), ["%s\n(%s)" % (et, sub) for et, _, sub in filas[::-1]], fontsize=12.5)
    ax.axhspan(1.5, 4.6, color="#EEF2F6", zorder=0)
    ax.text(-0.058, 4.38, "CNN contra transformer", ha="left", fontsize=11, color=MUT)
    ax.text(-0.058, 1.22, "moderno contra antiguo", ha="left", fontsize=11, color=MUT)
    ax.set_xlim(-0.06, 0.13)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlabel("diferencia de macro-F1 con su margen al 95 % · verde = se distingue de cero")
    limpio(ax, ("top", "right", "left"))
    ax.tick_params(axis="y", length=0)
    guardar(fig, "g_pares.png")


# ---------------------------------------------------------------- 3. camaras
def g_camaras():
    V = A["por_variante"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, m in enumerate(ORDEN):
        a = V["camara_vista"]["modelos"][m]["macro_f1"]
        b = V["camara_nueva"]["modelos"][m]["macro_f1"]
        ax.plot([b, a], [i, i], color=GRID, lw=6, zorder=1)
        ax.scatter([a], [i], s=170, color=COL[m], zorder=2, edgecolor="white", lw=1.5)
        ax.scatter([b], [i], s=170, color="white", zorder=2, edgecolor=COL[m], lw=2.5)
        ax.text(a + 0.01, i, coma(a), va="center", fontsize=13, color=INK)
        ax.text(b - 0.01, i, coma(b), va="center", ha="right", fontsize=13, color=INK, fontweight="bold")
    ax.set_yticks(range(len(ORDEN)), ORDEN, fontsize=14)
    ax.set_xlim(0.15, 0.5)
    ax.scatter([], [], s=120, color="#555", label="cámaras vistas en entrenamiento")
    ax.scatter([], [], s=120, color="white", edgecolor="#555", lw=2, label="cámaras nuevas (barcos no vistos)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=12)
    ax.set_xlabel("macro-F1 en la prueba limpia")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    limpio(ax, ("top", "right", "left"))
    ax.tick_params(axis="y", length=0)
    guardar(fig, "g_camaras.png")


# ---------------------------------------------------------------- 4. especies
def g_especies():
    pe = A["por_especie"]
    esp = sorted(pe["ResNet-50"], key=lambda k: -pe["ResNet-50"][k]["cajas"])
    M = np.array([[pe[m][e]["f1"] for m in ORDEN] for e in esp])
    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(esp)):
        for j in range(len(ORDEN)):
            ax.text(j, i, coma(M[i, j], 2), ha="center", va="center", fontsize=11.5,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(ORDEN)), ORDEN, fontsize=12.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(esp)), ["%s (%d)" % (e, round(pe["ResNet-50"][e]["cajas"])) for e in esp],
                  fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    guardar(fig, "g_especies.png")


# ---------------------------------------------------------------- 5. atunes y 6. fuga
def g_atunes_y_fuga():
    atunes = ["Albacore", "Yellowfin tuna", "Bigeye tuna", "Skipjack tuna"]
    of, li = {}, {}
    for m in ORDEN:
        r = Run(RUN[m])
        rep_o = report(r, "test", True, True, n_boot=1, solo_limpio=False)
        of[m] = rep_o["macro_f1"]["media"]
        li[m] = MOD[m]["macro_f1"]
        if m == "ConvNeXt-T":
            rep = report(r, "test", True, True, n_boot=1, solo_limpio=True)
            cm = sum(confusion(rep["_y"], p, r.K) for p in rep["_preds"].values()) / len(rep["_preds"])
            idx = [r.classes.index(a) for a in atunes]
            sub = cm[np.ix_(idx, range(r.K))]
            otros = sub.sum(1) - sub[:, idx].sum(1)
            T = np.c_[sub[:, idx], otros] / sub.sum(1, keepdims=True)

    # atunes
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(T, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            ax.text(j, i, "%d %%" % round(100 * T[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if T[i, j] > 0.55 else INK)
    nombres = ["Albacora", "Aleta\namarilla", "Patudo", "Barrilete"]
    ax.set_xticks(range(5), nombres + ["otra\nclase"], fontsize=12.5)
    ax.set_yticks(range(4), ["%s (%d)" % (n.replace("\n", " "), round(sub[i].sum())) for i, n in enumerate(nombres)],
                  fontsize=12.5)
    ax.set_xlabel("lo que dice ConvNeXt-T")
    ax.set_ylabel("atún real (cajas de prueba)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    guardar(fig, "g_atunes.png")

    # fuga: oficial contra limpio
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(ORDEN))
    ax.bar(x - 0.2, [of[m] for m in ORDEN], 0.38, color="#C9CFD4", label="reparto oficial (con fuga)")
    ax.bar(x + 0.2, [li[m] for m in ORDEN], 0.38, color=[COL[m] for m in ORDEN], label="prueba limpia")
    for i, m in enumerate(ORDEN):
        ax.text(i - 0.2, of[m] + 0.01, coma(of[m], 2), ha="center", fontsize=12, color=MUT)
        ax.text(i + 0.2, li[m] + 0.01, coma(li[m], 2), ha="center", fontsize=12, color=INK, fontweight="bold")
    ax.set_xticks(x, ORDEN, fontsize=13)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    limpio(ax)
    guardar(fig, "g_fuga.png")
    return of


# ---------------------------------------------------------------- 7. kaggle
def g_kaggle():
    with open(os.path.join(RUNS, "comparacion_e7.json"), encoding="utf-8") as fh:
        E7 = json.load(fh)
    val = {"ResNet-50": [E7["e1_resnet50_kaggle__vs__e7_resnet50_solo_fondo"]["macro_f1_a"],
                         E7["e1_resnet50_kaggle__vs__e7_resnet50_solo_fondo"]["macro_f1_b"],
                         E7["e1_resnet50_kaggle__vs__e7_resnet50_solo_pez"]["macro_f1_b"]],
           "ViT-B/16": [E7["e2_vit_b_16_kaggle__vs__e7_vit_b_16_solo_fondo"]["macro_f1_a"],
                        E7["e2_vit_b_16_kaggle__vs__e7_vit_b_16_solo_fondo"]["macro_f1_b"],
                        E7["e2_vit_b_16_kaggle__vs__e7_vit_b_16_solo_pez"]["macro_f1_b"]]}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(3)
    for k, (m, v) in enumerate(val.items()):
        ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=COL[m], label=m)
        for i, y in enumerate(v):
            ax.text(i + (k - 0.5) * 0.38, y + 0.004, coma(y), ha="center", fontsize=12, color=INK)
    ax.set_xticks(x, ["foto completa", "sin el pez", "solo el pez"], fontsize=14)
    ax.set_ylim(0.9, 1.02)
    ax.set_ylabel("macro-F1 en Kaggle")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=12)
    limpio(ax)
    guardar(fig, "g_kaggle.png")


# ---------------------------------------------------------------- 8. no se
def g_nose():
    with open(os.path.join(RUNS, "chile_ood.json"), encoding="utf-8") as fh:
        CH = json.load(fh)
    with open(os.path.join(RUNS, "calibracion_ood.json"), encoding="utf-8") as fh:
        CAL = json.load(fh)

    def ood(run):
        k = [k for k in CAL["ood"] if k.startswith(run) and k.endswith("id_sin_human")][0]
        return CAL["ood"][k]["energia"]
    filas = [("Especies chilenas\nrechazadas", [CH["modelos"][m]["rechaza_total"] for m in ("ResNet-50", "ViT-B/16")]),
             ("Peces de Kaggle\nrechazados", [1 - ood("e4")["fpr95"], 1 - ood("e5")["fpr95"]])]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(filas))
    for k, m in enumerate(("ResNet-50", "ViT-B/16")):
        v = [f[1][k] for f in filas]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.36, color=COL[m], label=m)
        for i, y in enumerate(v):
            ax.text(i + (k - 0.5) * 0.38, y + 0.015, "%d %%" % round(100 * y), ha="center", fontsize=15,
                    fontweight="bold", color=INK)
    ax.set_xticks(x, [f[0] for f in filas], fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_ylabel("responde «no sé»")
    ax.legend(frameon=False, loc="upper left", fontsize=12)
    limpio(ax)
    guardar(fig, "g_nose.png")


# ---------------------------------------------------------------- 9. composicion
def g_composicion():
    dss = data.fishnet()
    tr, te = dss["train"], dss["test"]
    y = te.labels[data.clean_mask(tr, te)]
    c = np.bincount(y, minlength=len(tr.classes))
    nombres = tr.classes
    grupos = {"Personas": c[nombres.index("Human")],
              "Sin pez": c[nombres.index("No fish")],
              "Pez sin especie\n(desconocido, atún, raros)": sum(c[nombres.index(k)] for k in ("Unknown", "Tuna", "Other_rare")),
              "Las 15 especies": sum(c[i] for i in range(len(nombres)) if tr.species_mask()[i])}
    fig, ax = plt.subplots(figsize=(10, 3.2))
    izq, cols = 0, ["#9AA5AE", "#C9CFD4", "#7FA7D6", "#1B3F73"]
    tot = sum(grupos.values())
    for (k, v), col in zip(grupos.items(), cols):
        ax.barh(0, v / tot, left=izq, color=col, height=0.6)
        if v / tot > 0.06:
            ax.text(izq + v / tot / 2, 0, "%d %%" % round(100 * v / tot), ha="center", va="center",
                    fontsize=17, fontweight="bold", color="white" if col in ("#1B3F73", "#9AA5AE") else INK)
        izq += v / tot
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    hand = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(hand, ["%s · %d cajas" % (k.replace("\n", " "), v) for k, v in grupos.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False, fontsize=12)
    guardar(fig, "g_composicion.png")


def main():
    os.makedirs(FIGS, exist_ok=True)
    g_modelos()
    g_pares()
    g_camaras()
    g_especies()
    of = g_atunes_y_fuga()
    g_kaggle()
    g_nose()
    g_composicion()
    with open(os.path.join(RUNS, "oficial_por_modelo.json"), "w", encoding="utf-8") as fh:
        json.dump(of, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
