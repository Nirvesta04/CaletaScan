"""
Tabla de metricas de los cinco modelos comparados.

    python -m scripts.tabla_metricas      # -> docs/tabla-metricas.md, .html y runs/tabla_metricas.csv

No calcula nada nuevo: junta lo que ya dejaron en runs/ analisis_final.py,
investigacion.py y graficos_mazo.py (reparto oficial), mas el tamano de cada
modelo contado en su punto de control. Todo sobre la prueba limpia de FishNet
(7.159 cajas de 612 lances), promedio de 3 semillas.
"""

from __future__ import annotations

import csv
import html
import json
import os

import torch

from src.models import SPECS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
DOCS = os.path.join(ROOT, "docs")

MODELOS = [  # (etiqueta, run, arquitectura, familia, anio)
    ("ResNet-50", "e4_resnet50_fishnet", "resnet50", "CNN", 2015),
    ("ViT-B/16", "e5_vit_b_16_fishnet", "vit_b_16", "transformer", 2020),
    ("DeiT-S", "e10_deit_s_fishnet", "deit_s", "transformer", 2021),
    ("ConvNeXt-T", "e11_convnext_t_fishnet", "convnext_t", "CNN", 2022),
    ("Swin-T", "e12_swin_t_fishnet", "swin_t", "transformer", 2021),
]
PARES = ["DeiT-S - ResNet-50", "Swin-T - ConvNeXt-T", "ViT-B/16 - ResNet-50",
         "ConvNeXt-T - ResNet-50", "Swin-T - ViT-B/16"]
QUE_CONTROLA = {"DeiT-S - ResNet-50": "CNN vs transformer, mismo tamaño",
                "Swin-T - ConvNeXt-T": "CNN vs transformer, misma época",
                "ViT-B/16 - ResNet-50": "par del enunciado",
                "ConvNeXt-T - ResNet-50": "CNN moderna vs antigua",
                "Swin-T - ViT-B/16": "transformer moderno vs antiguo"}


def leer(nombre):
    with open(os.path.join(RUNS, nombre), encoding="utf-8") as fh:
        return json.load(fh)


def parametros(run):
    """Parametros entrenables contados en el punto de control (sin buffers)."""
    sd = torch.load(os.path.join(RUNS, run, "seed0", "checkpoint.pt"), map_location="cpu",
                    weights_only=False)
    fuera = ("running_mean", "running_var", "num_batches_tracked", "relative_position_index",
             "attn_mask")
    return sum(v.numel() for k, v in sd.items()
               if torch.is_tensor(v) and v.is_floating_point() and not k.endswith(fuera)) / 1e6


def c(x, n=3):
    return ("%.*f" % (n, x)).replace(".", ",").replace("-", "−")


def ic(a):
    return "%s – %s" % (c(a[0]), c(a[1]))


def con_signo(x):
    return ("+" if x >= 0 else "−") + c(abs(x))


def main():
    A = leer("analisis_final.json")
    I = leer("investigacion.json")
    OF = leer("oficial_por_modelo.json")
    V = A["por_variante"]
    T = I["tamano"]
    n_esp = sum(T[t]["cajas_especie"] for t in T)

    filas = []
    for et, run, arq, fam, anio in MODELOS:
        m = A["modelos"][run]
        acierto = sum(T[t][et]["acierto"] * T[t]["cajas_especie"] for t in T) / n_esp
        minutos = sum(e["min"] for e in A["epocas"][et]["seed0"])
        filas.append({
            "modelo": et, "familia": fam, "anio": anio,
            "imagenet_top1": SPECS[arq].imagenet_top1,
            "params_m": parametros(run),
            "macro_f1": m["macro_f1"], "ic95_lo": m["ic95"][0], "ic95_hi": m["ic95"][1],
            "por_semilla": m["por_semilla"],
            "acierto_especies": acierto,
            "macro_f1_camara_vista": V["camara_vista"]["modelos"][et]["macro_f1"],
            "macro_f1_camara_nueva": V["camara_nueva"]["modelos"][et]["macro_f1"],
            "macro_f1_atunes_agrupados": V["atunes_agrupados"]["modelos"][et]["macro_f1"],
            "macro_f1_reparto_oficial": OF[et],
            "minutos_entrenamiento_semilla0": minutos,
        })

    pares = []
    for p in PARES:
        t, n = V["todas"]["pares"][p], V["camara_nueva"]["pares"][p]
        pares.append({"comparacion": p.replace(" - ", " − "), "controla": QUE_CONTROLA[p],
                      "delta": t["delta_medio"], "ic": t["ic95"], "distingue": t["distinguible"],
                      "delta_nueva": n["delta_medio"], "ic_nueva": n["ic95"],
                      "distingue_nueva": n["distinguible"]})

    # ------------------------------------------------------------------ CSV
    with open(os.path.join(RUNS, "tabla_metricas.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = [k for k in filas[0] if k != "por_semilla"]
        w.writerow(cols + ["macro_f1_semilla0", "macro_f1_semilla1", "macro_f1_semilla2"])
        for f in filas:
            w.writerow([round(f[k], 4) if isinstance(f[k], float) else f[k] for k in cols]
                       + [round(x, 4) for x in f["por_semilla"]])

    # ------------------------------------------------------------------ tablas
    enc1 = ["Modelo", "Familia", "Año", "ImageNet top-1", "Parámetros", "Macro-F1", "Margen 95 %",
            "Semillas (0 / 1 / 2)", "Acierto 15 especies", "Cámaras vistas", "Cámaras nuevas",
            "Atunes agrupados", "Reparto oficial (con fuga)", "Min. entrenamiento"]
    t1 = [[f["modelo"], f["familia"], str(f["anio"]), c(f["imagenet_top1"], 2) + " %",
           c(f["params_m"], 1) + " M", "**%s**" % c(f["macro_f1"]), ic([f["ic95_lo"], f["ic95_hi"]]),
           " / ".join(c(x) for x in f["por_semilla"]), "%d %%" % round(100 * f["acierto_especies"]),
           c(f["macro_f1_camara_vista"]), c(f["macro_f1_camara_nueva"]),
           c(f["macro_f1_atunes_agrupados"]), c(f["macro_f1_reparto_oficial"]),
           "%d" % round(f["minutos_entrenamiento_semilla0"])] for f in filas]
    enc2 = ["Comparación", "Qué se controla", "Diferencia", "Margen 95 %", "¿Se distingue?",
            "En cámaras nuevas", "Margen 95 %", "¿Se distingue?"]
    t2 = [[p["comparacion"], p["controla"], con_signo(p["delta"]), ic(p["ic"]),
           "sí" if p["distingue"] else "no", con_signo(p["delta_nueva"]), ic(p["ic_nueva"]),
           "sí" if p["distingue_nueva"] else "no"] for p in pares]

    notas = [
        "Todo sobre la **prueba limpia de FishNet**: 7.159 cajas de 612 lances, sin lances compartidos "
        "con entrenamiento. Promedio de 3 semillas; mismo presupuesto para los cinco (9.104 pasos, lote 64, 224 px).",
        "**Macro-F1**: promedio simple del F1 de las 15 especies (cada especie pesa lo mismo), con ajuste de "
        "logits τ elegido en validación y crédito jerárquico para la clase genérica `Tuna`. Un modelo al azar "
        "obtiene ≈ 0,02.",
        "**Margen 95 %**: bootstrap por clúster (2.000 réplicas remuestreando lances completos). Las "
        "comparaciones entre modelos son **pareadas**: los mismos lances remuestreados para los dos.",
        "**Acierto 15 especies**: porcentaje de peces de las 15 especies bien clasificados (1.450 cajas).",
        "**Cámaras vistas / nuevas**: cajas de cámaras que aparecen o no en entrenamiento (31 % / 69 % de la prueba).",
        "**Reparto oficial (con fuga)**: el mismo modelo evaluado en la prueba oficial, donde el 45,8 % de las "
        "cajas comparte lance con entrenamiento. Se muestra solo para ver cuánto infla la fuga; no es el resultado.",
        "**Parámetros**: contados en el punto de control, con la cabeza de 20 clases. **Min. entrenamiento**: "
        "semilla 0 en una RTX 3060 Ti.",
    ]

    def md_tabla(enc, filas_):
        s = "| " + " | ".join(enc) + " |\n|" + "---|" * len(enc) + "\n"
        return s + "".join("| " + " | ".join(f) + " |\n" for f in filas_)

    md = ["# Tabla de métricas: los cinco modelos comparados", "",
          "Generada por `python -m scripts.tabla_metricas` a partir de `runs/*.json`. "
          "Los mismos números en formato plano: `runs/tabla_metricas.csv`.", "",
          "## Modelos", "", md_tabla(enc1, t1), "## Comparaciones pareadas", "",
          "Diferencia de macro-F1 (primero menos segundo). «Se distingue» = el margen no incluye el cero.", "",
          md_tabla(enc2, t2), "## Cómo leer la tabla", ""] + ["- " + n for n in notas] + [
          "", "## Conclusión", "",
          "A igual tamaño o época, CNN y transformer no se distinguen (DeiT-S − ResNet-50 y Swin-T − ConvNeXt-T, "
          "diferencias de 0,003 y −0,001). Lo que sí mejora el resultado es pasar de un modelo antiguo a uno "
          "moderno: ConvNeXt-T supera a ResNet-50 por 0,069, siendo los dos CNN, y en cámaras nuevas la ventaja "
          "moderna se distingue en las dos familias. Los dos mejores modelos, ConvNeXt-T (0,423) y Swin-T (0,424), "
          "empatan.", ""]
    with open(os.path.join(DOCS, "tabla-metricas.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    # ------------------------------------------------------------------ HTML (para imprimir)
    def h_tabla(enc, filas_):
        def celda(x):
            x = html.escape(x)
            return "<b>%s</b>" % x[2:-2] if x.startswith("**") else x
        return ("<table><thead><tr>" + "".join("<th>%s</th>" % html.escape(e) for e in enc) +
                "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>%s</td>" % celda(x) for x in f) +
                                                 "</tr>" for f in filas_) + "</tbody></table>")

    def h_md(s):
        import re
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    cuerpo = ("<h1>Tabla de métricas: los cinco modelos comparados</h1>"
              "<p class='sub'>Caleta-Scan · G3 · Aprendizaje Automático II, PUCV · prueba limpia de FishNet</p>"
              "<h2>Modelos</h2>" + h_tabla(enc1, t1) +
              "<h2>Comparaciones pareadas</h2><p>Diferencia de macro-F1 (primero menos segundo). "
              "«Se distingue» = el margen no incluye el cero.</p>" + h_tabla(enc2, t2) +
              "<h2>Cómo leer la tabla</h2><ul>" + "".join("<li>%s</li>" % h_md(n) for n in notas) + "</ul>" +
              "<h2>Conclusión</h2><p>%s</p>" % h_md(md[-2]))
    estilo = """
    @page{size:A4 landscape;margin:11mm}
    body{font-family:"Segoe UI",Arial,sans-serif;color:#0D2240;font-size:9.5pt;margin:0}
    h1{font-size:17pt;margin:0 0 2px} h2{font-size:12pt;margin:10px 0 4px} p{margin:4px 0}
    .sub{color:#62708A;margin:0 0 8px}
    table{border-collapse:collapse;width:100%;font-size:8.5pt}
    th,td{border-bottom:1px solid #D5DCE6;padding:3px 5px;text-align:left;vertical-align:top}
    th{background:#0D2240;color:#fff;font-weight:600}
    tr:nth-child(even) td{background:#F3F6FA}
    li{margin:1px 0} td{white-space:nowrap} th{white-space:normal} code{font-size:9pt}
    """
    with open(os.path.join(DOCS, "tabla-metricas.html"), "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><html lang='es'><head><meta charset='utf-8'>"
                 "<title>Tabla de métricas · Caleta-Scan</title><style>%s</style></head><body>%s</body></html>"
                 % (estilo, cuerpo))
    print("docs/tabla-metricas.md, docs/tabla-metricas.html y runs/tabla_metricas.csv")
    for f in filas:
        print("  %-11s %.3f  %.1fM  %d min" % (f["modelo"], f["macro_f1"], f["params_m"],
                                               f["minutos_entrenamiento_semilla0"]))


if __name__ == "__main__":
    main()
