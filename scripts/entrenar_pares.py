"""
Entrena los modelos nuevos (E10 DeiT-S, E11 ConvNeXt-T, E12 Swin-T), uno tras
otro, y se puede detener y retomar cuantas veces haga falta.

    python -m scripts.entrenar_pares            # empieza o retoma
    python -m scripts.entrenar_pares --parar    # pide parar al terminar la epoca

Retomar no pierde trabajo: train.py guarda un punto de control al final de cada
epoca y salta las semillas ya terminadas. Al pedir parar se crea runs/STOP; el
entrenamiento termina la epoca en curso (unos 5 minutos como maximo), la
guarda y se detiene, y no arranca el siguiente modelo.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = os.path.join(ROOT, "runs", "STOP")
CONFIGS = ["e10_deit_s_fishnet", "e11_convnext_t_fishnet", "e12_swin_t_fishnet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parar", action="store_true", help="detener al terminar la epoca en curso")
    args = ap.parse_args()

    if args.parar:
        open(STOP, "w").close()
        print("pedido: se detiene al terminar la epoca en curso")
        return

    if os.path.exists(STOP):
        os.remove(STOP)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    for c in CONFIGS:
        if os.path.exists(STOP):
            break
        log = os.path.join(ROOT, "logs", c + ".log")
        print("== %s  (log: logs/%s.log)" % (c, c), flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, "-u", "-m", "src.train",
                                "--config", "configs/%s.yaml" % c],
                               cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == 3 or os.path.exists(STOP):
            print("detenido a pedido; se retoma con el mismo comando")
            break
        if r.returncode != 0:
            print("FALLO %s (codigo %d); ver el log" % (c, r.returncode))
            break
    else:
        print("LISTO: los tres modelos terminaron")
    if os.path.exists(STOP):
        os.remove(STOP)


if __name__ == "__main__":
    main()
