"""
Velocidad contra precision (§11 del protocolo).

    python -m src.speed          # ~15 min; usa GPU y CPU a fondo: con la maquina libre

Lo que importa para el muelle es la INFERENCIA, no el entrenamiento. Protocolo:

- model.eval() + torch.inference_mode(), sin el parche de atencion de
  interpret.py: con el, el ViT pierde el kernel fusionado y se mide un modelo
  mas lento que el que se desplegaria.
- 30 iteraciones de calentamiento y torch.cuda.synchronize() antes y despues
  de cada medicion. Sin sincronizar se cronometra el encolado asincrono.
- 200 repeticiones por escenario; se reporta mediana y p95, no la media: la
  arrastran los atipicos del sistema operativo.
- Los modelos se miden INTERCALADOS en rondas de 20, rotando el orden. Si la
  carga de la maquina cambia a mitad de la medicion (es un PC de uso diario),
  el cambio cae sobre todos por igual y no sesga la comparacion.
- Escenarios: GPU batch 1 (latencia de la demo), GPU batch 32 (lote), CPU
  batch 1 (una caleta sin GPU). En GPU, fp32 y fp16 (autocast, que es como se
  evaluo todo el proyecto). En CPU solo fp32.
- Sin torch.compile, channels_last, cudnn.benchmark, ONNX ni int8: benefician
  de forma desigual a cada arquitectura y quedan fuera del numero principal.

ViT-B/32 entra como tercer punto de costo con sus pesos de ImageNet y la misma
cabeza de 20 clases: el costo no depende de los pesos, y no tiene precision en
Fishnet porque no se entreno (E9).
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from . import data, models
from . import evaluate as E

RUNS = E.RUNS
WARMUP, REPS, RONDA = 30, 200, 20

MODELOS = (("resnet50", "e4_resnet50_fishnet"),
           ("vit_b_16", "e5_vit_b_16_fishnet"),
           ("vit_b_32", None))

ESCENARIOS = (("gpu", 1, "fp32"), ("gpu", 1, "fp16"),
              ("gpu", 32, "fp32"), ("gpu", 32, "fp16"),
              ("cpu", 1, "fp32"))


def cargar(nombre: str, run: str | None, K: int = 20) -> torch.nn.Module:
    m = models.build_model(nombre, K)
    if run:
        ck = os.path.join(RUNS, run, "seed0", "checkpoint.pt")
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
    return m.eval()


def macs(m: torch.nn.Module) -> float:
    """
    MACs por imagen a 224 px, MEDIDOS con FlopCounterMode (FLOPs / 2).

    Se cuenta con el gradiente activo a proposito: en inferencia el ViT toma
    la ruta rapida de nn.MultiheadAttention (_native_multi_head_attention),
    que el contador no conoce, y la atencion entera desapareceria del conteo.
    Con gradiente toma la ruta explicita y cada matmul se cuenta.
    """
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.zeros(1, 3, 224, 224)
    with torch.enable_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 2


def tamanos(m: torch.nn.Module) -> dict:
    """Tamano serializado del state_dict en fp32 y en fp16, en MB."""
    out = {}
    for tipo in ("fp32", "fp16"):
        sd = {k: (v.half() if tipo == "fp16" and v.is_floating_point() else v)
              for k, v in m.state_dict().items()}
        b = io.BytesIO()
        torch.save(sd, b)
        out[tipo] = b.tell() / 2 ** 20
    return out


def _paso(m, x, fp16):
    if fp16:
        with torch.autocast("cuda", dtype=torch.float16):
            return m(x)
    return m(x)


def medir(mods: dict, dev: str, batch: int, prec: str) -> dict[str, list[float]]:
    """Latencias en ms por llamada, con los modelos intercalados por rondas."""
    fp16 = prec == "fp16"
    gpu = dev == "cuda"
    x = torch.randn(batch, 3, 224, 224, device=dev)

    def sync():
        if gpu:
            torch.cuda.synchronize()

    t = {k: [] for k in mods}
    with torch.inference_mode():
        for m in mods.values():
            for _ in range(WARMUP):
                _paso(m, x, fp16)
            sync()
        orden = list(mods)
        for r in range(REPS // RONDA):
            for k in orden[r % len(orden):] + orden[:r % len(orden)]:
                m = mods[k]
                for _ in range(2):                  # reentrada tras cambiar de modelo
                    _paso(m, x, fp16)
                sync()
                for _ in range(RONDA):
                    t0 = time.perf_counter()
                    _paso(m, x, fp16)
                    sync()
                    t[k].append((time.perf_counter() - t0) * 1e3)
    return t


def resumen(v: list[float], batch: int) -> dict:
    a = np.asarray(v)
    med = float(np.median(a))
    return {"mediana_ms": med, "p95_ms": float(np.percentile(a, 95)),
            "media_ms": float(a.mean()), "n": int(len(a)),
            "img_por_s": batch * 1e3 / med}


def preproceso(n: int = 200, seed: int = 0) -> dict:
    """
    Costo de preparar UN recorte de Fishnet en CPU: abrir el JPEG, recortar con
    margen y transformar. Es identico para las dos arquitecturas, asi que no
    entra en la comparacion; es el contexto de la latencia total por pez.
    """
    ds = data.fishnet()["test"]
    tf = data.build_transform(train=False)
    rows = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    t = []
    for i in rows:
        t0 = time.perf_counter()
        img = Image.open(os.path.join(data.ROOT, ds.paths[i])).convert("RGB")
        tf(img.crop(ds._margin_box(i, img.size)))
        t.append((time.perf_counter() - t0) * 1e3)
    return resumen(t, 1)


def _cpu_nombre() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def precision_fishnet() -> dict:
    """Macro-F1 del test limpio, tal como lo reporta evaluate.py (tau y jerarquia)."""
    out = {}
    for nombre, run in MODELOS:
        if run:
            r = E.report(E.Run(run), solo_limpio=True)
            out[nombre] = {"macro_f1": r["macro_f1"]["media"],
                           "ic95": r["macro_f1"]["ic95_bootstrap"]}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="runs/velocidad.json")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    info = {"gpu": torch.cuda.get_device_name(0), "cpu": _cpu_nombre(),
            "hilos_cpu": torch.get_num_threads(), "torch": torch.__version__,
            "warmup": WARMUP, "reps": REPS, "ronda": RONDA,
            "cudnn_benchmark": False}
    print("GPU %s · CPU %s · %d hilos · torch %s"
          % (info["gpu"], info["cpu"], info["hilos_cpu"], info["torch"]))

    mods = {n: cargar(n, r) for n, r in MODELOS}
    costo = {}
    for n, m in mods.items():
        costo[n] = {"params_m": sum(p.numel() for p in m.parameters()) / 1e6,
                    "macs_g": macs(m) / 1e9,
                    "macs_g_torchvision": models.SPECS[n].macs_g,
                    "mb": tamanos(m)}
        print("  %-9s %6.2f M params · %5.2f GMACs medidos (torchvision %.2f) · "
              "%5.1f MB fp32 / %5.1f MB fp16"
              % (n, costo[n]["params_m"], costo[n]["macs_g"], models.SPECS[n].macs_g,
                 costo[n]["mb"]["fp32"], costo[n]["mb"]["fp16"]))

    lat = {}
    for dev, batch, prec in ESCENARIOS:
        clave = "%s · batch %d · %s" % (dev.upper(), batch, prec)
        d = "cuda" if dev == "gpu" else "cpu"
        for m in mods.values():
            m.to(d)
        t = medir(mods, d, batch, prec)
        lat[clave] = {n: resumen(v, batch) for n, v in t.items()}
        print("\n%s" % clave)
        for n, r in lat[clave].items():
            print("  %-9s mediana %8.2f ms · p95 %8.2f ms · %8.1f img/s"
                  % (n, r["mediana_ms"], r["p95_ms"], r["img_por_s"]))
        if d == "cuda":
            torch.cuda.empty_cache()

    pre = preproceso()
    print("\npreproceso de un recorte (CPU): mediana %.2f ms · p95 %.2f ms"
          % (pre["mediana_ms"], pre["p95_ms"]))

    prec = precision_fishnet()
    for n, p in prec.items():
        print("  %-9s macro-F1 limpio %.4f [%.4f, %.4f]" % (n, p["macro_f1"], *p["ic95"]))

    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"maquina": info, "costo": costo, "latencia": lat,
                   "preproceso_recorte_cpu": pre, "precision_fishnet_limpio": prec},
                  fh, indent=2, ensure_ascii=False)
    print("\nguardado en %s" % args.json)


if __name__ == "__main__":
    main()
