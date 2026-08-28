"""
Entrenamiento dirigido por config.

    python -m src.train --config configs/e1_resnet50_kaggle.yaml

Un YAML es un experimento. Cada corrida deja en runs/<name>/seed<k>/:

    config.yaml      el config exacto con que se corrio, con los overrides
    metrics.json     historia por epoca + metricas finales + entorno + commit
    val_logits.npy   logits crudos de validacion, alineados al orden del CSV
    test_logits.npy  idem test
    checkpoint.pt    pesos del mejor modelo segun macro-F1 de validacion

Guardar los logits crudos es la decision de mayor rendimiento del proyecto: el
ajuste de logits, el escalado de temperatura, el bootstrap de clusters, la
estratificacion por oclusion y el AUROC de OOD se recalculan todos desde
val_logits.npy / test_logits.npy en segundos, sin volver a tocar la GPU.

OJO: este script usa DataLoader con workers. En Windows arrancan por spawn, que
reimporta el modulo principal, por eso el guard __main__ del final NO es
opcional. Ver el docstring de data.make_loader().
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn

from . import data, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")

DEFAULTS = {
    "model": "resnet50",
    "dataset": "kaggle",          # kaggle | kaggle_random | fishnet
    "label_col": None,            # None -> el propio del dataset
    "purge": False,               # solo fishnet
    "epochs": 30,
    "total_steps": None,          # si se fija, manda sobre epochs
    "batch_size": 64,
    "accum_steps": 1,             # batch efectivo = batch_size * accum_steps
    "lr": 3.0e-5,
    "head_lr_mult": 10.0,
    "llrd": 0.70,
    "weight_decay": 0.05,
    "warmup_frac": 0.03,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
    "num_workers": 8,
    "seeds": [0, 1, 2],
    "amp": True,
}


# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "desconocido"


def build_datasets(cfg):
    kind = cfg["dataset"]
    if kind == "kaggle":
        return data.kaggle()
    if kind == "kaggle_random":
        return data.kaggle_random_split(seed=0)   # el split con fuga, para E3
    if kind == "fishnet":
        col = cfg["label_col"] or "label_model"
        return data.fishnet(label_col=col, purge=cfg["purge"])
    raise ValueError("dataset desconocido: %s" % kind)


def lr_schedule(step: int, total: int, warmup: int) -> float:
    """Warmup lineal y despues coseno hasta 0.

    Se aplica como multiplicador sobre el lr inicial de CADA grupo, asi que las
    razones del decaimiento por capa del ViT se conservan intactas.
    """
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def infer(model, loader, n_rows: int, n_classes: int, device) -> np.ndarray:
    """Logits alineados al orden del CSV: out[i] corresponde a ds.df.iloc[i].

    Se rellena por indice y no por orden de llegada, para que el resultado no
    dependa de como el DataLoader arme los lotes.
    """
    model.eval()
    out = np.full((n_rows, n_classes), np.nan, dtype=np.float32)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            z = model(x)
        out[idx.numpy()] = z.float().cpu().numpy()
    assert not np.isnan(out).any(), "quedaron filas sin logits"
    return out


def macro_f1(logits: np.ndarray, y: np.ndarray, keep: np.ndarray | None) -> float:
    """Macro-F1. `keep` restringe el promedio a un subconjunto de clases.

    En Fishnet se restringe a las especies reportables: promediar Human dentro
    del macro-F1 lo infla, porque es trivialmente facil y es el 54% del train.
    """
    pred = logits.argmax(1)
    labels = np.where(keep)[0] if keep is not None else None
    return float(f1_score(y, pred, average="macro", labels=labels, zero_division=0))


# ---------------------------------------------------------------------------

def run_one(cfg: dict, seed: int, device: str = "cuda") -> dict:
    t0 = time.time()
    data.set_seed(seed)

    dss = build_datasets(cfg)
    tr, va, te = dss["train"], dss["val"], dss["test"]
    n_classes = len(tr.classes)
    keep = tr.species_mask() if cfg["dataset"] == "fishnet" else None

    tl = data.make_loader(tr, cfg["batch_size"], True, cfg["num_workers"], seed)
    vl = data.make_loader(va, cfg["batch_size"], False, cfg["num_workers"], seed)
    el = data.make_loader(te, cfg["batch_size"], False, cfg["num_workers"], seed)

    model = models.build_model(cfg["model"], n_classes).to(device)
    groups = models.param_groups(model, cfg["model"], lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"],
                                 llrd=cfg["llrd"],
                                 head_lr_mult=cfg["head_lr_mult"])
    opt = torch.optim.AdamW(groups)

    # El presupuesto se fija en PASOS DE OPTIMIZADOR, no en epocas: si cambia el
    # batch de una rama, lo que hay que conservar entre arquitecturas es esto.
    per_epoch = len(tl) // cfg["accum_steps"]
    total = cfg["total_steps"] or per_epoch * cfg["epochs"]
    epochs = math.ceil(total / max(1, per_epoch))
    warmup = int(round(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_schedule(s, total, warmup))

    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    y_va = va.labels
    y_te = te.labels

    print("  clases %d | batch efectivo %d | %d pasos/epoca | %d pasos totales "
          "(%d de warmup) | %d epocas"
          % (n_classes, cfg["batch_size"] * cfg["accum_steps"], per_epoch,
             total, warmup, epochs), flush=True)

    hist, best = [], {"f1": -1.0, "epoch": -1, "state": None}
    step = 0
    for ep in range(epochs):
        model.train()
        te0, run_loss, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tl):
            if step >= total:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                loss = crit(model(x), y) / cfg["accum_steps"]
            scaler.scale(loss).backward()
            run_loss += loss.item() * cfg["accum_steps"]
            nb += 1
            if (i + 1) % cfg["accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

        zv = infer(model, vl, len(va), n_classes, device)
        f1 = macro_f1(zv, y_va, keep)
        acc = float((zv.argmax(1) == y_va).mean())
        hist.append({"epoch": ep, "step": step, "loss": run_loss / max(1, nb),
                     "val_macro_f1": f1, "val_acc": acc,
                     "lr": sched.get_last_lr()[-1], "sec": time.time() - te0})
        flag = ""
        if f1 > best["f1"]:
            # Seleccion por macro-F1 de validacion, no por perdida: con este
            # desbalance la perdida de val mejora mientras las clases raras
            # empeoran.
            best = {"f1": f1, "epoch": ep,
                    "state": copy.deepcopy(model.state_dict())}
            flag = "  <- mejor"
        print("    ep %2d/%d  loss %.4f  val_f1 %.4f  val_acc %.4f  %4.0fs%s"
              % (ep + 1, epochs, hist[-1]["loss"], f1, acc, hist[-1]["sec"], flag),
              flush=True)

    # El test se toca UNA sola vez, al final, con el modelo ya elegido en val.
    model.load_state_dict(best["state"])
    zv = infer(model, vl, len(va), n_classes, device)
    zt = infer(model, el, len(te), n_classes, device)

    return {
        "hist": hist,
        "best_epoch": best["epoch"],
        "val": {"macro_f1": macro_f1(zv, y_va, keep),
                "acc": float((zv.argmax(1) == y_va).mean())},
        "test": {"macro_f1": macro_f1(zt, y_te, keep),
                 "acc": float((zt.argmax(1) == y_te).mean())},
        "classes": tr.classes,
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "steps": {"total": total, "per_epoch": per_epoch, "warmup": warmup},
        "params_m": models.n_params(model) / 1e6,
        "sec": time.time() - t0,
        "_state": best["state"], "_zv": zv, "_zt": zt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="sobrescribe las semillas del config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="sufijo para el directorio de salida")
    ap.add_argument("--force", action="store_true",
                    help="rehace las semillas que ya tengan metrics.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = {**DEFAULTS, **yaml.safe_load(fh)}
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    name = cfg["name"] + (("_" + args.tag) if args.tag else "")

    spec = models.SPECS[cfg["model"]]
    print("[%s] %s (%s, ImageNet top1 %.2f) sobre %s"
          % (name, cfg["model"], spec.weights.name, spec.imagenet_top1, cfg["dataset"]),
          flush=True)

    resumen = []
    for seed in cfg["seeds"]:
        d = os.path.join(RUNS, name, "seed%d" % seed)
        hecho = os.path.join(d, "metrics.json")

        # Reanudacion a nivel de SEMILLA. No hay checkpoint intermedio dentro de
        # una semilla a proposito: guardarlo cada epoca costaria 94 MB por
        # escritura y una semilla de Kaggle son 15 min. Para Fishnet, donde una
        # semilla puede tardar 2,5 h, convendra reevaluarlo.
        if os.path.exists(hecho) and not args.force:
            with open(hecho, encoding="utf-8") as fh:
                m = json.load(fh)
            resumen.append((seed, m["val"]["macro_f1"], m["test"]["macro_f1"],
                            m["test"]["acc"], m["sec"]))
            print("\n  --- semilla %d: ya completada, se omite (--force para rehacerla) ---"
                  % seed, flush=True)
            print("    -> val_f1 %.4f | test_f1 %.4f | test_acc %.4f"
                  % (resumen[-1][1], resumen[-1][2], resumen[-1][3]), flush=True)
            continue

        print("\n  --- semilla %d ---" % seed, flush=True)
        out = run_one(cfg, seed)
        os.makedirs(d, exist_ok=True)

        torch.save(out.pop("_state"), os.path.join(d, "checkpoint.pt"))
        np.save(os.path.join(d, "val_logits.npy"), out.pop("_zv"))
        np.save(os.path.join(d, "test_logits.npy"), out.pop("_zt"))
        with open(os.path.join(d, "config.yaml"), "w", encoding="utf-8") as fh:
            yaml.safe_dump({**cfg, "name": name, "seed": seed}, fh,
                           sort_keys=False, allow_unicode=True)
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({**out, "seed": seed, "config": cfg, "env": {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "commit": git_commit(),
                "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
            }}, fh, indent=2, ensure_ascii=False)

        resumen.append((seed, out["val"]["macro_f1"], out["test"]["macro_f1"],
                        out["test"]["acc"], out["sec"]))
        print("    -> val_f1 %.4f | test_f1 %.4f | test_acc %.4f | %.1f min"
              % (resumen[-1][1], resumen[-1][2], resumen[-1][3], resumen[-1][4] / 60),
              flush=True)

    f1s = np.array([r[2] for r in resumen])
    accs = np.array([r[3] for r in resumen])
    print("\n[%s] %d semillas" % (name, len(resumen)))
    print("  test macro-F1 : %.4f +- %.4f" % (f1s.mean(), f1s.std(ddof=1) if len(f1s) > 1 else 0.0))
    print("  test accuracy : %.4f +- %.4f" % (accs.mean(), accs.std(ddof=1) if len(accs) > 1 else 0.0))
    print("  resultados en runs/%s/" % name)


if __name__ == "__main__":
    main()
