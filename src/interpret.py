"""
Interpretabilidad cuantitativa: Grad-CAM, attention rollout y relevancia de Chefer.

    python -m src.interpret --resnet e1_resnet50_kaggle --vit e2_vit_b_16_kaggle

Convierte los mapas de saliencia en un NUMERO usando las mascaras GT de Kaggle:
el "pointing game" mide que fraccion de la energia del mapa cae dentro del pez.

    E_dentro = sum_{p en M} CAM(p) / sum_p CAM(p)

Linea base medida: la mascara GT cubre en promedio el 16,1% de los pixeles, asi
que un mapa uniforme deposita 0,161 de su energia dentro del pez. Todo lo que
exceda esa cifra es senal; un modelo con E_dentro cercano a 0,16 esta mirando el
fondo aunque acierte la clase.

SOBRE LA COMPARACION ENTRE ARQUITECTURAS
----------------------------------------
Grad-CAM es especifico de clase: usa los gradientes del logit objetivo. El
attention rollout NO lo es: no involucra ningun gradiente y da el mismo mapa
para todas las clases. Ponerlos lado a lado y concluir algo sobre que modelo
"mira mejor" compara una explicacion condicionada a la clase contra una que no
lo esta.

Por eso se reportan los dos metodos para el ViT:
  rollout  la herramienta estandar, agnostica de clase (Abnar y Zuidema 2020)
  chefer   rollout ponderado por gradientes, SI especifico de clase, y por
           tanto la contraparte honesta de Grad-CAM (Chefer et al. 2021)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T

from . import data, models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


# ---------------------------------------------------------------------------
# Mascaras GT
# ---------------------------------------------------------------------------

def gt_path(path: str) -> str:
    """`.../Trout/Trout/00001.png` -> `.../Trout/Trout GT/00001.png`"""
    q = path.split("/")
    cls = q[-3]
    return "/".join(q[:-3] + [cls, cls + " GT", q[-1]])


def load_mask(path: str, size: int = data.IMG_SIZE) -> np.ndarray:
    """Mascara binaria a la MISMA resolucion que ve el modelo.

    NEAREST, no BICUBIC: interpolar una mascara binaria crea valores
    intermedios y el borde del pez dejaria de estar definido.
    """
    with Image.open(gt_path(path)) as im:
        m = im.convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(m) > 127


# ---------------------------------------------------------------------------
# Grad-CAM (ResNet)
# ---------------------------------------------------------------------------

def gradcam(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    Selvaraju et al. (2017) sobre la salida de layer4[-1] (7x7x2048 a 224 px).

        alpha_k = (1/Z) sum_ij  d y_c / d A_ij_k        gradientes promediados
        L_c     = ReLU( sum_k alpha_k * A_k )

    Dos detalles que se suelen equivocar:
      - la capa es la SALIDA de layer4, no despues del avgpool: ahi ya no queda
        informacion espacial y el mapa sale constante.
      - y_c es el LOGIT, no la probabilidad. Derivar la softmax mezcla los
        gradientes de las demas clases y el mapa deja de ser especifico de c.
    """
    acts, grads = [], []
    h1 = model.layer4.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        sel = logits.gather(1, target[:, None]).sum()   # el logit, no la softmax
        sel.backward()
    finally:
        h1.remove(); h2.remove()

    A, G = acts[0], grads[0]                       # (B,2048,7,7)
    alpha = G.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((alpha * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]),
                        mode="bilinear", align_corners=False)
    return cam[:, 0].detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# ViT: capturar la atencion
# ---------------------------------------------------------------------------

def _attention(mha: nn.MultiheadAttention, x: torch.Tensor):
    """
    Atencion multi-cabeza reimplementada, para poder quedarse con la matriz A.

    Por que no basta con need_weights=True: los pesos que devuelve
    nn.MultiheadAttention son un VIEW construido DESPUES de calcular la salida.
    El gradiente del output fluye hasta el tensor interno, no hasta ese view, que
    cuelga como rama muerta del grafo. retain_grad() sobre el no recibe nada y
    .grad queda en None. Chefer necesita justamente ese gradiente, asi que hay
    que tener el tensor en el camino real de computo.

    Usa los mismos pesos del modulo, asi que es el MISMO modelo; la equivalencia
    numerica con el forward original se verifica en verificar_atencion().
    """
    B, N, E = x.shape
    H = mha.num_heads
    d = E // H
    qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, N, H, d).transpose(1, 2)
    k = k.view(B, N, H, d).transpose(1, 2)
    v = v.view(B, N, H, d).transpose(1, 2)
    A = (q @ k.transpose(-2, -1)) / (d ** 0.5)
    A = A.softmax(-1)                       # (B, H, N, N)
    if A.requires_grad:                     # no bajo torch.no_grad()
        A.retain_grad()
    out = (A @ v).transpose(1, 2).reshape(B, N, E)
    return mha.out_proj(out), A


class _AttnTap:
    """
    Reemplaza el forward de cada EncoderBlock para capturar la atencion.

    Efecto colateral: se desactiva la ruta rapida fusionada y el forward se
    vuelve mas lento. Irrelevante aqui, pero por eso la latencia NUNCA se mide
    con este parche puesto.
    """

    def __init__(self, model):
        self.model = model
        self.attn: list[torch.Tensor] = []
        self._orig = []

    def __enter__(self):
        for blk in self.model.encoder.layers:
            self._orig.append(blk.forward)
            blk.forward = self._make(blk)
        return self

    def _make(self, blk):
        def fwd(inp):
            x = blk.ln_1(inp)
            x, A = _attention(blk.self_attention, x)
            self.attn.append(A)
            x = blk.dropout(x)
            x = x + inp
            y = blk.ln_2(x)
            y = blk.mlp(y)
            return x + y
        return fwd

    def __exit__(self, *a):
        for blk, f in zip(self.model.encoder.layers, self._orig):
            blk.forward = f
        return False


def verificar_atencion(model, x, tol=2e-4) -> dict:
    """
    Comprueba que el forward parcheado reproduce el original.

    Si no coincidiera, los mapas explicarian un modelo distinto del que se
    evaluo, que es una forma silenciosa de estar completamente equivocado.
    """
    model.eval()
    with torch.no_grad():
        z0 = model(x)
    with _AttnTap(model):
        with torch.no_grad():
            z1 = model(x)
    dif = (z0 - z1).abs().max().item()
    return {"max_dif_logits": dif, "ok": dif < tol,
            "argmax_igual": bool((z0.argmax(1) == z1.argmax(1)).all())}


def _rollout(mats: list[torch.Tensor], size: int) -> np.ndarray:
    """
    Producto de matrices por capa, con la identidad que representa el residual.

        A_l = mean_heads(A_l);  A_l = A_l + I;  normalizar filas
        R   = A_L @ ... @ A_1
        mapa = R[0, 1:]   (fila del CLS, sin su propia columna)

    Sumar I no es cosmetico: cada bloque tiene una conexion residual que deja
    pasar informacion sin atender, y la atencion cruda de la ultima capa ignora
    eso por completo.
    """
    R = None
    for a in mats:
        a = a + torch.eye(a.shape[-1], device=a.device)[None]
        a = a / a.sum(-1, keepdim=True)
        R = a if R is None else torch.bmm(a, R)
    n = int((R.shape[-1] - 1) ** 0.5)
    m = R[:, 0, 1:].reshape(-1, 1, n, n)
    m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
    return m[:, 0].detach().float().cpu().numpy()


def vit_maps(model: nn.Module, x: torch.Tensor, target: torch.Tensor
             ) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (rollout, chefer). El primero es agnostico de clase; el segundo
    pondera por gradientes y es el comparable con Grad-CAM."""
    size = x.shape[-1]
    with _AttnTap(model) as tap:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits.gather(1, target[:, None]).sum().backward()
        att = tap.attn

    if any(a.grad is None for a in att):
        raise RuntimeError("la atencion no recibio gradiente; ver _attention()")
    roll = _rollout([a.mean(1) for a in att], size)
    # Chefer: E_h[(grad * attn)^+], la parte positiva antes de promediar cabezas
    che = _rollout([(a.grad * a).clamp(min=0).mean(1) for a in att], size)
    return roll, che


# ---------------------------------------------------------------------------
# Pointing game
# ---------------------------------------------------------------------------

def energy_inside(cam: np.ndarray, mask: np.ndarray) -> float:
    """Fraccion de la energia del mapa que cae dentro de la mascara."""
    c = np.maximum(cam, 0)
    tot = c.sum()
    return float(c[mask].sum() / tot) if tot > 0 else float("nan")


# ---------------------------------------------------------------------------

def load_ckpt(run: str, seed: int, model_name: str, n_classes: int, device: str):
    p = os.path.join(RUNS, run, "seed%d" % seed, "checkpoint.pt")
    m = models.build_model(model_name, n_classes)
    m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False))
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", default="e1_resnet50_kaggle")
    ap.add_argument("--vit", default="e2_vit_b_16_kaggle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0 = todo el test")
    ap.add_argument("--json", default="runs/pointing_game.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = data.kaggle()["test"]
    n = len(ds) if args.n <= 0 else min(args.n, len(ds))
    rows = np.linspace(0, len(ds) - 1, n).astype(int)   # muestra uniforme
    tf = data.build_transform(train=False)

    rn = load_ckpt(args.resnet, args.seed, "resnet50", len(ds.classes), dev)
    vt = load_ckpt(args.vit, args.seed, "vit_b_16", len(ds.classes), dev)

    # el forward parcheado tiene que dar el MISMO modelo, o los mapas explican
    # algo que no es lo que se evaluo
    x0 = torch.stack([tf(Image.open(os.path.join(ROOT, ds.paths[j])).convert("RGB"))
                      for j in rows[:8]]).to(dev)
    v = verificar_atencion(vt, x0)
    print("verificacion del forward parcheado del ViT: max|dif logits| = %.2e, "
          "argmax igual = %s -> %s" % (v["max_dif_logits"], v["argmax_igual"],
                                       "OK" if v["ok"] else "FALLA"))
    if not v["ok"]:
        raise SystemExit("el forward parcheado no reproduce el original")

    base, out = [], {k: [] for k in ("gradcam", "rollout", "chefer")}
    bs = 8
    for i0 in range(0, n, bs):
        idx = rows[i0:i0 + bs]
        paths = [os.path.join(ROOT, ds.paths[j]) for j in idx]
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(dev)
        masks = [load_mask(ds.paths[j]) for j in idx]
        base += [float(m.mean()) for m in masks]

        with torch.no_grad():
            tgt_r = rn(xs).argmax(1)
            tgt_v = vt(xs).argmax(1)

        cams = gradcam(rn, xs, tgt_r)
        roll, che = vit_maps(vt, xs, tgt_v)
        for k, arr in (("gradcam", cams), ("rollout", roll), ("chefer", che)):
            out[k] += [energy_inside(a, m) for a, m in zip(arr, masks)]
        if (i0 // bs) % 20 == 0:
            print("  %d/%d" % (i0, n), flush=True)

    b = float(np.mean(base))
    print("\n" + "=" * 66)
    print("POINTING GAME  ·  %d imagenes del test de Kaggle" % n)
    print("=" * 66)
    print("  linea base (mapa uniforme = area de la mascara): %.3f" % b)
    print()
    print("  %-28s %8s %8s %10s" % ("metodo", "E_dentro", "std", "x linea base"))
    print("  " + "-" * 58)
    etiq = {"gradcam": "ResNet · Grad-CAM",
            "rollout": "ViT · attention rollout",
            "chefer": "ViT · relevancia de Chefer"}
    res = {}
    for k in ("gradcam", "rollout", "chefer"):
        v = np.array(out[k]); v = v[~np.isnan(v)]
        res[k] = {"media": float(v.mean()), "std": float(v.std()),
                  "mediana": float(np.median(v)), "n": int(len(v))}
        print("  %-28s %8.3f %8.3f %9.2fx" % (etiq[k], v.mean(), v.std(), v.mean() / b))
    print("\n  (rollout es agnostico de clase; el comparable con Grad-CAM es Chefer)")

    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump({"n": n, "linea_base": b, "resultados": res,
                   "resnet": args.resnet, "vit": args.vit, "seed": args.seed},
                  fh, indent=2, ensure_ascii=False)
    print("  guardado en %s" % args.json)


if __name__ == "__main__":
    main()
