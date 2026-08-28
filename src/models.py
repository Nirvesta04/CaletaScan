"""
Fabrica de modelos de Caleta-Scan (G3).

Un solo lugar donde se decide que pesos preentrenados usa cada arquitectura,
como se reemplaza la cabeza y como se arman los grupos de parametros del
optimizador.

Regla que no se rompe: los pesos van FIJADOS EXPLICITAMENTE, nunca `DEFAULT`.
`ResNet50_Weights.DEFAULT` apunta a IMAGENET1K_V1 (76,13% en ImageNet) mientras
ViT_B_16_Weights.IMAGENET1K_V1 esta en 81,07%. Comparar esos dos y concluir que
la atencion global gana mide 4,7 puntos de receta de preentrenamiento, no de
arquitectura. El par correcto es ResNet-50 V2 (80,86) contra ViT-B/16 (81,07):
0,21 puntos de diferencia de partida.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision.models import (
    ResNet50_Weights, ViT_B_16_Weights, ViT_B_32_Weights,
    resnet50, vit_b_16, vit_b_32,
)


@dataclass(frozen=True)
class Spec:
    ctor: Callable
    weights: object
    family: str          # "resnet" | "vit"
    imagenet_top1: float
    params_m: float
    macs_g: float
    note: str


SPECS: dict[str, Spec] = {
    # --- par primario: preentrenamiento emparejado ---------------------------
    "resnet50": Spec(
        resnet50, ResNet50_Weights.IMAGENET1K_V2, "resnet",
        80.86, 25.56, 4.09,
        "rama ResNet del par primario; receta moderna"),
    "vit_b_16": Spec(
        vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1, "vit",
        81.07, 86.57, 16.85,
        "rama ViT del par primario"),

    # --- puntos secundarios --------------------------------------------------
    "vit_b_32": Spec(
        vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1, "vit",
        75.91, 88.22, 4.36,
        "iso-FLOPs con resnet50 (4,36 vs 4,09 G); preentrenamiento 5 pts peor"),
    "resnet50_v1": Spec(
        resnet50, ResNet50_Weights.IMAGENET1K_V1, "resnet",
        76.13, 25.56, 4.09,
        "ablacion E8: misma arquitectura que resnet50, receta antigua. "
        "La brecha contra resnet50 es enteramente de preentrenamiento"),
}

# Parametros del ViT que la regla habitual "ndim <= 1" NO excluye del weight
# decay, porque tienen ndim=3. Aplicarles decay encoge el embedding de posicion
# hacia cero y borra la nocion de posicion del modelo.
VIT_NO_DECAY_NAMES = {"class_token", "encoder.pos_embedding"}


def build_model(name: str, num_classes: int) -> nn.Module:
    """Modelo preentrenado con la cabeza reemplazada por una de num_classes."""
    if name not in SPECS:
        raise KeyError("modelo desconocido: %s (hay: %s)" % (name, ", ".join(SPECS)))
    spec = SPECS[name]
    model = spec.ctor(weights=spec.weights)

    if spec.family == "resnet":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Grupos de parametros
# ---------------------------------------------------------------------------

def _no_decay(pname: str, p: torch.Tensor) -> bool:
    return p.ndim <= 1 or pname in VIT_NO_DECAY_NAMES


def _vit_depth(pname: str, n_blocks: int = 12) -> int:
    """
    Profundidad de un parametro del ViT, de 0 (entrada) a n_blocks+1 (cabeza).

    0            conv_proj, class_token, pos_embedding
    1..n_blocks  encoder_layer_0 .. encoder_layer_{n-1}
    n_blocks+1   encoder.ln, heads.head
    """
    if pname == "class_token" or pname == "encoder.pos_embedding" \
            or pname.startswith("conv_proj"):
        return 0
    if pname.startswith("encoder.layers.encoder_layer_"):
        return int(pname.split("encoder_layer_")[1].split(".")[0]) + 1
    return n_blocks + 1


def param_groups(model: nn.Module, name: str, lr: float, weight_decay: float,
                 llrd: float = 0.70, head_lr_mult: float = 10.0) -> list[dict]:
    """
    Grupos de parametros con la receta propia de cada familia.

    El protocolo empareja el presupuesto, no la receta: forzar SGD sobre un ViT
    o AdamW plano sobre un ResNet no es "justo", es sabotear una de las ramas.

    ResNet  dos niveles. El backbone aprende a `lr`, la cabeza a
            `lr * head_lr_mult` porque parte de cero.
    ViT     decaimiento por capa: la capa de profundidad d usa
            lr * llrd^(n_blocks+1-d). Con llrd=0.70 y 12 bloques, la cabeza
            recibe lr y el embedding de parches lr*0.70^13 = lr*0.0097.

    En ambos casos se excluyen del weight decay los bias, los parametros de
    normalizacion y -- en el ViT -- class_token y pos_embedding.
    """
    family = SPECS[name].family
    groups: dict[tuple, dict] = {}

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if _no_decay(pname, p) else weight_decay

        if family == "vit":
            d = _vit_depth(pname)
            scale = llrd ** (12 + 1 - d)
            key = (d, wd)
        else:
            is_head = pname.startswith("fc.")
            scale = head_lr_mult if is_head else 1.0
            key = (int(is_head), wd)

        g = groups.setdefault(key, {"params": [], "lr": lr * scale,
                                    "weight_decay": wd, "names": []})
        g["params"].append(p)
        g["names"].append(pname)

    return [groups[k] for k in sorted(groups)]


def freeze_bn(model: nn.Module) -> None:
    """
    Pone los BatchNorm en modo eval.

    Llamar DESPUES de cada model.train() si el backbone esta congelado.
    `requires_grad=False` no congela las estadisticas moviles de BN: siguen
    actualizandose en cada forward mientras el modulo este en modo train, y el
    modelo se degrada sin que ningun gradiente lo explique.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("%-13s %-9s %7s %7s  %s" % ("modelo", "familia", "top1", "params", "pesos"))
    for name, s in SPECS.items():
        print("%-13s %-9s %7.2f %6.1fM  %s" % (name, s.family, s.imagenet_top1,
                                               s.params_m, s.weights.name))
        print("%15s%s" % ("", s.note))
