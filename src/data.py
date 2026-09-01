"""
Datasets y dataloaders de Caleta-Scan (G3).

Lee los manifiestos de `data/splits/` y entrega tensores listos para entrenar.
Dos fuentes, un solo pipeline:

  Kaggle   imagen completa, 9 clases, split por foto original.
  Fishnet  recorte de caja con margen, `label_model` (19 clases + Other_rare),
           split oficial del CSV.

El pipeline de preprocesamiento es IDENTICO para ResNet y para ViT a proposito.
Ver la nota en build_transform() sobre por que no se usa weights.transforms().
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPLITS = os.path.join(ROOT, "data", "splits")

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Margen que se agrega alrededor de la caja antes de recortar. Le da al modelo
# algo del entorno (cubierta, otro pez, mano del operario), que es justamente lo
# que la atencion global del ViT deberia poder explotar.
CROP_MARGIN = 0.10

# ---------------------------------------------------------------------------
# Particion del espacio de etiquetas de Fishnet (protocolo, secciones 2 y 8.1).
# El macro-F1 de especies se calcula sobre label_model MENOS la union de estos
# tres conjuntos. La decision vive aqui, en un solo lugar, para que se pueda
# revisar y cambiar sin buscarla por el codigo.
# ---------------------------------------------------------------------------
NON_FISH_CLASSES = {"Human", "No fish"}      # no son peces
AMBIGUOUS_CLASSES = {"Unknown", "Tuna"}      # son peces, sin especie resuelta
AGGREGATE_CLASSES = {"Other_rare"}           # agregado de la cola larga
NON_SPECIES = NON_FISH_CLASSES | AMBIGUOUS_CLASSES | AGGREGATE_CLASSES

# Atunes especificos que un anotador pudo haber etiquetado como "Tuna". Se usan
# para el credito jerarquico al evaluar: acertar la especie cuando la verdad es
# la superclase no puede contar como error.
TUNA_SPECIES = {"Yellowfin tuna", "Albacore", "Bigeye tuna", "Skipjack tuna"}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform(train: bool):
    """
    Pipeline unico para ambas arquitecturas.

    NO se usa `weights.transforms()` a proposito, aunque sea lo idiomatico:
    ResNet50_Weights.IMAGENET1K_V2 redimensiona a 232 y ViT_B_16_Weights.
    IMAGENET1K_V1 a 256. Dejar que cada modelo traiga su propio preprocesamiento
    haria que vieran imagenes distintas, y la comparacion dejaria de medir solo
    la arquitectura. Se fija el mismo pipeline para los dos.

    En eval se deforma a 224x224 en vez de recortar al centro: los recortes de
    Fishnet van de razon de aspecto 0,32 a 3,89 y un recorte cuadrado cortaria
    colas y morros, que son diagnosticos. Rellenar con letterbox tampoco sirve:
    el grosor del borde negro correlacionaria con la razon de aspecto y por
    tanto con la especie, creando un atajo nuevo.
    """
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            # hue muy bajo: el color separa especies (mahi mahi, red mullet) y
            # rotarlo destruiria senal legitima
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        ] + norm)
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    ] + norm)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FishDataset(Dataset):
    """
    Devuelve (imagen, etiqueta, indice).

    El indice es la posicion en `self.df` y permite a evaluate.py volver a unir
    cada prediccion con su fila para estratificar por oclusion, por secuencia o
    por lo que haga falta, sin reordenar nada.
    """

    def __init__(self, df: pd.DataFrame, label_col: str, classes: list[str],
                 train: bool, crop: bool):
        missing = set(df[label_col]) - set(classes)
        if missing:
            raise ValueError("etiquetas fuera del espacio de clases: %s" % sorted(missing))

        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.crop = crop
        self.transform = build_transform(train)

        # arrays planos: df.iloc[] por item es demasiado lento dentro del loader
        self.paths = self.df["path"].to_numpy()
        self.labels = self.df[label_col].map(self.class_to_idx).to_numpy(dtype=np.int64)
        self.boxes = (self.df[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.int64)
                      if crop else None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(os.path.join(ROOT, self.paths[i])).convert("RGB")
        if self.crop:
            img = img.crop(self._margin_box(i, img.size))
        return self.transform(img), int(self.labels[i]), i

    def _margin_box(self, i, size):
        """Caja expandida por CROP_MARGIN y RECORTADA a los bordes de la imagen.

        El recorte a los bordes no es cosmetico: PIL.crop() rellena con negro lo
        que quede fuera de la imagen, y ese borde negro reintroduciria justo el
        tipo de atajo que el proyecto esta tratando de medir.
        """
        W, H = size
        x0, y0, x1, y1 = self.boxes[i]
        mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
        return (max(0, int(round(x0 - mx))), max(0, int(round(y0 - my))),
                min(W, int(round(x1 + mx))), min(H, int(round(y1 + my))))

    def class_counts(self) -> np.ndarray:
        """Conteo por clase, en el orden de self.classes. Base para el ajuste
        de logits y para el reponderado por numero efectivo."""
        return np.bincount(self.labels, minlength=len(self.classes))

    def species_mask(self) -> np.ndarray:
        """Mascara booleana sobre self.classes: True donde la clase es una
        especie reportable. Fishnet solamente; en Kaggle todas lo son."""
        return np.array([c not in NON_SPECIES for c in self.classes])


# ---------------------------------------------------------------------------
# Constructores
# ---------------------------------------------------------------------------

def purge_sequence_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Saca de val/test toda caja cuya `seq_id` aparezca tambien en train.

    Una secuencia son fotogramas consecutivos del mismo lance, practicamente
    duplicados entre si. El split oficial de Fishnet se armo caja por caja, asi
    que el 44,5% de las cajas de test comparte secuencia con train: evaluar ahi
    mide memorizacion. Purgar deja el test en 7.159 cajas.

    No toca train: la redundancia dentro de train es ineficiencia, no error.
    """
    train_seq = set(df.loc[df["split"] == "train", "seq_id"])
    leaked = df["split"].isin(["val", "test"]) & df["seq_id"].isin(train_seq)
    return df.loc[~leaked].reset_index(drop=True)


def clean_mask(ds_train, ds_part) -> np.ndarray:
    """
    True en las filas de `ds_part` cuya secuencia NO aparece en train.

    Sirve para seleccionar el modelo sobre la parte limpia de validacion sin
    tener que purgar el dataset entero: asi los logits guardados siguen
    cubriendo TODAS las filas y el test purgado se calcula despues como un
    subconjunto, sin volver a inferir.
    """
    if "seq_id" not in ds_part.df.columns:
        return np.ones(len(ds_part), dtype=bool)
    tr_seq = set(ds_train.df["seq_id"])
    return ~ds_part.df["seq_id"].isin(tr_seq).to_numpy()


def _datasets(df, label_col, crop, splits=("train", "val", "test")):
    # El espacio de clases se fija con TRAIN y se comparte: si val o test
    # trajeran una clase inedita, el indice se correria en silencio.
    classes = sorted(df.loc[df["split"] == "train", label_col].unique())
    return {s: FishDataset(df[df["split"] == s], label_col, classes,
                           train=(s == "train"), crop=crop)
            for s in splits}


def kaggle(label_col: str = "clase") -> dict[str, FishDataset]:
    """9 especies, imagen completa, split por foto original."""
    df = pd.read_csv(os.path.join(SPLITS, "kaggle_split.csv"))
    return _datasets(df, label_col, crop=False)


def kaggle_random_split(seed: int = 0) -> dict[str, FishDataset]:
    """
    Mismo dataset, split ALEATORIO por imagen: la version con fuga.

    Existe solo para el experimento E3, que pone las dos accuracies lado a lado
    y cuantifica cuanto infla la fuga. No usar para nada mas.
    """
    df = pd.read_csv(os.path.join(SPLITS, "kaggle_split.csv"))
    rng = np.random.default_rng(seed)
    n = len(df)
    frac = df["split"].value_counts(normalize=True)
    shuffled = rng.permutation(n)
    n_tr = int(round(frac.get("train", 0.7) * n))
    n_va = int(round(frac.get("val", 0.15) * n))
    lab = np.empty(n, dtype=object)
    lab[shuffled[:n_tr]] = "train"
    lab[shuffled[n_tr:n_tr + n_va]] = "val"
    lab[shuffled[n_tr + n_va:]] = "test"
    df = df.assign(split=lab)
    return _datasets(df, "clase", crop=False)


def fishnet(label_col: str = "label_model", purge: bool = False) -> dict[str, FishDataset]:
    """
    Recortes de caja. `purge=True` aplica purge_sequence_leak() a val/test.

    Se entrena una sola vez y se evalua sobre las dos versiones del test: la
    oficial (comparable con la literatura) y la purgada (honesta).
    """
    df = pd.read_csv(os.path.join(SPLITS, "fishnet_crops.csv"))
    if purge:
        df = purge_sequence_leak(df)
    return _datasets(df, label_col, crop=True)


# ---------------------------------------------------------------------------
# Loaders y semillas
# ---------------------------------------------------------------------------

def _worker_init(worker_id):
    # torch.initial_seed() dentro de un worker ya devuelve base_seed+worker_id,
    # asi que cada worker recibe una semilla distinta. Sin esto, numpy y random
    # arrancan identicos en todos los workers y repiten la misma aumentacion.
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(ds: FishDataset, batch_size: int, train: bool,
                num_workers: int = 8, seed: int = 0,
                persistent: bool | None = None) -> DataLoader:
    """
    OJO EN WINDOWS: con num_workers > 0 los workers arrancan por `spawn`, que
    REIMPORTA el modulo principal en cada proceso hijo. Todo script que llame a
    make_loader() tiene que proteger su punto de entrada:

        if __name__ == "__main__":
            main()

    Sin ese guard, cada worker vuelve a ejecutar el script entero y se generan
    procesos sin parar hasta colgar la maquina. Por la misma razon un script con
    workers no se puede pasar por stdin (`python - < script.py`): el hijo
    intenta reejecutar '<stdin>' y falla con OSError [Errno 22].

    `persistent` solo por defecto en el loader de TRAIN. Mantener vivos tambien
    los de val y test triplicaba el numero de workers por semilla (8 x 3 = 24) y
    los dejaba ocupando memoria hasta el final del proceso: con tres semillas
    eso agoto el limite de commit de Windows y reventó con
    "Couldn't open shared file mapping ... error code 1455". Los de evaluacion
    se recrean en cada epoca; el arranque se amortiza sobre una epoca entera.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        pin_memory=True, drop_last=train,
        persistent_workers=(train if persistent is None else persistent) and num_workers > 0,
        worker_init_fn=_worker_init, generator=g,
    )


def shutdown_loaders(*loaders) -> None:
    """
    Cierra los workers de forma explicita.

    Salir de la funcion no basta: con persistent_workers el DataLoader mantiene
    vivo su iterador, y el recolector de basura de CPython no corre a tiempo. En
    Windows cada worker huerfano retiene ~340 MB, asi que tres semillas dejaban
    ~6 GB sin liberar.
    """
    import gc
    for ld in loaders:
        it = getattr(ld, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
            ld._iterator = None
    gc.collect()


def set_seed(seed: int):
    """
    Siembra las tres fuentes de aleatoriedad.

    No se activa torch.use_deterministic_algorithms(True) a proposito: exige
    CUBLAS_WORKSPACE_CONFIG=:4096:8, cuesta velocidad y algunas operaciones no
    tienen kernel determinista y lanzan excepcion. La alternativa correcta, y
    mas informativa, es correr 3 semillas y reportar media +- desviacion.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    for name, dss in [("kaggle", kaggle()),
                      ("fishnet", fishnet()),
                      ("fishnet purgado", fishnet(purge=True))]:
        d = dss["train"]
        print("%-16s %d clases | train %d val %d test %d"
              % (name, len(d.classes), len(dss["train"]), len(dss["val"]), len(dss["test"])))
