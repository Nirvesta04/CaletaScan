"""
Demo de Caleta-Scan: la comparacion, en vivo.

    python -m src.demo            # http://127.0.0.1:8000

Sin dependencias nuevas: servidor de la biblioteca estandar y una pagina
propia. Gradio o Streamlit arrastran un arbol de dependencias grande para algo
que aqui son tres endpoints, y el requirements.txt del proyecto esta fijado a
proposito.

Que hace, y por que asi:

- Corre los CINCO modelos comparados sobre la misma caja y los muestra uno bajo
  otro, de mejor a peor macro-F1, con su latencia. La demo no es "un clasificador": es el experimento del proyecto.
- Cada modelo trae su propio ajuste de logits (tau elegido en validacion: de
  0,03 el ConvNeXt a 0,18 el ResNet) porque esa es la regla de decision que se reporto.
- Cada modelo puede decir NO SE. El umbral es la energia que deja pasar el 95 % de
  los recortes de val limpios que no son personas (elegido en validacion). Sobre
  peces de otro dominio el ViT rechaza bastante mas que el ResNet (27 % contra
  84 % aceptados en Kaggle), pero ninguno rechaza todo: un caso suelto puede
  pasar en los dos.
- No hay detector (§2). El usuario marca la caja arrastrando, o elige una de
  las cajas anotadas de una foto de ejemplo. Prometer deteccion seria vender lo
  que no se midio.
- La confianza que muestra NO es una probabilidad calibrada (§10.3). La pagina
  lo dice; callarlo seria el error mas facil de cometer en una demo.

Sobre los milisegundos que muestra: son los de ESA llamada, con la GPU
despertando entre peticiones separadas por el trabajo de la interfaz. Salen 3
a 5 veces por encima del benchmark de §11, donde las llamadas van seguidas y
la GPU se mantiene caliente. Sirven para comparar los modelos entre
si en el mismo momento, no como medicion de latencia: esa es la de §11.
"""

from __future__ import annotations

import base64
import io
import json
import os
import socketserver
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from . import calibrate as C
from . import data, models
from . import evaluate as E

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = data.ROOT
PUERTO = 8000
N_MUESTRAS = 12
TPR = 0.95          # el umbral de rechazo conserva el 95 % del test limpio

MODELOS = (("Swin-T", "swin_t", "e12_swin_t_fishnet"),   # de mejor a peor macro-F1 en la prueba limpia
           ("ConvNeXt-T", "convnext_t", "e11_convnext_t_fishnet"),
           ("ViT-B/16", "vit_b_16", "e5_vit_b_16_fishnet"),
           ("DeiT-S", "deit_s", "e10_deit_s_fishnet"),
           ("ResNet-50", "resnet50", "e4_resnet50_fishnet"))


class Motor:
    """Modelos cargados, umbrales de rechazo y el pipeline de una prediccion."""

    def __init__(self, device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ds = data.fishnet()
        self.clases = self.ds["train"].classes
        self.tf = data.build_transform(train=False)
        self.modelos, self.tau, self.umbral = {}, {}, {}

        for etiqueta, arch, run in MODELOS:
            run_obj = E.Run(run)
            m = models.build_model(arch, len(self.clases))
            ck = os.path.join(E.RUNS, run, "seed0", "checkpoint.pt")
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
            self.modelos[etiqueta] = m.to(self.dev).eval()
            self.tau[etiqueta] = C._tau(run_obj, limpio=True)
            # umbral de energia elegido en VALIDACION: deja pasar el 95 % de los
            # recortes de val limpios que no son personas. Elegirlo en test
            # usaria el test para decidir algo, y el test no se toca.
            iv = run_obj.rows("val", True)
            iv = iv[run_obj.y("val")[iv] != self.clases.index("Human")]
            s = C.score_energy(run_obj.z[0]["val"][iv])
            k = int(np.ceil(TPR * len(s)))
            self.umbral[etiqueta] = float(np.sort(s)[len(s) - k])

        self.prior = self.ds["train"].class_counts() / len(self.ds["train"])
        self.muestras = self._muestras()
        self._calentar()

    @torch.inference_mode()
    def _calentar(self, n: int = 3):
        """Sin esto la primera prediccion tarda ~1 s y parece que el modelo es lento."""
        x = torch.zeros(1, 3, data.IMG_SIZE, data.IMG_SIZE, device=self.dev)
        for m in self.modelos.values():
            for _ in range(n):
                m(x)
        if self.dev == "cuda":
            torch.cuda.synchronize()

    def _muestras(self) -> list[dict]:
        """Fotos del test limpio, con sus cajas anotadas y variedad de especies."""
        df = self.ds["test"].df
        limpio = data.clean_mask(self.ds["train"], self.ds["test"])
        d = df[limpio & (df["is_fish"] == 1)]
        elegidas, vistas = [], set()
        for _, fila in d.iterrows():                      # una foto por especie
            if fila["label_model"] in vistas:
                continue
            vistas.add(fila["label_model"])
            cajas = df[df["path"] == fila["path"]]
            elegidas.append({
                "path": fila["path"],
                "etiqueta": fila["label_model"],
                "cajas": [{"x0": int(c.x0), "y0": int(c.y0), "x1": int(c.x1),
                           "y1": int(c.y1), "gt": c.label_model}
                          for c in cajas.itertuples()],
            })
            if len(elegidas) >= N_MUESTRAS:
                break
        return elegidas

    def recortar(self, img: Image.Image, caja: dict | None) -> Image.Image:
        """Mismo margen del 10 % y mismo recorte a los bordes que en entrenamiento."""
        if not caja:
            return img
        W, H = img.size
        x0, y0, x1, y1 = (max(0, min(W, int(caja["x0"]))), max(0, min(H, int(caja["y0"]))),
                          max(0, min(W, int(caja["x1"]))), max(0, min(H, int(caja["y1"]))))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return img
        mx, my = (x1 - x0) * data.CROP_MARGIN, (y1 - y0) * data.CROP_MARGIN
        return img.crop((max(0, int(x0 - mx)), max(0, int(y0 - my)),
                         min(W, int(x1 + mx)), min(H, int(y1 + my))))

    @torch.inference_mode()
    def predecir(self, img: Image.Image, caja: dict | None) -> dict:
        x = self.tf(self.recortar(img, caja)).unsqueeze(0).to(self.dev)
        salida = []
        for etiqueta in self.modelos:
            m = self.modelos[etiqueta]
            if self.dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = m(x).float().cpu().numpy()
            if self.dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3

            energia = float(C.score_energy(z)[0])
            p = C.softmax(E.logit_adjust(z, self.prior, self.tau[etiqueta]).astype(np.float64))[0]
            orden = np.argsort(p)[::-1][:3]
            salida.append({
                "modelo": etiqueta, "ms": ms,
                "energia": energia, "umbral": self.umbral[etiqueta],
                "acepta": bool(energia >= self.umbral[etiqueta]),
                "tau": self.tau[etiqueta],
                "top": [{"clase": self.clases[i], "p": float(p[i])} for i in orden],
            })
        return {"pred": salida, "dispositivo": self.dev}


def _leer_imagen(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


class Handler(BaseHTTPRequestHandler):
    motor: Motor = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):            # sin ruido en consola
        pass

    def _enviar(self, code, tipo, cuerpo: bytes):
        self.send_response(code)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, obj, code=200):
        self._enviar(code, "application/json; charset=utf-8",
                     json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        ruta = self.path.split("?")[0]
        if ruta in ("/", "/index.html"):
            with open(os.path.join(HERE, "demo.html"), "rb") as fh:
                return self._enviar(200, "text/html; charset=utf-8", fh.read())
        if ruta == "/api/muestras":
            return self._json({"muestras": self.motor.muestras,
                               "clases": self.motor.clases,
                               "dispositivo": self.motor.dev,
                               "umbrales": self.motor.umbral,
                               "tau": self.motor.tau})
        if ruta == "/api/imagen":
            q = dict(p.split("=", 1) for p in self.path.split("?")[1].split("&"))
            i = int(q.get("i", 0))
            if not 0 <= i < len(self.motor.muestras):
                return self._json({"error": "indice fuera de rango"}, 404)
            # la ruta viene del CSV del proyecto, no del cliente: no hay travesia
            img = Image.open(os.path.join(ROOT, self.motor.muestras[i]["path"])).convert("RGB")
            b = io.BytesIO()
            img.save(b, "JPEG", quality=88)
            return self._enviar(200, "image/jpeg", b.getvalue())
        return self._json({"error": "no existe"}, 404)

    def do_POST(self):
        if self.path != "/api/predecir":
            return self._json({"error": "no existe"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        if n > 20 * 2 ** 20:
            return self._json({"error": "imagen demasiado grande"}, 413)
        try:
            cuerpo = json.loads(self.rfile.read(n))
            if "muestra" in cuerpo:
                p = self.motor.muestras[int(cuerpo["muestra"])]["path"]
                img = Image.open(os.path.join(ROOT, p)).convert("RGB")
            else:
                img = _leer_imagen(base64.b64decode(cuerpo["imagen"].split(",")[-1]))
            r = self.motor.predecir(img, cuerpo.get("caja"))
            r["tam"] = list(img.size)
            return self._json(r)
        except Exception as e:                       # la demo no se cae con una imagen rara
            return self._json({"error": "%s: %s" % (type(e).__name__, e)}, 400)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=PUERTO)
    ap.add_argument("--cpu", action="store_true", help="forzar CPU (escenario caleta sin GPU)")
    args = ap.parse_args()

    print("cargando modelos...", flush=True)
    Handler.motor = Motor("cpu" if args.cpu else None)
    m = Handler.motor
    print("listo en %s · tau %s · umbral de energia %s"
          % (m.dev, {k: round(v, 2) for k, v in m.tau.items()},
             {k: round(v, 1) for k, v in m.umbral.items()}))
    print("http://127.0.0.1:%d   (ctrl+c para parar)" % args.puerto)

    socketserver.TCPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", args.puerto), Handler) as s:
        try:
            s.serve_forever()
        except KeyboardInterrupt:
            print("\nadios")


if __name__ == "__main__":
    main()
