"""
nodo.py — Un nodo detector de intrusiones corriendo como servicio (un contenedor por organización).

Carga un modelo ya entrenado y clasifica los flujos de red que le llegan. No
entrena nada: sirve para mostrar qué detecta una organización según los pesos
que tenga cargados.

Atiende tres pedidos HTTP (puerto 8000):
    GET  /estado    -> nombre del nodo y pesos cargados
    POST /analizar  -> recibe {"flujos": [[...], ...]} y devuelve la familia predicha de cada uno
    POST /modelo    -> recibe {"archivo": "x.pt"} y cambia los pesos por los de ese archivo.
                       Representa el momento en que el servidor federado le envía el modelo global.

Variables de entorno:
    NOMBRE  nombre de la organización (solo para los mensajes)
    PESOS   archivo .pt con el que arranca, dentro de la carpeta /modelos
"""
import json
import os
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from src.config import FAMILIAS
from src.models import MLP, FlowTransformer

DIR_MODELOS = Path(os.environ.get("DIR_MODELOS", "/modelos"))
NOMBRE = os.environ.get("NOMBRE", "Nodo")
PUERTO = 8000
CLASES = {"mlp": MLP, "transformer": FlowTransformer}

estado = {"modelo": None, "archivo": None}  # lo que el nodo tiene cargado en este momento


def cargar_pesos(archivo: str) -> None:
    """Reemplaza el modelo del nodo por el guardado en /modelos/<archivo>."""
    ruta = DIR_MODELOS / Path(archivo).name  # .name: solo se aceptan archivos de esa carpeta
    # weights_only: el archivo solo puede traer números y texto, no código
    guardado = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo = CLASES[guardado["modelo"]](guardado["caracteristicas"])
    modelo.load_state_dict(guardado["pesos"])
    estado["modelo"], estado["archivo"] = modelo.eval(), ruta.name
    print(f"[{NOMBRE}] pesos cargados: {ruta.name}")


@torch.no_grad()
def analizar(flujos: list[list[float]]) -> list[str]:
    """Familia que el modelo le asigna a cada flujo."""
    clases = estado["modelo"](torch.tensor(flujos, dtype=torch.float32)).argmax(dim=1)
    familias = [FAMILIAS[c] for c in clases.tolist()]
    alertas = Counter(f for f in familias if f != "Benign")
    detalle = ", ".join(f"{n} {f}" for f, n in alertas.most_common()) or "ninguna"
    print(f"[{NOMBRE}] {len(flujos)} flujos analizados | alertas: {detalle}")
    return familias


class Pedidos(BaseHTTPRequestHandler):
    def responder(self, datos: dict, codigo: int = 200) -> None:
        cuerpo = json.dumps(datos).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        self.responder({"nombre": NOMBRE, "pesos": estado["archivo"]})

    def do_POST(self):
        pedido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path == "/analizar":
                self.responder({"familias": analizar(pedido["flujos"])})
            elif self.path == "/modelo":
                cargar_pesos(pedido["archivo"])
                self.responder({"nombre": NOMBRE, "pesos": estado["archivo"]})
            else:
                self.responder({"error": "ruta desconocida"}, 404)
        except Exception as error:  # el nodo sigue en pie y avisa qué falló
            self.responder({"error": f"{type(error).__name__}: {error}"}, 500)

    def log_message(self, *_):
        pass  # sin el registro por defecto de cada pedido: ya se imprime un resumen propio


if __name__ == "__main__":
    cargar_pesos(os.environ["PESOS"])
    print(f"[{NOMBRE}] escuchando en el puerto {PUERTO}")
    HTTPServer(("0.0.0.0", PUERTO), Pedidos).serve_forever()
