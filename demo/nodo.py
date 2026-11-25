"""
nodo.py — Un agente detector de intrusiones (un contenedor por organización).

Cada agente repite el mismo ciclo con el tráfico que le llega:
    1. Percibe:   recibe flujos de red y los clasifica con su modelo.
    2. Decide:    permitir (tráfico normal), bloquear (ataque con mucha confianza)
                  o solo alertar (ataque con poca confianza).
    3. Coopera:   si reconoció un ataque y tiene un modelo para ofrecer, avisa a sus
                  pares y se lo ofrece. Nadie le ordena hacerlo: no hay servidor.

Cuando a un agente le ofrecen un modelo, decide solo si lo acepta:
    - Lo prueba con sus propios flujos de validación. Lo acepta si no acierta
      menos ni da más falsas alarmas que el modelo que ya tiene.
    - Lleva una reputación de cada par: sube cuando una oferta resulta buena y
      baja a la mitad cuando resulta mala. Por debajo de un mínimo, deja de escucharlo.

El agente no entrena: los modelos que se ofrecen se entrenaron antes (experimento E3)
y están en la carpeta compartida /modelos. Un mensaje lleva el nombre del archivo.

Pedidos HTTP (puerto 8000):
    GET  /estado     -> modelo cargado, reputación de sus pares y mensajes recibidos
    POST /analizar   -> {"flujos": [[...], ...]}  => familia y acción para cada flujo
    POST /mensaje    -> oferta de un par          => {"decision": ..., "motivo": ...}
    POST /reiniciar  -> {"comunicar": true|false, "familia": "Web"}: vuelve a su modelo propio y
                        olvida lo demás. Si se indica otra familia, pasa a usar los archivos de esa
                        (lo usa el panel para preparar cada fase de la demo y para cambiar de ataque)

Variables de entorno:
    NOMBRE      nombre de la organización
    DIRECCION   nombre de este contenedor en la red (cómo lo llaman sus pares)
    PARES       direcciones de los otros agentes, separadas por comas
    FAMILIA     familia de ataque con la que arranca (el "ataque nuevo" de la demo)
    PESOS       archivo .pt con su modelo propio
    VALIDACION  archivo .csv con sus flujos de validación
    OFRECE      archivo .pt que ofrece a sus pares cuando reconoce un ataque (opcional)
  En esos tres nombres, el texto <familia> se reemplaza por la familia elegida.
"""
import csv
import json
import os
import threading
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from src.config import COLUMNA_ETIQUETA, FAMILIAS
from src.models import MLP, FlowTransformer

DIR_MODELOS = Path(os.environ.get("DIR_MODELOS", "/modelos"))
NOMBRE = os.environ.get("NOMBRE", "Nodo")
DIRECCION = os.environ.get("DIRECCION", "nodo")
PARES = [par for par in os.environ.get("PARES", "").split(",") if par]
PUERTO = 8000
CLASES = {"mlp": MLP, "transformer": FlowTransformer}

CONFIANZA_PARA_BLOQUEAR = 0.9  # probabilidad mínima de la familia predicha para bloquear el flujo
MINIMO_PARA_AVISAR = 5         # flujos de un mismo ataque en un lote para avisar a los pares
TOLERANCIA = 0.02              # cuánto peor que el modelo actual puede ser uno ofrecido y aceptarse igual
REPUTACION_INICIAL = 0.5
REPUTACION_MINIMA = 0.2        # por debajo de esto, las ofertas de ese par se ignoran sin probarlas
PREMIO = 0.25                  # lo que sube la reputación con una oferta buena (una mala la baja a la mitad)

# Todo lo que el agente sabe en este momento. Se llena en reiniciar().
estado = {"modelo": None, "archivo": None, "comunicar": False, "medidas": {},
          "reputacion": {}, "pendientes": set(), "mensajes": []}
# Archivos del agente para la familia de ataque elegida. Se llena en elegir_familia().
propio = {"familia": None, "pesos": None, "ofrece": None}
validacion = {"X": None, "y": None}
candado = threading.Lock()  # el servidor atiende varios pedidos a la vez: el modelo se cambia de a uno


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def leer_modelo(archivo: str) -> torch.nn.Module:
    ruta = DIR_MODELOS / Path(archivo).name  # .name: solo se aceptan archivos de esa carpeta
    # weights_only: el archivo solo puede traer números y texto, no código
    guardado = torch.load(ruta, map_location="cpu", weights_only=True)
    modelo = CLASES[guardado["modelo"]](guardado["caracteristicas"])
    modelo.load_state_dict(guardado["pesos"])
    return modelo.eval()


@torch.no_grad()
def medir(modelo: torch.nn.Module) -> dict:
    """Cómo le va a un modelo con los flujos de validación de este agente."""
    prediccion = modelo(validacion["X"]).argmax(dim=1)
    normales = validacion["y"] == 0
    return {"acierto": (prediccion == validacion["y"]).float().mean().item(),
            "falsas": (prediccion[normales] != 0).float().mean().item() if normales.any() else 0.0}


def adoptar(modelo: torch.nn.Module, archivo: str, medidas: dict) -> None:
    estado["modelo"], estado["archivo"], estado["medidas"] = modelo, Path(archivo).name, medidas
    print(f"[{NOMBRE}] modelo cargado: {estado['archivo']}")


def elegir_familia(familia: str) -> None:
    """Pasa a usar el modelo propio, la validación y la oferta que corresponden a esa familia de ataque."""
    if familia not in FAMILIAS[1:]:
        raise ValueError(f"Familia desconocida: {familia}")
    archivo = lambda variable: (os.environ.get(variable) or "").replace("<familia>", familia) or None
    leer_validacion(archivo("VALIDACION"))
    propio.update(familia=familia, pesos=archivo("PESOS"), ofrece=archivo("OFRECE"))


def reiniciar(comunicar: bool, familia: str | None = None) -> None:
    """Vuelve al punto de partida: su modelo propio, reputaciones iniciales y sin mensajes."""
    if familia and familia != propio["familia"]:
        elegir_familia(familia)
    modelo = leer_modelo(propio["pesos"])
    with candado:
        adoptar(modelo, propio["pesos"], medir(modelo))
        estado.update(comunicar=comunicar, mensajes=[], pendientes=set(PARES),
                      reputacion={par: REPUTACION_INICIAL for par in PARES})


# ---------------------------------------------------------------------------
# Percibir y decidir
# ---------------------------------------------------------------------------
@torch.no_grad()
def analizar(flujos: list[list[float]]) -> dict:
    """Clasifica cada flujo y decide qué hacer con él. Si ve un ataque, avisa a sus pares."""
    with candado:
        probabilidades = estado["modelo"](torch.tensor(flujos, dtype=torch.float32)).softmax(dim=1)
    confianzas, clases = probabilidades.max(dim=1)
    familias = [FAMILIAS[clase] for clase in clases.tolist()]
    acciones = ["permitir" if familia == "Benign" else
                "bloquear" if confianza >= CONFIANZA_PARA_BLOQUEAR else "alertar"
                for familia, confianza in zip(familias, confianzas.tolist())]

    ataques = Counter(familia for familia in familias if familia != "Benign")
    detalle = ", ".join(f"{n} {familia}" for familia, n in ataques.most_common()) or "ninguno"
    print(f"[{NOMBRE}] {len(flujos)} flujos | ataques: {detalle} | bloqueados: {acciones.count('bloquear')}")
    if ataques:
        familia, cantidad = ataques.most_common(1)[0]
        if cantidad >= MINIMO_PARA_AVISAR:
            ofrecer_a_los_pares(familia, cantidad)
    return {"familias": familias, "acciones": acciones}


# ---------------------------------------------------------------------------
# Cooperar: ofrecer un modelo y decidir sobre los que ofrecen los demás
# ---------------------------------------------------------------------------
def ofrecer_a_los_pares(familia: str, cantidad: int) -> None:
    """Avisa del ataque a cada par que todavía no respondió que sí (o que no quiere oír más)."""
    if not (estado["comunicar"] and propio["ofrece"]):
        return
    oferta = {"de": DIRECCION, "nombre": NOMBRE, "familia": familia, "cantidad": cantidad,
              "archivo": propio["ofrece"]}
    for par in sorted(estado["pendientes"]):
        try:
            pedido = urllib.request.Request(f"http://{par}:{PUERTO}/mensaje", data=json.dumps(oferta).encode(),
                                            headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(pedido, timeout=10) as respuesta:
                decision = json.loads(respuesta.read()).get("decision")
        except OSError:
            continue  # el par no responde: se intenta de nuevo la próxima vez
        print(f"[{NOMBRE}] ofreció {propio['ofrece']} a {par}: {decision}")
        if decision in ("acepta", "ya_lo_tiene", "ignora"):
            estado["pendientes"].discard(par)  # si lo rechazó, insiste en el próximo ataque que vea


def recibir_oferta(oferta: dict) -> dict:
    """Un par vio un ataque y ofrece un modelo. El agente decide si lo adopta."""
    with candado:
        if not estado["comunicar"]:
            return {"decision": "sin_comunicacion"}
        de, archivo = oferta["de"], Path(oferta["archivo"]).name
        reputacion = estado["reputacion"].get(de, REPUTACION_INICIAL)

        if reputacion < REPUTACION_MINIMA:
            decision, motivo = "ignora", f"ya no confía en {oferta['nombre']}: ni prueba el modelo"
        elif archivo == estado["archivo"]:
            decision, motivo = "ya_lo_tiene", "es el modelo que ya tiene"
        else:
            candidato = leer_modelo(archivo)
            nuevo, actual = medir(candidato), estado["medidas"]
            motivo = (f"con sus propios flujos acierta {nuevo['acierto']:.0%} (su modelo actual: "
                      f"{actual['acierto']:.0%}) y da {nuevo['falsas']:.0%} de falsas alarmas "
                      f"(actual: {actual['falsas']:.0%})")
            sirve = (nuevo["acierto"] >= actual["acierto"] - TOLERANCIA
                     and nuevo["falsas"] <= actual["falsas"] + TOLERANCIA)
            if sirve:
                decision = "acepta"
                reputacion = min(1.0, reputacion + PREMIO)
                adoptar(candidato, archivo, nuevo)
            else:
                decision = "rechaza"
                reputacion /= 2
            estado["reputacion"][de] = reputacion

        estado["mensajes"].append({
            "hora": time.time(), "de": oferta["nombre"], "para": NOMBRE, "familia": oferta["familia"],
            "cantidad": oferta["cantidad"], "archivo": archivo, "decision": decision, "motivo": motivo,
            "reputacion": reputacion,
        })
        print(f"[{NOMBRE}] oferta de {oferta['nombre']} ({archivo}): {decision} — {motivo}")
        return {"decision": decision, "motivo": motivo}


def resumen_del_estado() -> dict:
    return {"nombre": NOMBRE, "direccion": DIRECCION, "pesos": estado["archivo"], "propio": propio["pesos"],
            "familia": propio["familia"],
            "comunicar": estado["comunicar"], "medidas": estado["medidas"],
            "reputacion": estado["reputacion"], "mensajes": estado["mensajes"]}


# ---------------------------------------------------------------------------
# Servidor HTTP
# ---------------------------------------------------------------------------
class Pedidos(BaseHTTPRequestHandler):
    def responder(self, datos: dict, codigo: int = 200) -> None:
        cuerpo = json.dumps(datos).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        self.responder(resumen_del_estado())

    def do_POST(self):
        pedido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        try:
            if self.path == "/analizar":
                self.responder(analizar(pedido["flujos"]))
            elif self.path == "/mensaje":
                self.responder(recibir_oferta(pedido))
            elif self.path == "/reiniciar":
                reiniciar(bool(pedido.get("comunicar")), pedido.get("familia"))
                self.responder(resumen_del_estado())
            else:
                self.responder({"error": "ruta desconocida"}, 404)
        except Exception as error:  # el agente sigue en pie y avisa qué falló
            self.responder({"error": f"{type(error).__name__}: {error}"}, 500)

    def log_message(self, *_):
        pass  # sin el registro por defecto de cada pedido: ya se imprime un resumen propio


def leer_validacion(archivo: str) -> None:
    X, y = [], []
    with open(DIR_MODELOS / Path(archivo).name) as contenido:
        for fila in csv.DictReader(contenido):
            y.append(int(fila.pop(COLUMNA_ETIQUETA)))
            X.append([float(valor) for valor in fila.values()])
    validacion["X"], validacion["y"] = torch.tensor(X, dtype=torch.float32), torch.tensor(y)


if __name__ == "__main__":
    reiniciar(comunicar=False, familia=os.environ.get("FAMILIA", "PortScan"))
    print(f"[{NOMBRE}] escuchando en el puerto {PUERTO} | pares: {', '.join(PARES) or 'ninguno'}")
    ThreadingHTTPServer(("0.0.0.0", PUERTO), Pedidos).serve_forever()
