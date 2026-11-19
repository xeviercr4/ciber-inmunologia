"""
panel.py — Panel web de la demo (http://localhost:8080).

Sirve una página que muestra a los agentes y permite, con botones:
  - elegir la fase (si los agentes pueden hablar entre ellos y si participa el malicioso), y
  - enviarles el tráfico y ver, flujo por flujo, qué hizo cada uno y qué se dijeron.

El panel no clasifica nada ni reparte modelos: solo envía tráfico a los agentes
(los otros contenedores) y dibuja lo que responden. Es la versión visual de ataque.py.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from demo.ataque import (
    ESPECIALISTA, FAMILIA, FASES, MALICIOSO, PUERTO, conversacion, enviar_lote, esperar_nodos, leer_flujos,
    nodos_de_fase, pedir, preparar_fase,
)

PAGINA = Path(__file__).with_name("panel.html").read_bytes()
BENIGNOS, ATAQUE = [], []  # se llenan al arrancar
fase_actual = 1


def estado() -> dict:
    """Lo que la página necesita para dibujarse: los agentes de la fase y su conversación."""
    nodos = [pedir(nodo, "/estado") for nodo in nodos_de_fase(fase_actual)]
    return {"familia": FAMILIA, "fases": FASES, "fase": fase_actual,
            "especialista": ESPECIALISTA, "malicioso": MALICIOSO,
            "flujos_ataque": len(ATAQUE), "flujos_normales": len(BENIGNOS),
            "nodos": nodos, "conversacion": conversacion(nodos)}


def cambiar_fase(numero: int) -> dict:
    global fase_actual
    fase_actual = numero
    preparar_fase(numero)
    return estado()


def atacar(desde: int, hasta: int) -> dict:
    """Envía un lote a cada agente y devuelve lo que respondieron y cómo quedaron después."""
    nodos = nodos_de_fase(fase_actual)
    respuestas = enviar_lote(nodos, ATAQUE, BENIGNOS, desde, hasta)
    return {"lote": [respuestas[nodo] for nodo in nodos], **estado()}


class Pedidos(BaseHTTPRequestHandler):
    def responder(self, cuerpo: bytes, tipo: str = "application/json", codigo: int = 200) -> None:
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def responder_json(self, funcion, *argumentos) -> None:
        try:
            self.responder(json.dumps(funcion(*argumentos)).encode())
        except (Exception, SystemExit) as error:  # pedir() termina con SystemExit si un nodo falla
            self.responder(json.dumps({"error": str(error)}).encode(), codigo=500)

    def do_GET(self):
        if self.path.startswith("/api/estado"):
            self.responder_json(estado)
        else:
            self.responder(PAGINA, "text/html; charset=utf-8")

    def do_POST(self):
        pedido = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/fase":
            self.responder_json(cambiar_fase, int(pedido["numero"]))
        elif self.path == "/api/atacar":
            self.responder_json(atacar, int(pedido["desde"]), int(pedido["hasta"]))
        else:
            self.responder(b'{"error": "ruta desconocida"}', codigo=404)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    esperar_nodos()
    BENIGNOS, ATAQUE = leer_flujos()
    cambiar_fase(1)  # siempre se arranca desde la fase 1
    print("Panel listo: abrí http://localhost:8080 en el navegador")
    ThreadingHTTPServer(("0.0.0.0", PUERTO), Pedidos).serve_forever()
