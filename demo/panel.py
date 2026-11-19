"""
panel.py — Panel web de la demo (http://localhost:8080).

Sirve una página que muestra los tres nodos y permite, con botones:
  - elegir qué pesos tiene cargados cada nodo (las tres fases de la demo), y
  - enviarles el tráfico y ver, flujo por flujo, qué hizo cada nodo con él.

El panel no clasifica nada: solo reenvía los pedidos a los nodos (los otros
contenedores) y dibuja lo que responden. Es la versión visual de ataque.py.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from demo.ataque import FAMILIA, FASES, MODELO, NODOS, PUERTO, esperar_nodos, leer_flujos, pedir

PAGINA = Path(__file__).with_name("panel.html").read_bytes()
BENIGNOS, ATAQUE = [], []  # se llenan al arrancar


def pesos_de_fase(numero: int, posicion: int) -> str:
    """Archivo de pesos que le toca al nodo `posicion` (0, 1, 2) en la fase `numero` (1, 2, 3)."""
    if numero == 1:  # sin federación: cada nodo vuelve a su propio modelo aislado
        return f"e3_{MODELO}_{FAMILIA}_aislado_nodo{posicion + 1}.pt"
    return FASES[numero - 1][1]


def estado() -> dict:
    return {"familia": FAMILIA, "fases": [titulo for titulo, _ in FASES],
            "flujos_ataque": len(ATAQUE), "flujos_normales": len(BENIGNOS),
            "nodos": [pedir(nodo, "/estado") for nodo in NODOS]}


def cambiar_fase(numero: int) -> dict:
    """El 'servidor' entrega a cada nodo los pesos de esa fase."""
    for posicion, nodo in enumerate(NODOS):
        pedir(nodo, "/modelo", {"archivo": pesos_de_fase(numero, posicion)})
    return estado()


def atacar(desde: int, hasta: int) -> dict:
    """Envía a cada nodo los flujos [desde, hasta) de cada tipo y devuelve lo que respondió."""
    return {"nodos": [{"ataque": pedir(nodo, "/analizar", {"flujos": ATAQUE[desde:hasta]})["familias"],
                       "normal": pedir(nodo, "/analizar", {"flujos": BENIGNOS[desde:hasta]})["familias"]}
                      for nodo in NODOS]}


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
