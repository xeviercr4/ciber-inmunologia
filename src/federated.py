"""
federated.py — Aprendizaje federado con la simulación de Flower.

Cómo funciona una ronda:
  1. El servidor envía los pesos globales a todos los nodos.
  2. Cada nodo entrena con sus datos privados y devuelve sus pesos nuevos.
  3. El servidor combina (agrega) los pesos de todos y obtiene el nuevo modelo global.
Los datos nunca salen del nodo: por la red solo viajan pesos.

Estrategias disponibles:
  - fedavg:  promedio de los pesos, ponderado por la cantidad de datos de cada nodo.
  - fedprox: como FedAvg, pero cada nodo es penalizado si se aleja mucho del
             modelo global (ayuda cuando los nodos tienen datos muy distintos).
  - local:   variante personalizada. Solo se federa el encoder; cada nodo
             conserva su propia cabeza de clasificación, que nunca sale del nodo.
  - krum:    agregación robusta. Se queda con la actualización del nodo más
             parecido a la mayoría y descarta el resto.
  - mediana: agregación robusta. Usa la mediana de cada peso en vez del promedio.

Nodos maliciosos: entrenan con las etiquetas invertidas para sabotear el modelo.

Uso:
    python -m src.federated --estrategia fedavg --modo dirichlet --alpha 0.5
    python -m src.federated --estrategia krum --modo iid --nodos 10 --maliciosos 2
"""
import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ArrayRecord, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedMedian, FedProx, Krum
from flwr.simulation import run_simulation

from src.config import (
    ALPHA_DIRICHLET,
    DIR_RESULTADOS,
    EPOCAS_LOCALES,
    FAMILIAS,
    GPU_POR_NODO,
    HILOS_POR_NODO,
    MU_FEDPROX,
    NUM_NODOS,
    RONDAS,
    SEMILLA,
)
from src.partition import cargar_particion, crear_particion, guardar_particion, ruta_particion
from src.train_utils import (
    cargar_datos,
    cargar_pesos,
    crear_modelo,
    entrenar,
    evaluar,
    guardar_modelo,
    obtener_pesos,
)

ESTRATEGIAS = ["fedavg", "fedprox", "local", "krum", "mediana"]

logging.getLogger("flwr").setLevel(logging.WARNING)  # oculta los mensajes informativos de Flower


class SinAvisosDeObsolescencia(logging.Filter):
    """
    Oculta el aviso que las versiones nuevas de Flower repiten en cada simulación:
    `run_simulation` será reemplazada por el comando `flwr run`. Hoy funciona igual,
    y la versión de Flower está fijada en los archivos de requisitos.
    """

    def filter(self, registro: logging.LogRecord) -> bool:
        return "deprecated" not in registro.getMessage().lower()


logging.getLogger("flwr").addFilter(SinAvisosDeObsolescencia())
warnings.filterwarnings("ignore", category=FutureWarning, module=r"ray\..*")  # aviso informativo de Ray


def invertir_etiquetas(y: np.ndarray) -> np.ndarray:
    """
    Ataque de inversión de etiquetas: cada clase se cambia por su "espejo".
    Benign(0) <-> Bot(6),  DoS(1) <-> Web(5),  DDoS(2) <-> BruteForce(4).
    El efecto más dañino: el nodo enseña que el tráfico normal es un ataque y viceversa.
    """
    return (len(FAMILIAS) - 1) - y


# ---------------------------------------------------------------------------
# Cliente: un nodo (una organización)
# ---------------------------------------------------------------------------
class NodoIDS(NumPyClient):
    def __init__(self, nodo: int, contexto: Context, opciones: dict):
        self.nodo = nodo
        self.contexto = contexto  # contexto.state es la "memoria" del nodo entre rondas
        self.personalizado = opciones["estrategia"] == "local"

        # Datos privados del nodo: solo las filas que le asigna la partición
        X_train, y_train, X_test, y_test = cargar_datos()
        particion = cargar_particion(opciones["particion"])
        self.X, self.y = X_train[particion["train"][nodo]], y_train[particion["train"][nodo]]
        self.X_local, self.y_local = X_test[particion["test"][nodo]], y_test[particion["test"][nodo]]
        if nodo in opciones["maliciosos"]:
            self.y = invertir_etiquetas(self.y)

        torch.set_num_threads(HILOS_POR_NODO)
        self.modelo = crear_modelo(opciones["modelo"])

    def parte_federada(self) -> torch.nn.Module:
        """Lo que viaja por la red: el modelo entero o, en la variante personalizada, solo el encoder."""
        return self.modelo.encoder if self.personalizado else self.modelo

    def recibir(self, pesos: list[np.ndarray]) -> None:
        """Carga los pesos que llegan del servidor y, si corresponde, la cabeza local guardada."""
        cargar_pesos(self.parte_federada(), pesos)
        if self.personalizado and "cabeza" in self.contexto.state:
            cargar_pesos(self.modelo.cabeza, self.contexto.state["cabeza"].to_numpy_ndarrays())

    def fit(self, pesos, config):
        """Una ronda de entrenamiento local."""
        self.recibir(pesos)
        entrenar(
            self.modelo, self.X, self.y,
            epocas=config["epocas"],
            semilla=SEMILLA + 1000 * config["ronda"] + self.nodo,  # distinta por ronda y nodo, pero fija
            mu=config.get("proximal_mu", 0.0),                     # solo FedProx envía este valor
        )
        if self.personalizado:
            # La cabeza se guarda en la memoria del nodo: nunca se envía al servidor
            self.contexto.state["cabeza"] = ArrayRecord(obtener_pesos(self.modelo.cabeza))
        return obtener_pesos(self.parte_federada()), len(self.X), {"nodo": self.nodo}

    def evaluate(self, pesos, config):
        """Mide el modelo del nodo: siempre en su test local y, si es personalizado, también en el global."""
        self.recibir(pesos)
        metricas = {"nodo": self.nodo,
                    "f1_local": evaluar(self.modelo, self.X_local, self.y_local).get("f1_macro", float("nan"))}
        if self.personalizado:
            # No existe un modelo global completo (cada nodo tiene su cabeza),
            # así que el test global lo mide cada nodo con su propio modelo.
            _, _, X_test, y_test = cargar_datos()
            metricas.update(evaluar(self.modelo, X_test, y_test))
        return 0.0, len(self.y_local), metricas


# ---------------------------------------------------------------------------
# Servidor: estrategias de agregación
# ---------------------------------------------------------------------------
class EnOrden:
    """
    Ordena por número de nodo los resultados antes de agregarlos. Flower los
    entrega en orden de llegada (que cambia entre corridas); al ordenarlos, la
    suma se hace siempre igual y el resultado es reproducible bit a bit.
    """

    def aggregate_fit(self, server_round, results, failures):
        if failures:  # si un nodo falla, se corta con el error en vez de seguir con datos incompletos
            raise RuntimeError(f"Fallaron {len(failures)} nodos en la ronda {server_round}: {failures[0]}")
        results = sorted(results, key=lambda resultado: resultado[1].metrics["nodo"])
        return super().aggregate_fit(server_round, results, failures)


class FedAvgOrdenado(EnOrden, FedAvg): pass
class FedProxOrdenado(EnOrden, FedProx): pass
class KrumOrdenado(EnOrden, Krum): pass
class MedianaOrdenada(EnOrden, FedMedian): pass


def crear_estrategia(nombre: str, mu: float, num_maliciosos: int, **comunes):
    """Devuelve la estrategia de agregación pedida."""
    if nombre in ("fedavg", "local"):  # "local" agrega igual que FedAvg, pero solo recibe encoders
        return FedAvgOrdenado(**comunes)
    if nombre == "fedprox":
        return FedProxOrdenado(proximal_mu=mu, **comunes)
    if nombre == "krum":
        # Krum necesita saber cuántos nodos maliciosos se quieren tolerar (al menos 1)
        return KrumOrdenado(num_malicious_clients=max(num_maliciosos, 1), **comunes)
    if nombre == "mediana":
        return MedianaOrdenada(**comunes)
    raise ValueError(f"Estrategia desconocida: {nombre}")


# ---------------------------------------------------------------------------
# Simulación completa
# ---------------------------------------------------------------------------
def ejecutar_federado(estrategia: str, particion: dict, modelo: str = "mlp", rondas: int = RONDAS,
                      epocas: int = EPOCAS_LOCALES, mu: float = MU_FEDPROX,
                      maliciosos: tuple[int, ...] = (), mostrar: bool = True,
                      guardar_como: str | None = None) -> list[dict]:
    """
    Corre una simulación federada y devuelve una fila de métricas por ronda.
    Si se pasa `guardar_como`, al terminar guarda los pesos globales en un .pt.

    - "f1_macro", "fpr", "recall_X"...: medidos en el test global. Con las
      estrategias normales los calcula el servidor con el modelo global; con
      "local" son el promedio de los modelos personalizados de los nodos.
    - "f1_local": promedio entre nodos del F1 macro en el test local de cada uno.
    """
    num_nodos = len(particion["train"])
    personalizado = estrategia == "local"
    _, _, X_test, y_test = cargar_datos()

    # Los nodos corren en procesos separados: leen la partición desde un archivo temporal
    nombre_particion = f"tmp_{os.getpid()}"
    guardar_particion(particion, nombre_particion)
    opciones = {"estrategia": estrategia, "particion": nombre_particion,
                "modelo": modelo, "maliciosos": tuple(maliciosos)}

    modelo_global = crear_modelo(modelo)
    parte_federada = modelo_global.encoder if personalizado else modelo_global
    historial: dict[int, dict] = {}  # ronda -> métricas

    def evaluar_en_servidor(ronda: int, pesos: list[np.ndarray], _config):
        """Después de cada ronda, el servidor mide el modelo global en el test global."""
        cargar_pesos(parte_federada, pesos)  # el servidor se queda con los últimos pesos globales
        if personalizado:
            return None  # sin cabeza global no hay modelo completo que medir: lo miden los nodos
        metricas = evaluar(modelo_global, X_test, y_test)
        historial.setdefault(ronda, {}).update(metricas)
        return 0.0, {"f1_macro": metricas["f1_macro"]}

    def promediar_metricas_de_nodos(resultados):
        """Promedia (entre nodos) las métricas que cada nodo midió por su cuenta."""
        ronda = max(historial, default=0) + 1 if personalizado else max(historial)
        metricas = sorted((m for _, m in resultados), key=lambda m: m["nodo"])  # orden fijo => reproducible
        claves = [clave for clave in metricas[0] if clave != "nodo"]
        promedio = {clave: float(np.nanmean([m[clave] for m in metricas])) for clave in claves}
        fila = historial.setdefault(ronda, {})
        fila.update(promedio)
        if mostrar:
            print(f"    ronda {ronda:>2}: F1 macro global {fila['f1_macro']:.3f} | "
                  f"F1 macro local {fila['f1_local']:.3f} | falsos positivos {fila['fpr']:.3f}")
        return {}

    def crear_servidor(_contexto: Context) -> ServerAppComponents:
        estrategia_flower = crear_estrategia(
            estrategia, mu, len(maliciosos),
            fraction_fit=1.0, fraction_evaluate=1.0,           # participan todos los nodos en cada ronda
            min_fit_clients=num_nodos, min_evaluate_clients=num_nodos, min_available_clients=num_nodos,
            initial_parameters=ndarrays_to_parameters(obtener_pesos(parte_federada)),
            evaluate_fn=evaluar_en_servidor,
            on_fit_config_fn=lambda ronda: {"ronda": ronda, "epocas": epocas},
            fit_metrics_aggregation_fn=lambda _resultados: {},
            evaluate_metrics_aggregation_fn=promediar_metricas_de_nodos,
        )
        return ServerAppComponents(strategy=estrategia_flower, config=ServerConfig(num_rounds=rondas))

    def crear_nodo(contexto: Context):
        return NodoIDS(int(contexto.node_config["partition-id"]), contexto, opciones).to_client()

    try:
        run_simulation(
            server_app=ServerApp(server_fn=crear_servidor),
            client_app=ClientApp(client_fn=crear_nodo),
            num_supernodes=num_nodos,
            backend_config={
                "client_resources": {"num_cpus": HILOS_POR_NODO, "num_gpus": GPU_POR_NODO},
                "init_args": {"logging_level": logging.ERROR, "log_to_driver": False},
            },
        )
    finally:
        ruta_particion(nombre_particion).unlink(missing_ok=True)

    if guardar_como:
        guardar_modelo(parte_federada, modelo, guardar_como, parte="encoder" if personalizado else "completo")
    return [{"ronda": ronda, **historial[ronda]} for ronda in sorted(historial)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de aprendizaje federado")
    parser.add_argument("--estrategia", choices=ESTRATEGIAS, default="fedavg")
    parser.add_argument("--modelo", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--modo", choices=["iid", "familia", "dirichlet"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=ALPHA_DIRICHLET)
    parser.add_argument("--nodos", type=int, default=NUM_NODOS)
    parser.add_argument("--rondas", type=int, default=RONDAS)
    parser.add_argument("--maliciosos", type=int, default=0, help="cantidad de nodos maliciosos (los primeros)")
    args = parser.parse_args()

    print(f"Federado: {args.estrategia} | modelo {args.modelo} | reparto {args.modo} | "
          f"{args.nodos} nodos ({args.maliciosos} maliciosos) | {args.rondas} rondas")
    particion = crear_particion(args.modo, args.nodos, args.alpha, SEMILLA)
    nombre = f"federado_{args.estrategia}_{args.modelo}"
    filas = ejecutar_federado(args.estrategia, particion, args.modelo, args.rondas,
                              maliciosos=tuple(range(args.maliciosos)), guardar_como=nombre)

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(DIR_RESULTADOS / f"{nombre}.csv", index=False)
    print(f"Guardado en {DIR_RESULTADOS}: {nombre}.csv y modelos/{nombre}.pt")
