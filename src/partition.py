"""
partition.py — Reparto de los datos entre los nodos (organizaciones simuladas).

Tres modos:
  - iid:       todos los nodos reciben la misma mezcla de familias.
  - familia:   cada nodo ve tráfico benigno y solo algunas familias de ataque.
  - dirichlet: la proporción de cada familia en cada nodo se sortea con una
               distribución de Dirichlet. Con alpha chico (0.1) los nodos quedan
               muy distintos entre sí; con alpha grande (1.0 o más) se parecen.

Idea común a los tres: se arma una tabla "proporciones[familia, nodo]" que dice
qué fracción de las filas de cada familia le toca a cada nodo, y después se
reparte siguiendo esa tabla. El conjunto de prueba se reparte con la MISMA
tabla, así cada nodo tiene un test local con la misma mezcla que su train.

Una partición guarda solo los índices de las filas (no copia los datos).

Uso:
    python -m src.partition --modo iid
    python -m src.partition --modo familia
    python -m src.partition --modo dirichlet --alpha 0.1
"""
import argparse

import numpy as np

from src.config import ALPHA_DIRICHLET, DIR_PROCESSED, FAMILIAS, NUM_NODOS, SEMILLA
from src.train_utils import cargar_datos

MIN_FILAS_POR_NODO = 50  # ningún nodo puede quedar con menos filas de train que esto


# ---------------------------------------------------------------------------
# Tablas de proporciones [familia, nodo]
# ---------------------------------------------------------------------------
def proporciones_iid(num_nodos: int) -> np.ndarray:
    """Cada familia se reparte en partes iguales entre todos los nodos."""
    return np.full((len(FAMILIAS), num_nodos), 1.0 / num_nodos)


def proporciones_familia(num_nodos: int, familias_por_nodo: int = 2) -> np.ndarray:
    """
    Cada nodo ve solo `familias_por_nodo` familias de ataque consecutivas.
    Con 5 nodos:  nodo 0 -> DoS, DDoS      nodo 1 -> DDoS, PortScan
                  nodo 2 -> PortScan, BruteForce   ... etc.
    El tráfico benigno (familia 0) se reparte entre todos por igual.
    """
    num_ataques = len(FAMILIAS) - 1
    tiene = np.zeros((len(FAMILIAS), num_nodos))
    tiene[0, :] = 1  # todos los nodos tienen tráfico benigno
    for nodo in range(num_nodos):
        for k in range(familias_por_nodo):
            tiene[1 + (nodo + k) % num_ataques, nodo] = 1
    # Si una familia la tienen 2 nodos, cada uno recibe la mitad de sus filas
    return tiene / tiene.sum(axis=1, keepdims=True)


def proporciones_dirichlet(num_nodos: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Para cada familia sortea cómo se reparte entre los nodos (cada fila suma 1)."""
    return rng.dirichlet([alpha] * num_nodos, size=len(FAMILIAS))


# ---------------------------------------------------------------------------
# Reparto
# ---------------------------------------------------------------------------
def repartir(y: np.ndarray, proporciones: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Devuelve, para cada nodo, los índices de las filas de `y` que le tocan."""
    num_nodos = proporciones.shape[1]
    por_nodo = [[] for _ in range(num_nodos)]
    for familia in range(len(FAMILIAS)):
        indices = rng.permutation(np.flatnonzero(y == familia))
        # Puntos de corte: p. ej. proporciones 0.5/0.3/0.2 sobre 100 filas -> cortes en 50 y 80
        cortes = (np.cumsum(proporciones[familia])[:-1] * len(indices)).round().astype(int)
        for nodo, trozo in enumerate(np.split(indices, cortes)):
            por_nodo[nodo].append(trozo)
    return [np.sort(np.concatenate(trozos)) for trozos in por_nodo]


def crear_particion(modo: str, num_nodos: int = NUM_NODOS, alpha: float = ALPHA_DIRICHLET,
                    semilla: int = SEMILLA) -> dict:
    """
    Crea una partición: {"train": [índices del nodo 0, del nodo 1, ...], "test": [...]}.
    Con la misma semilla siempre devuelve exactamente el mismo reparto.
    """
    _, y_train, _, y_test = cargar_datos()
    rng = np.random.default_rng(semilla)

    for _ in range(100):
        if modo == "iid":
            proporciones = proporciones_iid(num_nodos)
        elif modo == "familia":
            proporciones = proporciones_familia(num_nodos)
        elif modo == "dirichlet":
            proporciones = proporciones_dirichlet(num_nodos, alpha, rng)
        else:
            raise ValueError(f"Modo desconocido: {modo}")

        particion = {"train": repartir(y_train, proporciones, rng),
                     "test": repartir(y_test, proporciones, rng)}
        # Con alpha muy chico puede salir algún nodo casi vacío: en ese caso se sortea de nuevo
        if min(len(indices) for indices in particion["train"]) >= MIN_FILAS_POR_NODO:
            return particion
    raise RuntimeError("No se logró un reparto con suficientes filas en todos los nodos.")


def quitar_familia(particion: dict, familia: str, nodo_que_la_conserva: int) -> dict:
    """
    Quita una familia de ataque del train de todos los nodos menos uno.
    Se usa para simular un ataque nuevo (zero-day) que solo un nodo ha visto.
    """
    _, y_train, _, _ = cargar_datos()
    clase = FAMILIAS.index(familia)
    train = [indices if nodo == nodo_que_la_conserva else indices[y_train[indices] != clase]
             for nodo, indices in enumerate(particion["train"])]
    return {"train": train, "test": particion["test"]}


# ---------------------------------------------------------------------------
# Guardado y carga (un archivo .npz con los índices de cada nodo)
# ---------------------------------------------------------------------------
def ruta_particion(nombre: str):
    return DIR_PROCESSED / f"particion_{nombre}.npz"


def guardar_particion(particion: dict, nombre: str) -> None:
    arreglos = {f"{parte}_{nodo}": indices
                for parte in ("train", "test")
                for nodo, indices in enumerate(particion[parte])}
    np.savez(ruta_particion(nombre), **arreglos)


def cargar_particion(nombre: str) -> dict:
    with np.load(ruta_particion(nombre)) as archivo:
        num_nodos = len(archivo.files) // 2
        return {parte: [archivo[f"{parte}_{nodo}"] for nodo in range(num_nodos)]
                for parte in ("train", "test")}


def resumen(particion: dict) -> str:
    """Tabla de texto: cuántas filas de train de cada familia tiene cada nodo."""
    _, y_train, _, _ = cargar_datos()
    lineas = ["  nodo " + "".join(f"{nombre:>11}" for nombre in FAMILIAS) + f"{'TOTAL':>11}"]
    for nodo, indices in enumerate(particion["train"]):
        conteo = np.bincount(y_train[indices], minlength=len(FAMILIAS))
        lineas.append(f"  {nodo:>4} " + "".join(f"{c:>11,}" for c in conteo) + f"{len(indices):>11,}")
    return "\n".join(lineas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reparto de los datos entre nodos")
    parser.add_argument("--modo", choices=["iid", "familia", "dirichlet"], default="iid")
    parser.add_argument("--alpha", type=float, default=ALPHA_DIRICHLET, help="solo para el modo dirichlet")
    parser.add_argument("--nodos", type=int, default=NUM_NODOS)
    args = parser.parse_args()

    nombre = args.modo if args.modo != "dirichlet" else f"dirichlet_{args.alpha}"
    particion = crear_particion(args.modo, args.nodos, args.alpha)
    guardar_particion(particion, nombre)
    print(f"Partición '{nombre}' ({args.nodos} nodos) — filas de train por familia:")
    print(resumen(particion))
    print(f"Guardada en {ruta_particion(nombre)}")
