"""
config.py — Configuración central del proyecto.

Acá viven las rutas, las semillas y (más adelante) los hiperparámetros.
Todos los demás scripts importan desde este archivo, así cualquier cambio
se hace en un solo lugar.
"""
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Rutas (todas relativas a la raíz del proyecto, sin importar desde dónde se ejecute)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"              # CSV originales de CIC-IDS2017
DIR_PROCESSED = RAIZ / "data" / "processed"  # parquet limpio y particiones
DIR_RESULTADOS = RAIZ / "experiments" / "resultados"  # CSV y PNG de los experimentos

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEMILLA = 42


def fijar_semillas(semilla: int = SEMILLA) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para que los resultados se repitan."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


# ---------------------------------------------------------------------------
# Dispositivo de cómputo
# ---------------------------------------------------------------------------
def obtener_dispositivo() -> torch.device:
    """
    Elige dónde se hacen los cálculos.

    - En Mac con Apple Silicon se usa la GPU (MPS) si está disponible.
    - En Mac Intel también puede existir MPS (con placa AMD), pero en las
      pruebas resultó ~15 veces más lento que la CPU para el Transformer,
      así que ahí se usa CPU.
    - Se puede forzar con la variable de entorno CIBER_DISPOSITIVO=mps o =cpu.
    """
    forzado = os.environ.get("CIBER_DISPOSITIVO")
    if forzado:
        return torch.device(forzado)
    if torch.backends.mps.is_available() and platform.machine() == "arm64":
        return torch.device("mps")
    return torch.device("cpu")


DISPOSITIVO = obtener_dispositivo()

# ---------------------------------------------------------------------------
# Federación
# ---------------------------------------------------------------------------
NUM_NODOS = 5           # cantidad de organizaciones simuladas
RONDAS = 20             # rondas de comunicación servidor <-> nodos
EPOCAS_LOCALES = 1      # pasadas sobre sus datos que hace cada nodo por ronda
ALPHA_DIRICHLET = 0.5   # heterogeneidad del reparto: más chico = nodos más distintos
MU_FEDPROX = 0.1        # fuerza del término proximal de FedProx
HILOS_POR_NODO = 2      # hilos de CPU que usa cada nodo simulado

# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
TAM_LOTE = 64
TASA_APRENDIZAJE = 1e-3

# ---------------------------------------------------------------------------
# Arquitectura del FlowTransformer
# ---------------------------------------------------------------------------
D_MODELO = 64     # tamaño del vector que representa a cada token
NUM_CABEZAS = 4   # cabezas de atención
NUM_CAPAS = 2     # capas del encoder

# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
# Familias de ataque. El orden define el número de clase: Benign=0, DoS=1, ...
FAMILIAS = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Web", "Bot"]
COLUMNA_ETIQUETA = "label"  # nombre de la columna de clase en los parquet

FRACCION_TEST = 0.2  # 80% entrenamiento, 20% prueba

RUTA_TRAIN = DIR_PROCESSED / "train.parquet"
RUTA_TEST = DIR_PROCESSED / "test.parquet"
RUTA_METADATA = DIR_PROCESSED / "metadata.json"  # columnas, conteos y parámetros de normalización
