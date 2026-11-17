"""
config.py — Configuración central del proyecto.

Acá viven las rutas, las semillas y (más adelante) los hiperparámetros.
Todos los demás scripts importan desde este archivo, así cualquier cambio
se hace en un solo lugar.
"""
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
DIR_EXPERIMENTS = RAIZ / "experiments"       # resultados (CSV y PNG)

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
    """Devuelve la GPU de Apple (MPS) si está disponible; si no, la CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DISPOSITIVO = obtener_dispositivo()

# ---------------------------------------------------------------------------
# Federación
# ---------------------------------------------------------------------------
NUM_NODOS = 5  # cantidad de organizaciones simuladas

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
