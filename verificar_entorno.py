"""
verificar_entorno.py — Comprueba que el entorno quedó bien instalado.

Uso:  python verificar_entorno.py
"""
import sys

print(f"Python {sys.version.split()[0]}")

# Importamos cada dependencia y mostramos su versión
import torch
import numpy
import sklearn
import pandas
import pyarrow
import matplotlib
import yaml
import flwr
import ray  # lo instala flwr[simulation]; es el motor de la simulación

for modulo in (torch, numpy, sklearn, pandas, pyarrow, matplotlib, yaml, flwr, ray):
    print(f"  {modulo.__name__:<12} {modulo.__version__}")

# Dispositivo que va a usar el proyecto
from src.config import DISPOSITIVO, fijar_semillas

fijar_semillas()
print(f"Dispositivo: {DISPOSITIVO}")

# Prueba mínima: una operación con tensores en ese dispositivo
x = torch.randn(256, 64, device=DISPOSITIVO)
y = (x @ x.T).sum().item()
print(f"Operación de prueba en {DISPOSITIVO}: OK  (MPS disponible: {torch.backends.mps.is_available()})")
print("Entorno verificado correctamente.")
