"""
centralized.py — Líneas base sin federación.

  - Centralizado: un solo modelo entrenado con TODOS los datos juntos. Es el
    techo ideal: lo que se lograría si las organizaciones compartieran sus datos.
  - Aislado: cada nodo entrena su propio modelo solo con sus datos. Es el piso:
    lo que logra cada organización sin colaborar.
  - Autoencoder: detector de anomalías entrenado solo con tráfico benigno.

Uso:
    python -m src.centralized --modelo mlp --epocas 20
    python -m src.centralized --modelo transformer --epocas 10 --modo dirichlet --alpha 0.5
"""
import argparse

import numpy as np
import pandas as pd
import torch

from src.config import ALPHA_DIRICHLET, DIR_RESULTADOS, DISPOSITIVO, FAMILIAS, NUM_NODOS, RONDAS, SEMILLA
from src.partition import crear_particion
from src.train_utils import (
    cargar_datos,
    crear_modelo,
    entrenar,
    entrenar_y_evaluar,
    graficar_matriz,
    predecir,
)


def entrenar_centralizado(nombre_modelo: str, epocas: int) -> tuple[torch.nn.Module, list[dict]]:
    """Entrena un modelo con todo el train. Devuelve el modelo y sus métricas por época."""
    X_train, y_train, _, _ = cargar_datos()
    modelo = crear_modelo(nombre_modelo)
    return modelo, entrenar_y_evaluar(modelo, X_train, y_train, epocas)


def entrenar_aislados(particion: dict, nombre_modelo: str, epocas: int) -> list[list[dict]]:
    """
    Entrena un modelo por nodo, cada uno solo con sus datos.
    Devuelve una lista (un elemento por nodo) con las métricas por época.
    Cada modelo se mide en el test global ("f1_macro") y en el test local del nodo ("f1_local").
    """
    X_train, y_train, X_test, y_test = cargar_datos()
    historiales = []
    for indices_train, indices_test in zip(particion["train"], particion["test"]):
        modelo = crear_modelo(nombre_modelo)
        historiales.append(entrenar_y_evaluar(
            modelo, X_train[indices_train], y_train[indices_train], epocas,
            X_local=X_test[indices_test], y_local=y_test[indices_test],
        ))
    return historiales


def promediar_nodos(historiales: list[list[dict]]) -> list[dict]:
    """Promedia entre nodos, época por época, las métricas de los modelos aislados."""
    tabla = pd.concat([pd.DataFrame(h) for h in historiales])
    return tabla.groupby("ronda").mean().reset_index().to_dict("records")


# ---------------------------------------------------------------------------
# Autoencoder: detección de anomalías
# ---------------------------------------------------------------------------
@torch.no_grad()
def errores_reconstruccion(autoencoder: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Error de reconstrucción de cada flujo."""
    autoencoder.to(DISPOSITIVO).eval()
    partes = [autoencoder.error(torch.from_numpy(X[i:i + 2048]).to(DISPOSITIVO)).cpu().numpy()
              for i in range(0, len(X), 2048)]
    return np.concatenate(partes)


def entrenar_autoencoder(X_benigno: np.ndarray, epocas: int, percentil: float = 95.0):
    """
    Entrena el autoencoder solo con tráfico benigno y fija el umbral de alarma:
    el error que deja por debajo al `percentil`% del tráfico benigno de train.
    (Con percentil 95 se acepta de antemano ~5% de falsos positivos.)
    """
    autoencoder = crear_modelo("autoencoder")
    entrenar(autoencoder, X_benigno, None, epocas)
    umbral = float(np.percentile(errores_reconstruccion(autoencoder, X_benigno), percentil))
    return autoencoder, umbral


def evaluar_autoencoder(autoencoder, umbral: float) -> dict:
    """Sobre el test global: tasa de falsos positivos y fracción detectada de cada familia."""
    _, _, X_test, y_test = cargar_datos()
    es_anomalia = errores_reconstruccion(autoencoder, X_test) > umbral
    metricas = {"fpr": float(es_anomalia[y_test == 0].mean())}
    for clase, nombre in enumerate(FAMILIAS[1:], start=1):
        metricas[f"detectado_{nombre}"] = float(es_anomalia[y_test == clase].mean())
    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Líneas base: centralizado, aislado y autoencoder")
    parser.add_argument("--modelo", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--epocas", type=int, default=RONDAS)
    parser.add_argument("--modo", choices=["iid", "familia", "dirichlet"], default="dirichlet",
                        help="cómo se reparten los datos para los modelos aislados")
    parser.add_argument("--alpha", type=float, default=ALPHA_DIRICHLET)
    args = parser.parse_args()
    X_train, y_train, X_test, y_test = cargar_datos()

    print(f"Dispositivo: {DISPOSITIVO} | modelo: {args.modelo} | épocas: {args.epocas}\n")

    # --- Centralizado ---
    modelo, historial = entrenar_centralizado(args.modelo, args.epocas)
    final = historial[-1]
    print(f"CENTRALIZADO   F1 macro {final['f1_macro']:.3f} | exactitud {final['exactitud']:.3f} | "
          f"falsos positivos {final['fpr']:.3f}")
    print("  recall por familia: " + "  ".join(f"{n} {final[f'recall_{n}']:.2f}" for n in FAMILIAS))
    ruta_matriz = DIR_RESULTADOS / f"centralizado_{args.modelo}_matriz.png"
    graficar_matriz(y_test, predecir(modelo, X_test), f"Matriz de confusión — {args.modelo} centralizado", ruta_matriz)

    # --- Aislados ---
    particion = crear_particion(args.modo, NUM_NODOS, args.alpha, SEMILLA)
    aislados = entrenar_aislados(particion, args.modelo, args.epocas)
    print(f"\nAISLADOS (reparto {args.modo})")
    for nodo, h in enumerate(aislados):
        print(f"  nodo {nodo}: F1 macro global {h[-1]['f1_macro']:.3f} | F1 macro local {h[-1]['f1_local']:.3f} | "
              f"falsos positivos {h[-1]['fpr']:.3f} | filas {len(particion['train'][nodo]):,}")
    promedio = promediar_nodos(aislados)[-1]
    print(f"  promedio: F1 macro global {promedio['f1_macro']:.3f} | F1 macro local {promedio['f1_local']:.3f}")

    # --- Autoencoder ---
    autoencoder, umbral = entrenar_autoencoder(X_train[y_train == 0], args.epocas)
    metricas_ae = evaluar_autoencoder(autoencoder, umbral)
    print(f"\nAUTOENCODER (solo vio tráfico benigno)   falsos positivos {metricas_ae['fpr']:.3f}")
    print("  detectado por familia: " + "  ".join(f"{n} {metricas_ae[f'detectado_{n}']:.2f}" for n in FAMILIAS[1:]))

    # --- CSV con las métricas por época ---
    filas = [{"metodo": "centralizado", **f} for f in historial]
    filas += [{"metodo": f"aislado_nodo{nodo}", **f} for nodo, h in enumerate(aislados) for f in h]
    ruta_csv = DIR_RESULTADOS / f"centralizado_{args.modelo}_metricas.csv"
    pd.DataFrame(filas).to_csv(ruta_csv, index=False)
    print(f"\nGuardado: {ruta_csv.name} y {ruta_matriz.name} en {DIR_RESULTADOS}")
