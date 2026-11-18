"""
E3 (resumen) — Compara en una sola figura todas las familias probadas como zero-day.

No entrena nada: lee los CSV que dejó e3_zero_day (uno por familia) y dibuja
un panel por familia. Además guarda una tabla con los valores de la última ronda.

Uso:
    python -m experiments.e3_zero_day --familia DDoS    # una corrida por familia
    python -m experiments.e3_zero_day --familia Bot
    python -m experiments.e3_resumen                    # junta las que encuentre
"""
import argparse

import pandas as pd

from experiments.comun import (
    COLOR_REFERENCIA, dibujar_lineas, guardar_csv, guardar_figura, nueva_figura, preparar_ejes,
)
from experiments.e3_zero_day import SERIES
from src.config import DIR_RESULTADOS, FAMILIAS

COLUMNAS = 3  # paneles por fila
REFERENCIAS = {"aislado": ("Nodo aislado que no la vio", "-"), "autoencoder": ("Autoencoder de ese nodo", "--")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E3: resumen de todas las familias zero-day")
    parser.add_argument("--modelo", choices=["mlp", "transformer"], default="mlp")
    args = parser.parse_args()

    # --- Leer los CSV disponibles, en el orden de FAMILIAS ---
    tablas = {}
    for familia in FAMILIAS[1:]:
        ruta = DIR_RESULTADOS / f"e3_{args.modelo}_{familia}.csv"
        if ruta.exists():
            tablas[familia] = pd.read_csv(ruta)
    if not tablas:
        raise SystemExit(f"No hay resultados de E3 para el modelo {args.modelo} en {DIR_RESULTADOS}. "
                         "Ejecutá primero experiments.e3_zero_day.")

    # --- Tabla resumen: valores de la última ronda de cada familia ---
    filas = []
    for familia, tabla in tablas.items():
        ultima = tabla[tabla["ronda"] == tabla["ronda"].max()]
        fila = {"familia": familia, "rondas": int(tabla["ronda"].max())}
        for serie in SERIES:
            fila[serie] = float(ultima[ultima["serie"] == serie]["valor"].iloc[0])
        for referencia in REFERENCIAS:  # los CSV viejos no traen estas columnas
            fila[referencia] = float(tabla[referencia].iloc[0]) if referencia in tabla else float("nan")
        filas.append(fila)
    nombre = f"e3_{args.modelo}_resumen"
    resumen = guardar_csv(filas, nombre)

    print(f"E3 resumen | modelo {args.modelo} | última ronda de cada familia")
    print(f"  {'Familia':<12}{'reconocida':>12}{'detectada':>12}{'aislado':>10}{'autoencoder':>13}")
    for fila in filas:
        print(f"  {fila['familia']:<12}{fila['recall']:>12.1%}{fila['detectado']:>12.1%}"
              f"{fila['aislado']:>10.1%}{fila['autoencoder']:>13.1%}")

    # --- Figura: un panel por familia ---
    n_filas = -(-len(tablas) // COLUMNAS)  # división redondeada hacia arriba
    fig, ejes = nueva_figura(n_filas, COLUMNAS, f"E3 — Zero-day: cada familia vista por un solo nodo ({args.modelo})")
    paneles = ejes.flatten()
    for ax, (familia, tabla) in zip(paneles, tablas.items()):
        dibujar_lineas(ax, tabla, "valor", "serie", SERIES)
        for referencia, (texto, estilo) in REFERENCIAS.items():
            if referencia in tabla:
                ax.axhline(tabla[referencia].iloc[0], color=COLOR_REFERENCIA, linewidth=1.5,
                           linestyle=estilo, label=texto)
        preparar_ejes(ax, familia, xlabel="Ronda", ylabel="Fracción de los flujos")
        ax.set_ylim(-0.02, 1.02)
    for ax in paneles[len(tablas):]:  # paneles que sobran
        ax.set_visible(False)
    guardar_figura(fig, nombre)
