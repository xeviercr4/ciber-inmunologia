"""
E3 (resumen) — Compara en una sola figura todas las familias probadas como zero-day.

No entrena nada: lee los CSV que dejó e3_zero_day (uno por familia) y dibuja
un panel por familia. Además guarda una tabla con los valores de la última ronda.

Uso:
    python -m experiments.e3_zero_day --familia DDoS    # una corrida por familia
    python -m experiments.e3_zero_day --familia Bot
    python -m experiments.e3_resumen                    # junta las que encuentre
    python -m experiments.e3_resumen --mascara          # lo mismo para las corridas con --mascara,
                                                        # comparadas con las corridas sin máscara
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
    parser.add_argument("--mascara", action="store_true", help="resumir las corridas hechas con --mascara")
    args = parser.parse_args()
    sufijo = "_mascara" if args.mascara else ""

    # --- Leer los CSV disponibles, en el orden de FAMILIAS ---
    tablas = {}
    for familia in FAMILIAS[1:]:
        ruta = DIR_RESULTADOS / f"e3_{args.modelo}_{familia}{sufijo}.csv"
        if ruta.exists():
            tablas[familia] = pd.read_csv(ruta)
    if not tablas:
        raise SystemExit(f"No hay resultados de E3{sufijo} para el modelo {args.modelo} en {DIR_RESULTADOS}. "
                         "Ejecutá primero experiments.e3_zero_day.")

    # --- Tabla resumen: valores de la última ronda de cada familia ---
    filas = []
    for familia, tabla in tablas.items():
        ultima = tabla[tabla["ronda"] == tabla["ronda"].max()]
        fila = {"familia": familia, "rondas": int(tabla["ronda"].max()),
                "f1_macro": float(ultima["f1_macro"].iloc[0]), "fpr": float(ultima["fpr"].iloc[0])}
        for serie in SERIES:
            fila[serie] = float(ultima[ultima["serie"] == serie]["valor"].iloc[0])
        for referencia in REFERENCIAS:  # los CSV viejos no traen estas columnas
            fila[referencia] = float(tabla[referencia].iloc[0]) if referencia in tabla else float("nan")
        # Con --mascara se agrega, como comparación, la misma familia sin máscara (si ya se corrió)
        ruta_base = DIR_RESULTADOS / f"e3_{args.modelo}_{familia}.csv"
        if args.mascara and ruta_base.exists():
            base = pd.read_csv(ruta_base)
            base = base[base["ronda"] == base["ronda"].max()]
            for serie in SERIES:
                fila[f"{serie}_sin_mascara"] = float(base[base["serie"] == serie]["valor"].iloc[0])
        filas.append(fila)
    nombre = f"e3_{args.modelo}_resumen{sufijo}"
    resumen = guardar_csv(filas, nombre)

    print(f"E3 resumen | modelo {args.modelo}{' | con máscara de clases' if args.mascara else ''} | "
          "última ronda de cada familia")
    print(f"  {'Familia':<12}{'reconocida':>12}{'detectada':>12}{'aislado':>10}{'autoencoder':>13}"
          f"{'F1 macro':>10}{'falsos pos.':>13}")
    for fila in filas:
        print(f"  {fila['familia']:<12}{fila['recall']:>12.1%}{fila['detectado']:>12.1%}"
              f"{fila['aislado']:>10.1%}{fila['autoencoder']:>13.1%}{fila['f1_macro']:>10.3f}{fila['fpr']:>13.1%}")
    if "recall_sin_mascara" in resumen:
        print("  Sin máscara, para comparar:")
        for fila in filas:
            if "recall_sin_mascara" in fila:
                print(f"  {fila['familia']:<12}{fila['recall_sin_mascara']:>12.1%}{fila['detectado_sin_mascara']:>12.1%}")

    # --- Figura: un panel por familia ---
    n_filas = -(-len(tablas) // COLUMNAS)  # división redondeada hacia arriba
    variante = ", con máscara de clases" if args.mascara else ""
    fig, ejes = nueva_figura(n_filas, COLUMNAS,
                             f"E3 — Zero-day: cada familia vista por un solo nodo ({args.modelo}{variante})")
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
