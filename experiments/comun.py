"""
comun.py — Utilidades compartidas por los cuatro experimentos:
argumentos de línea de comandos, guardado de CSV y estilo de las gráficas.
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.config import DIR_RESULTADOS, RONDAS
from src.train_utils import COLOR_FONDO, COLOR_REFERENCIA, COLOR_TEXTO, COLOR_TEXTO_SUAVE, COLORES, preparar_ejes


def leer_argumentos(descripcion: str, extra=None) -> argparse.Namespace:
    """Argumentos comunes: --modelo y --rondas. `extra(parser)` permite agregar otros."""
    parser = argparse.ArgumentParser(description=descripcion)
    parser.add_argument("--modelo", choices=["mlp", "transformer"], default="mlp",
                        help="mlp es rápido; transformer es el modelo principal pero tarda mucho más")
    parser.add_argument("--rondas", type=int, default=RONDAS)
    if extra:
        extra(parser)
    return parser.parse_args()


def guardar_csv(filas: list[dict], nombre: str) -> pd.DataFrame:
    """Guarda las métricas por ronda en experiments/resultados/<nombre>.csv y las devuelve como tabla."""
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(DIR_RESULTADOS / f"{nombre}.csv", index=False)
    return tabla


def nueva_figura(filas: int, columnas: int, titulo: str):
    """Crea una figura con una grilla de paneles que comparten ejes."""
    fig, ejes = plt.subplots(filas, columnas, figsize=(4.6 * columnas, 3.4 * filas + 1.0),
                             sharex=True, sharey="row", squeeze=False, facecolor=COLOR_FONDO)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, color=COLOR_TEXTO)
    return fig, ejes


def dibujar_lineas(ax, tabla: pd.DataFrame, metrica: str, columna_serie: str, series: dict,
                   referencia: str | None = None) -> None:
    """
    Dibuja una línea por serie: `metrica` en función de la ronda.
    `series` es {valor en la tabla: nombre a mostrar}. El color depende del
    orden en `series`, así cada método tiene siempre el mismo color.
    `referencia` es una serie que se dibuja en gris, como punto de comparación.
    """
    # La ronda 0 (modelo sin entrenar) queda en el CSV pero no se dibuja: responde al azar
    # y sus valores extremos aplastarían la escala del resto de la curva.
    tabla = tabla[tabla["ronda"] >= 1]
    for color, (valor, nombre) in zip(COLORES, series.items()):
        datos = tabla[tabla[columna_serie] == valor]
        ax.plot(datos["ronda"], datos[metrica], color=color, linewidth=2, label=nombre)
    if referencia is not None:
        datos = tabla[tabla[columna_serie] == referencia]
        ax.plot(datos["ronda"], datos[metrica], color=COLOR_REFERENCIA, linewidth=1.5, label=referencia)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # las rondas son números enteros


def guardar_figura(fig, nombre: str) -> None:
    """Pone una sola leyenda debajo de los paneles (sin repetir nombres) y guarda el PNG."""
    etiquetas = {}
    for ax in fig.axes:
        for linea, nombre_serie in zip(*ax.get_legend_handles_labels()):
            etiquetas.setdefault(nombre_serie, linea)
    fig.legend(etiquetas.values(), etiquetas.keys(), loc="lower center", ncol=len(etiquetas),
               frameon=False, labelcolor=COLOR_TEXTO_SUAVE)
    margen_leyenda = 0.45 / fig.get_figheight()  # deja ~0.45 pulgadas abajo para la leyenda
    fig.tight_layout(rect=(0, margen_leyenda, 1, 1))
    fig.savefig(DIR_RESULTADOS / f"{nombre}.png", dpi=150)
    plt.close(fig)
    print(f"Guardado: {nombre}.csv y {nombre}.png en {DIR_RESULTADOS}")


__all__ = ["leer_argumentos", "guardar_csv", "nueva_figura", "dibujar_lineas", "guardar_figura",
           "preparar_ejes", "COLOR_REFERENCIA", "COLOR_TEXTO_SUAVE", "COLORES"]
