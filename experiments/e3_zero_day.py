"""
E3 — Zero-day simulado.

Pregunta: si un ataque nuevo aparece en una sola organización, ¿en cuántas
rondas quedan protegidas las demás?

Se quita una familia de ataque del entrenamiento de todos los nodos menos uno.
En cada ronda se mide, sobre los flujos de esa familia en el test global:
  - reconocida:  fracción que el modelo global clasifica como esa familia.
  - detectada:   fracción que marca como ataque (de cualquier familia).
Referencias (lo que tendría un nodo que nunca vio el ataque, sin federación):
  - su modelo aislado, y
  - un autoencoder entrenado solo con su tráfico benigno.

Uso:
    python -m experiments.e3_zero_day
    python -m experiments.e3_zero_day --familia Bot --modelo transformer --rondas 10
    python -m experiments.e3_zero_day --mascara   # cada nodo entrena solo sobre las clases que conoce
    python -m experiments.e3_resumen     # compara todas las familias ya corridas

    # Guardar los pesos y una muestra de flujos para la demo con Docker (ver demo/README.md):
    python -m experiments.e3_zero_day --familia PortScan --guardar
    python -m experiments.e3_zero_day --familia PortScan --guardar --mascara
"""
import numpy as np
import pandas as pd

from experiments.comun import (
    COLOR_REFERENCIA, COLOR_TEXTO_SUAVE, dibujar_lineas, guardar_csv, guardar_figura, leer_argumentos,
    nueva_figura, preparar_ejes,
)
from src.centralized import entrenar_aislados, entrenar_autoencoder, evaluar_autoencoder
from src.config import COLUMNA_ETIQUETA, DIR_MODELOS, FAMILIAS, SEMILLA
from src.federated import ejecutar_federado
from src.partition import crear_particion, quitar_familia
from src.train_utils import cargar_datos

NODO_CON_ATAQUE = 0   # el único nodo que vio la familia nueva
NODO_SIN_ATAQUE = 1   # nodo que se usa para las referencias
NODOS_DEMO = [1, 2, 3]  # con --guardar: nodos que nunca vieron el ataque y cuyo modelo aislado se guarda
FLUJOS_DEMO = 300       # con --guardar: flujos de cada tipo (benignos y del ataque) que se guardan
UMBRALES = [0.5, 0.8, 0.9]
SERIES = {"recall": "Reconocida como esa familia", "detectado": "Marcada como ataque (cualquier familia)"}


def agregar_argumentos(parser):
    parser.add_argument("--familia", choices=FAMILIAS[1:], default="DDoS", help="familia que hace de zero-day")
    parser.add_argument("--mascara", action="store_true",
                        help="cada nodo entrena solo sobre las clases que tiene en sus datos")
    parser.add_argument("--guardar", action="store_true",
                        help="guarda los pesos y una muestra de flujos para la demo con Docker")


def guardar_flujos_demo(familia: str) -> None:
    """
    Guarda una muestra del test global (flujos que ningún modelo usó para entrenar):
    FLUJOS_DEMO benignos y FLUJOS_DEMO de la familia nueva. Es el tráfico que se envía en la demo.
    """
    _, _, X_test, y_test = cargar_datos()
    azar = np.random.default_rng(SEMILLA)
    elegidos = []
    for clase in (0, FAMILIAS.index(familia)):
        indices = np.flatnonzero(y_test == clase)
        elegidos.extend(azar.choice(indices, min(FLUJOS_DEMO, len(indices)), replace=False))
    tabla = pd.DataFrame(X_test[elegidos])
    tabla[COLUMNA_ETIQUETA] = y_test[elegidos]
    tabla.to_csv(DIR_MODELOS / f"e3_{familia}_flujos.csv", index=False)


if __name__ == "__main__":
    args = leer_argumentos("E3: zero-day simulado", agregar_argumentos)
    familia = args.familia
    variante = " | con máscara de clases" if args.mascara else ""
    print(f"E3 | modelo {args.modelo} | zero-day: {familia} (solo la ve el nodo {NODO_CON_ATAQUE}) | "
          f"{args.rondas} rondas{variante}")

    particion = quitar_familia(crear_particion("iid"), familia, NODO_CON_ATAQUE)
    X_train, y_train, _, _ = cargar_datos()
    base = f"e3_{args.modelo}_{familia}"
    sufijo = "_mascara" if args.mascara else ""

    # --- Federado ---
    federado = ejecutar_federado("fedavg", particion, args.modelo, args.rondas, mostrar=False,
                                 mascara=args.mascara,
                                 guardar_como=f"{base}_federado{sufijo}" if args.guardar else None)

    # --- Referencias: un nodo que nunca vio el ataque, sin federación ---
    # Para la demo se entrenan (y guardan) varios nodos aislados; la referencia sigue siendo el primero
    nodos = NODOS_DEMO if args.guardar else [NODO_SIN_ATAQUE]
    solo_esos_nodos = {parte: [particion[parte][n] for n in nodos] for parte in ("train", "test")}
    nombres = [f"{base}_aislado_nodo{n}" for n in nodos] if args.guardar else None
    aislado = entrenar_aislados(solo_esos_nodos, args.modelo, args.rondas, guardar_como=nombres)[0][-1]
    if args.guardar:
        guardar_flujos_demo(familia)
        print(f"  Pesos y flujos para la demo guardados en {DIR_MODELOS}")
    indices = particion["train"][NODO_SIN_ATAQUE]
    autoencoder, umbral = entrenar_autoencoder(X_train[indices][y_train[indices] == 0], args.rondas)
    metricas_ae = evaluar_autoencoder(autoencoder, umbral)

    # --- CSV: una fila por ronda y por serie ---
    # Se omite la ronda 0: el modelo sin entrenar responde al azar y marca casi todo como ataque
    # Las dos referencias se repiten en cada fila para que el CSV alcance para rehacer la gráfica
    filas = []
    for fila in federado[1:]:
        comunes = {"ronda": fila["ronda"], "f1_macro": fila["f1_macro"], "fpr": fila["fpr"],
                   "aislado": aislado[f"detectado_{familia}"], "autoencoder": metricas_ae[f"detectado_{familia}"]}
        filas.append({"serie": "recall", "valor": fila[f"recall_{familia}"], **comunes})
        filas.append({"serie": "detectado", "valor": fila[f"detectado_{familia}"], **comunes})
    nombre = base + sufijo
    tabla = guardar_csv(filas, nombre)

    # --- ¿En cuántas rondas se alcanza cada umbral? ---
    print(f"\n  Rondas hasta que el modelo global detecta {familia}:")
    for serie, titulo in SERIES.items():
        datos = tabla[tabla["serie"] == serie]
        for u in UMBRALES:
            alcanzado = datos[datos["valor"] >= u]
            cuando = f"ronda {int(alcanzado['ronda'].iloc[0])}" if len(alcanzado) else f"no se alcanza en {args.rondas} rondas"
            print(f"    {titulo:<40} >= {u:.0%}: {cuando}")
    print(f"  Modelo global en la última ronda: F1 macro {federado[-1]['f1_macro']:.3f}, "
          f"falsos positivos {federado[-1]['fpr']:.1%}")
    print(f"  Referencias de un nodo que nunca vio {familia} (sin federación):")
    print(f"    modelo aislado: la marca como ataque en {aislado[f'detectado_{familia}']:.1%} de los casos")
    print(f"    autoencoder:    la marca como anomalía en {metricas_ae[f'detectado_{familia}']:.1%} "
          f"(con {metricas_ae['fpr']:.1%} de falsos positivos)")

    # --- Gráfica ---
    fig, ejes = nueva_figura(1, 1, f"E3 — Zero-day: {familia} solo en un nodo ({args.modelo}{variante.replace(' | ', ', ')})")
    ax = ejes[0][0]
    dibujar_lineas(ax, tabla, "valor", "serie", SERIES)
    for valor, texto in [(aislado[f"detectado_{familia}"], "Nodo aislado que no la vio"),
                         (metricas_ae[f"detectado_{familia}"], "Autoencoder de ese nodo")]:
        ax.axhline(valor, color=COLOR_REFERENCIA, linewidth=1.5)
        ax.annotate(texto, (args.rondas, valor), xytext=(0, 4), textcoords="offset points",
                    ha="right", fontsize=8, color=COLOR_TEXTO_SUAVE)
    preparar_ejes(ax, f"Flujos de {familia} detectados por el modelo global", xlabel="Ronda",
                  ylabel="Fracción de los flujos")
    ax.set_ylim(-0.02, 1.02)
    fig.set_size_inches(8, 5)
    guardar_figura(fig, nombre)
