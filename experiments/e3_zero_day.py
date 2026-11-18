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
"""
from experiments.comun import (
    COLOR_REFERENCIA, COLOR_TEXTO_SUAVE, dibujar_lineas, guardar_csv, guardar_figura, leer_argumentos,
    nueva_figura, preparar_ejes,
)
from src.centralized import entrenar_aislados, entrenar_autoencoder, evaluar_autoencoder
from src.config import FAMILIAS
from src.federated import ejecutar_federado
from src.partition import crear_particion, quitar_familia
from src.train_utils import cargar_datos

NODO_CON_ATAQUE = 0   # el único nodo que vio la familia nueva
NODO_SIN_ATAQUE = 1   # nodo que se usa para las referencias
UMBRALES = [0.5, 0.8, 0.9]
SERIES = {"recall": "Reconocida como esa familia", "detectado": "Marcada como ataque (cualquier familia)"}


def agregar_argumentos(parser):
    parser.add_argument("--familia", choices=FAMILIAS[1:], default="DDoS", help="familia que hace de zero-day")
    parser.add_argument("--mascara", action="store_true",
                        help="cada nodo entrena solo sobre las clases que tiene en sus datos")


if __name__ == "__main__":
    args = leer_argumentos("E3: zero-day simulado", agregar_argumentos)
    familia = args.familia
    variante = " | con máscara de clases" if args.mascara else ""
    print(f"E3 | modelo {args.modelo} | zero-day: {familia} (solo la ve el nodo {NODO_CON_ATAQUE}) | "
          f"{args.rondas} rondas{variante}")

    particion = quitar_familia(crear_particion("iid"), familia, NODO_CON_ATAQUE)
    X_train, y_train, _, _ = cargar_datos()

    # --- Federado ---
    federado = ejecutar_federado("fedavg", particion, args.modelo, args.rondas, mostrar=False,
                                 mascara=args.mascara)

    # --- Referencias: un nodo que nunca vio el ataque, sin federación ---
    solo_ese_nodo = {parte: [particion[parte][NODO_SIN_ATAQUE]] for parte in ("train", "test")}
    aislado = entrenar_aislados(solo_ese_nodo, args.modelo, args.rondas)[0][-1]
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
    nombre = f"e3_{args.modelo}_{familia}" + ("_mascara" if args.mascara else "")
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
