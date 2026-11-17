"""
E2 — FedAvg vs. FedProx vs. cabezas locales, según cuán distintos son los nodos.

Pregunta: ¿qué estrategia aguanta mejor cuando cada nodo ve tráfico distinto?
El reparto es Dirichlet con alpha = 0.1 (nodos muy distintos), 0.5 y 1.0 (más parecidos).

Se miden dos cosas:
  - F1 macro en el test global: ¿sirve el modelo para cualquier tráfico?
  - F1 macro en el test local:  ¿le sirve a cada nodo para SU tráfico?

Uso:
    python -m experiments.e2_heterogeneidad
    python -m experiments.e2_heterogeneidad --modelo transformer --rondas 10
"""
from experiments.comun import dibujar_lineas, guardar_csv, guardar_figura, leer_argumentos, nueva_figura, preparar_ejes
from src.federated import ejecutar_federado
from src.partition import crear_particion

ALPHAS = [0.1, 0.5, 1.0]
METODOS = {"fedavg": "FedAvg", "fedprox": "FedProx", "local": "Encoder federado + cabeza local"}

if __name__ == "__main__":
    args = leer_argumentos("E2: estrategias frente a datos heterogéneos")
    print(f"E2 | modelo {args.modelo} | {args.rondas} rondas")

    filas = []
    for alpha in ALPHAS:
        particion = crear_particion("dirichlet", alpha=alpha)
        for metodo, titulo in METODOS.items():
            historial = ejecutar_federado(metodo, particion, args.modelo, args.rondas, mostrar=False)
            filas += [{"alpha": alpha, "metodo": metodo, **fila} for fila in historial]
            print(f"  alpha {alpha} | {titulo:<32} F1 global {historial[-1]['f1_macro']:.3f} | "
                  f"F1 local {historial[-1]['f1_local']:.3f}")

    nombre = f"e2_{args.modelo}"
    tabla = guardar_csv(filas, nombre)

    fig, ejes = nueva_figura(2, len(ALPHAS), f"E2 — Estrategias frente a nodos heterogéneos ({args.modelo})")
    for fila_ejes, (metrica, titulo) in zip(ejes, [("f1_macro", "F1 macro, test global"),
                                                    ("f1_local", "F1 macro, test local de cada nodo")]):
        for ax, alpha in zip(fila_ejes, ALPHAS):
            dibujar_lineas(ax, tabla[tabla["alpha"] == alpha], metrica, "metodo", METODOS)
            preparar_ejes(ax, f"{titulo} — alpha = {alpha}", xlabel="Ronda" if metrica == "f1_local" else "")
            ax.set_ylim(0, 1)
    guardar_figura(fig, nombre)
