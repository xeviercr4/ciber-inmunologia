"""
E1 — Centralizado vs. aislado vs. federado.

Pregunta: ¿cuánto se gana colaborando sin compartir datos?
  - Centralizado: todos los datos juntos (techo ideal, pero viola la privacidad).
  - Aislado:      cada nodo por su cuenta (promedio de los 5 nodos).
  - Federado:     FedAvg; los nodos solo comparten pesos.
Los tres hacen la misma cantidad de pasadas sobre los datos (1 ronda = 1 época).

Uso:
    python -m experiments.e1_comparacion
    python -m experiments.e1_comparacion --modelo transformer --rondas 10 --modo familia
"""
from experiments.comun import dibujar_lineas, guardar_csv, guardar_figura, leer_argumentos, nueva_figura, preparar_ejes
from src.centralized import entrenar_aislados, entrenar_centralizado, promediar_nodos
from src.federated import ejecutar_federado
from src.partition import crear_particion, resumen

METODOS = {"centralizado": "Centralizado", "aislado": "Aislado (promedio de nodos)", "federado": "Federado (FedAvg)"}

if __name__ == "__main__":
    args = leer_argumentos("E1: centralizado vs. aislado vs. federado", lambda p: p.add_argument(
        "--modo", choices=["iid", "familia", "dirichlet"], default="dirichlet",
        help="reparto entre nodos; 'familia' = caso extremo, cada nodo ve solo algunas familias de ataque"))

    particion = crear_particion(args.modo)
    print(f"E1 | modelo {args.modelo} | reparto {args.modo} | {args.rondas} rondas\n{resumen(particion)}\n")

    print("  Entrenando centralizado ...")
    _, centralizado = entrenar_centralizado(args.modelo, args.rondas)
    print("  Entrenando un modelo aislado por nodo ...")
    aislado = promediar_nodos(entrenar_aislados(particion, args.modelo, args.rondas))
    print("  Entrenando federado ...")
    federado = ejecutar_federado("fedavg", particion, args.modelo, args.rondas, mostrar=False)

    filas = [{"metodo": metodo, **fila}
             for metodo, historial in [("centralizado", centralizado), ("aislado", aislado), ("federado", federado)]
             for fila in historial]
    nombre = f"e1_{args.modelo}"
    tabla = guardar_csv(filas, nombre)

    print("\n  Resultado en la última ronda:")
    for metodo, titulo in METODOS.items():
        ultima = tabla[tabla["metodo"] == metodo].iloc[-1]
        print(f"    {titulo:<30} F1 macro {ultima['f1_macro']:.3f} | falsos positivos {ultima['fpr']:.3f}")

    fig, ejes = nueva_figura(1, 2, f"E1 — Colaborar sin compartir datos ({args.modelo}, reparto {args.modo})")
    for ax, (metrica, titulo) in zip(ejes[0], [("f1_macro", "F1 macro (test global)"),
                                               ("fpr", "Tasa de falsos positivos")]):
        dibujar_lineas(ax, tabla, metrica, "metodo", METODOS)
        preparar_ejes(ax, titulo, xlabel="Ronda / época")
    ejes[0][0].set_ylim(0, 1)
    guardar_figura(fig, nombre)
