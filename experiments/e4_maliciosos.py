"""
E4 — Nodos maliciosos: FedAvg vs. Krum vs. mediana.

Pregunta: ¿qué pasa si algunos nodos intentan sabotear el modelo, y cuánto
ayudan las agregaciones robustas?

De 1 a 3 nodos entrenan con las etiquetas invertidas. Se usan 10 nodos (y no 5)
porque Krum necesita n >= 2f + 3 nodos para tolerar f maliciosos: con f = 3
hacen falta al menos 9. El reparto es IID para aislar el efecto del ataque.
La línea gris es FedAvg sin ningún atacante (lo mejor que se puede esperar).

Uso:
    python -m experiments.e4_maliciosos
    python -m experiments.e4_maliciosos --modelo transformer --rondas 10
"""
from experiments.comun import dibujar_lineas, guardar_csv, guardar_figura, leer_argumentos, nueva_figura, preparar_ejes
from src.federated import ejecutar_federado
from src.partition import crear_particion

NUM_NODOS_E4 = 10
MALICIOSOS = [1, 2, 3]
ESTRATEGIAS = {"fedavg": "FedAvg", "krum": "Krum", "mediana": "Mediana"}
REFERENCIA = "FedAvg sin atacantes"

if __name__ == "__main__":
    args = leer_argumentos("E4: nodos maliciosos y agregación robusta")
    print(f"E4 | modelo {args.modelo} | {NUM_NODOS_E4} nodos | {args.rondas} rondas")

    particion = crear_particion("iid", num_nodos=NUM_NODOS_E4)
    sin_ataque = ejecutar_federado("fedavg", particion, args.modelo, args.rondas, mostrar=False)
    print(f"  {REFERENCIA:<28} F1 macro {sin_ataque[-1]['f1_macro']:.3f} | falsos positivos {sin_ataque[-1]['fpr']:.3f}")

    filas = []
    for cantidad in MALICIOSOS:
        filas += [{"maliciosos": cantidad, "estrategia": REFERENCIA, **fila} for fila in sin_ataque]
        for estrategia, titulo in ESTRATEGIAS.items():
            historial = ejecutar_federado(estrategia, particion, args.modelo, args.rondas,
                                          maliciosos=tuple(range(cantidad)), mostrar=False)
            filas += [{"maliciosos": cantidad, "estrategia": estrategia, **fila} for fila in historial]
            print(f"  maliciosos: {cantidad} | {titulo:<8} F1 macro {historial[-1]['f1_macro']:.3f} | "
                  f"falsos positivos {historial[-1]['fpr']:.3f}")

    nombre = f"e4_{args.modelo}"
    tabla = guardar_csv(filas, nombre)

    fig, ejes = nueva_figura(2, len(MALICIOSOS), f"E4 — Nodos con etiquetas invertidas, {NUM_NODOS_E4} nodos ({args.modelo})")
    for fila_ejes, (metrica, titulo) in zip(ejes, [("f1_macro", "F1 macro"), ("fpr", "Falsos positivos")]):
        for ax, cantidad in zip(fila_ejes, MALICIOSOS):
            dibujar_lineas(ax, tabla[tabla["maliciosos"] == cantidad], metrica, "estrategia", ESTRATEGIAS,
                           referencia=REFERENCIA)
            preparar_ejes(ax, f"{titulo} — {cantidad} malicioso{'s' if cantidad > 1 else ''}",
                          xlabel="Ronda" if metrica == "fpr" else "")
        if metrica == "f1_macro":
            fila_ejes[0].set_ylim(0, 1)
    guardar_figura(fig, nombre)
