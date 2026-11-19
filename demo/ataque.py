"""
ataque.py — Envía tráfico a los nodos de la demo y muestra qué detecta cada uno.

Usa flujos del conjunto de prueba (ningún modelo los vio al entrenar): una
parte es tráfico normal y la otra pertenece a la familia de ataque "nueva",
la que solo una organización (la A, que no participa de la demo) había visto.

Repite el mismo envío en tres momentos:
    1. Cada organización usa su propio modelo, entrenado solo con sus datos.
    2. Reciben el modelo global de la federación con promedio simple (FedAvg).
    3. Reciben el modelo global de la federación con máscara de clases.

Uso (desde la carpeta demo/, con los nodos ya levantados):
    docker compose run --rm atacante
    docker compose run --rm atacante python -u -m demo.ataque --sin-pausas
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from src.config import COLUMNA_ETIQUETA, FAMILIAS

DIR_MODELOS = Path(os.environ.get("DIR_MODELOS", "/modelos"))
FAMILIA = os.environ.get("FAMILIA", "PortScan")
MODELO = os.environ.get("MODELO", "mlp")
NODOS = os.environ.get("NODOS", "nodo-b,nodo-c,nodo-d").split(",")
PUERTO = 8000

# (título, archivo de pesos que recibe cada nodo; None = se queda con el que ya tiene)
FASES = [
    ("Cada organización por su cuenta (sin federación)", None),
    ("Federación con promedio simple (FedAvg)", f"e3_{MODELO}_{FAMILIA}_federado.pt"),
    ("Federación con máscara de clases", f"e3_{MODELO}_{FAMILIA}_federado_mascara.pt"),
]


def pedir(nodo: str, ruta: str, datos: dict | None = None) -> dict:
    """Pedido HTTP a un nodo (GET si no hay datos, POST si los hay). Devuelve la respuesta."""
    cuerpo = json.dumps(datos).encode() if datos is not None else None
    pedido = urllib.request.Request(f"http://{nodo}:{PUERTO}{ruta}", data=cuerpo,
                                    headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(pedido, timeout=30) as respuesta:
            return json.loads(respuesta.read())
    except urllib.error.HTTPError as error:
        sys.exit(f"El nodo {nodo} respondió con un error: {json.loads(error.read())['error']}")


def esperar_nodos() -> None:
    """Los contenedores tardan unos segundos en cargar PyTorch: se espera a que respondan."""
    for nodo in NODOS:
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://{nodo}:{PUERTO}/estado", timeout=2)
                break
            except OSError:
                time.sleep(1)
        else:
            sys.exit(f"El nodo {nodo} no responde. ¿Corriste 'docker compose up -d'?")


def leer_flujos() -> tuple[list[list[float]], list[list[float]]]:
    """Lee la muestra guardada por E3. Devuelve (flujos benignos, flujos del ataque)."""
    ruta = DIR_MODELOS / f"e3_{FAMILIA}_flujos.csv"
    if not ruta.exists():
        sys.exit(f"Falta {ruta.name}. Se genera con: python -m experiments.e3_zero_day --familia {FAMILIA} --guardar")
    benignos, ataque = [], []
    with open(ruta) as archivo:
        for fila in csv.DictReader(archivo):
            clase = int(fila.pop(COLUMNA_ETIQUETA))
            (benignos if clase == 0 else ataque).append([float(v) for v in fila.values()])
    return benignos, ataque


def enviar(benignos, ataque) -> list[dict]:
    """Envía el tráfico a cada nodo, imprime qué hizo con él y devuelve los conteos."""
    print(f"  {'Nodo':<16}{'Reconoce ' + FAMILIA:>20}{'Lo toma por otro ataque':>26}"
          f"{'Lo deja pasar':>16}{'Falsas alarmas':>17}")
    resultados = []
    for nodo in NODOS:
        nombre = pedir(nodo, "/estado")["nombre"]
        sobre_ataque = pedir(nodo, "/analizar", {"flujos": ataque})["familias"]
        sobre_benigno = pedir(nodo, "/analizar", {"flujos": benignos})["familias"]
        r = {"reconocidos": sobre_ataque.count(FAMILIA), "pasaron": sobre_ataque.count("Benign"),
             "falsas": len(sobre_benigno) - sobre_benigno.count("Benign")}
        r["otros"] = len(ataque) - r["reconocidos"] - r["pasaron"]
        resultados.append(r)
        print(f"  {nombre:<16}{porcentaje(r['reconocidos'], len(ataque)):>20}"
              f"{porcentaje(r['otros'], len(ataque)):>26}{porcentaje(r['pasaron'], len(ataque)):>16}"
              f"{porcentaje(r['falsas'], len(benignos)):>17}")
    return resultados


def porcentaje(parte: int, total: int) -> str:
    return f"{parte} ({parte / total:.0%})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: tráfico de un ataque nuevo contra los nodos")
    parser.add_argument("--sin-pausas", action="store_true", help="no espera Enter entre una fase y la siguiente")
    args = parser.parse_args()
    assert FAMILIA in FAMILIAS[1:], f"Familia desconocida: {FAMILIA}"

    esperar_nodos()
    benignos, ataque = leer_flujos()
    print(f"\nAtaque nuevo: {FAMILIA}. Solo la organización A lo tiene en sus datos; B, C y D nunca lo vieron.")
    print(f"A cada nodo se le envían {len(ataque)} flujos de {FAMILIA} y {len(benignos)} flujos normales "
          f"(del conjunto de prueba).")

    resumen = []
    for numero, (titulo, pesos) in enumerate(FASES, start=1):
        if numero > 1 and not args.sin_pausas:
            input("\nEnter para continuar...")
        print(f"\n=== Fase {numero}: {titulo} ===")
        if pesos:
            for nodo in NODOS:
                pedir(nodo, "/modelo", {"archivo": pesos})
            print(f"  El servidor envió el modelo global a los {len(NODOS)} nodos ({pesos}).")
        resultados = enviar(benignos, ataque)
        resumen.append((titulo, sum(r["reconocidos"] for r in resultados), sum(r["pasaron"] for r in resultados)))

    total = len(ataque) * len(NODOS)
    print(f"\n=== Resumen: {total} flujos de {FAMILIA} entre los {len(NODOS)} nodos ===")
    for titulo, reconocidos, pasaron in resumen:
        print(f"  {titulo:<50} reconocidos {porcentaje(reconocidos, total):>12} | "
              f"pasaron como normales {porcentaje(pasaron, total):>12}")
