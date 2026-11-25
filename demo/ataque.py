"""
ataque.py — Envía tráfico a los agentes de la demo y muestra qué hace cada uno.

Usa flujos del conjunto de prueba (ningún modelo los vio al entrenar): una
parte es tráfico normal y la otra pertenece a la familia de ataque "nueva",
la que solo la organización A había visto.

Repite el mismo envío en tres fases:
    1. Cada agente por su cuenta: nadie se comunica.
    2. Los agentes cooperan: el que reconoce el ataque ofrece un modelo a sus pares
       y cada uno decide si lo acepta.
    3. Lo mismo, con un agente malicioso que ofrece un modelo envenenado.

Este script solo envía tráfico y prepara cada fase. Qué modelo usa cada agente
lo deciden los agentes entre ellos (ver nodo.py).

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
from collections import Counter
from pathlib import Path

from src.config import COLUMNA_ETIQUETA, FAMILIAS

DIR_MODELOS = Path(os.environ.get("DIR_MODELOS", "/modelos"))
FAMILIA = os.environ.get("FAMILIA", "PortScan")
NODOS = os.environ.get("NODOS", "nodo-a,nodo-b,nodo-c,nodo-d,nodo-e").split(",")
ESPECIALISTA = os.environ.get("ESPECIALISTA", "nodo-a")  # el único agente que conoce el ataque nuevo
MALICIOSO = os.environ.get("MALICIOSO", "nodo-e")        # el agente con el modelo envenenado
PUERTO = 8000
LOTE = 15  # flujos por envío: los agentes conversan entre un lote y el siguiente

FASES = [
    "Cada agente por su cuenta",
    "Los agentes cooperan",
    "Cooperan, y hay un agente malicioso",
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


def familias_disponibles() -> list[str]:
    """Familias de ataque que tienen su muestra de flujos en la carpeta de modelos."""
    return [familia for familia in FAMILIAS[1:] if (DIR_MODELOS / f"e3_{familia}_flujos.csv").exists()]


def leer_flujos(familia: str = FAMILIA) -> tuple[list[list[float]], list[list[float]]]:
    """Lee la muestra guardada por E3. Devuelve (flujos benignos, flujos del ataque)."""
    ruta = DIR_MODELOS / f"e3_{familia}_flujos.csv"
    if not ruta.exists():
        sys.exit(f"Falta {ruta.name}. Se genera con: python -m experiments.e3_zero_day --familia {familia} --guardar")
    benignos, ataque = [], []
    with open(ruta) as archivo:
        for fila in csv.DictReader(archivo):
            clase = int(fila.pop(COLUMNA_ETIQUETA))
            (benignos if clase == 0 else ataque).append([float(v) for v in fila.values()])
    return benignos, ataque


def nodos_de_fase(numero: int) -> list[str]:
    """Agentes que reciben tráfico en esa fase: el malicioso solo aparece en la 3."""
    return [nodo for nodo in NODOS if numero == 3 or nodo != MALICIOSO]


def preparar_fase(numero: int, familia: str = FAMILIA) -> None:
    """Cada agente vuelve a su modelo propio para esa familia. Desde la fase 2 pueden hablar entre ellos."""
    for nodo in NODOS:
        pedir(nodo, "/reiniciar", {"comunicar": numero >= 2 and nodo in nodos_de_fase(numero),
                                   "familia": familia})


def enviar_lote(nodos: list[str], ataque: list, benignos: list, desde: int, hasta: int) -> dict:
    """
    Envía a cada agente los flujos [desde, hasta) de cada tipo. Devuelve {nodo: {"ataque": ..., "normal": ...}}.
    Se recorre la lista al revés para que el agente que conoce el ataque (el primero)
    reciba su lote al final: así el aviso a sus pares llega después de que el primer lote ya les pasó.
    """
    return {nodo: {"ataque": pedir(nodo, "/analizar", {"flujos": ataque[desde:hasta]}),
                   "normal": pedir(nodo, "/analizar", {"flujos": benignos[desde:hasta]})}
            for nodo in reversed(nodos)}


def conversacion(estados: list[dict]) -> list[dict]:
    """Todos los mensajes que recibieron los agentes, en orden de llegada."""
    return sorted((mensaje for estado in estados for mensaje in estado["mensajes"]), key=lambda m: m["hora"])


def contar(sobre_ataque: dict, sobre_benigno: dict) -> dict:
    """Resume lo que hizo un agente con un lote."""
    familias = sobre_ataque["familias"]
    r = {"reconocidos": familias.count(FAMILIA), "pasaron": familias.count("Benign"),
         "bloqueados": sobre_ataque["acciones"].count("bloquear"),
         "falsas": len(sobre_benigno["familias"]) - sobre_benigno["familias"].count("Benign")}
    r["otros"] = len(familias) - r["reconocidos"] - r["pasaron"]
    return r


def enviar(nodos: list[str], benignos: list, ataque: list) -> dict:
    """Envía todo el tráfico de a lotes, imprime qué hizo cada agente y devuelve los conteos por nodo."""
    totales = {nodo: Counter() for nodo in nodos}
    for desde in range(0, max(len(ataque), len(benignos)), LOTE):
        for nodo, respuestas in enviar_lote(nodos, ataque, benignos, desde, desde + LOTE).items():
            totales[nodo].update(contar(respuestas["ataque"], respuestas["normal"]))

    estados = [pedir(nodo, "/estado") for nodo in nodos]
    print(f"  {'Agente':<16}{'Reconoce ' + FAMILIA:>20}{'Otro ataque':>14}{'Lo deja pasar':>16}"
          f"{'Bloqueados':>14}{'Falsas alarmas':>17}   Modelo al terminar")
    for nodo, estado in zip(nodos, estados):
        r = totales[nodo]
        print(f"  {estado['nombre']:<16}{porcentaje(r['reconocidos'], len(ataque)):>20}"
              f"{porcentaje(r['otros'], len(ataque)):>14}{porcentaje(r['pasaron'], len(ataque)):>16}"
              f"{porcentaje(r['bloqueados'], len(ataque)):>14}{porcentaje(r['falsas'], len(benignos)):>17}"
              f"   {estado['pesos']}")
    mensajes = conversacion(estados)
    if mensajes:
        print("  Conversación entre agentes:")
        for m in mensajes:
            print(f"    {m['de']} -> {m['para']}: \"vi {m['cantidad']} flujos de {m['familia']}, te ofrezco "
                  f"{m['archivo']}\" => {m['decision']} ({m['motivo']}; reputación {m['reputacion']:.2f})")
    return totales


def porcentaje(parte: int, total: int) -> str:
    return f"{parte} ({parte / total:.0%})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: tráfico de un ataque nuevo contra los agentes")
    parser.add_argument("--sin-pausas", action="store_true", help="no espera Enter entre una fase y la siguiente")
    args = parser.parse_args()
    assert FAMILIA in FAMILIAS[1:], f"Familia desconocida: {FAMILIA}"

    esperar_nodos()
    benignos, ataque = leer_flujos()
    sin_el_ataque = [nodo for nodo in NODOS if nodo not in (ESPECIALISTA, MALICIOSO)]
    print(f"\nAtaque nuevo: {FAMILIA}. Solo la organización A lo tiene en sus datos.")
    print(f"A cada agente se le envían {len(ataque)} flujos de {FAMILIA} y {len(benignos)} flujos normales "
          f"(del conjunto de prueba), de a {LOTE}.")

    resumen = []
    for numero, titulo in enumerate(FASES, start=1):
        if numero > 1 and not args.sin_pausas:
            input("\nEnter para continuar...")
        print(f"\n=== Fase {numero}: {titulo} ===")
        preparar_fase(numero)
        totales = enviar(nodos_de_fase(numero), benignos, ataque)
        resumen.append((titulo, sum((totales[nodo] for nodo in sin_el_ataque), Counter())))

    total = len(ataque) * len(sin_el_ataque)
    print(f"\n=== Resumen: {total} flujos de {FAMILIA} entre los {len(sin_el_ataque)} agentes que no lo conocían ===")
    for titulo, r in resumen:
        print(f"  {titulo:<38} reconocidos {porcentaje(r['reconocidos'], total):>12} | "
              f"pasaron como normales {porcentaje(r['pasaron'], total):>12} | "
              f"falsas alarmas {porcentaje(r['falsas'], len(benignos) * len(sin_el_ataque)):>10}")
