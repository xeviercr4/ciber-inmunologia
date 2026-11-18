"""
preprocess.py — Limpieza del dataset CIC-IDS2017 (versión MachineLearningCVE).

Qué hace, en orden:
  1. Lee los 8 CSV de data/raw (si no hay ninguno, genera un dataset sintético).
  2. Quita los espacios de los nombres de columna.
  3. Elimina filas con inf/NaN.
  4. Agrupa las etiquetas originales en 7 familias de ataque.
  5. Elimina filas duplicadas y columnas constantes.
  6. (Opcional) reduce el tráfico benigno: --sample 0.1 se queda con el 10% de
     los flujos benignos y conserva TODOS los ataques.
  7. Separa train/test de forma estratificada.
  8. Normaliza (logaritmo + estandarización) usando solo estadísticas de train.
  9. Guarda train.parquet, test.parquet y metadata.json en data/processed.

Uso:
    python -m src.preprocess               # dataset completo
    python -m src.preprocess --sample 0.1  # 10% del tráfico benigno + todos los ataques
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    COLUMNA_ETIQUETA,
    DIR_PROCESSED,
    DIR_RAW,
    FAMILIAS,
    FRACCION_TEST,
    RUTA_METADATA,
    RUTA_TEST,
    RUTA_TRAIN,
    SEMILLA,
)

# Las 78 características de CIC-IDS2017, tal como aparecen en los CSV.
# Solo se usan para que el dataset sintético tenga el mismo esquema que el real.
COLUMNAS_CIC = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min",
]

# Columnas que en el dataset real valen siempre 0 (el sintético las imita).
COLUMNAS_CONSTANTES_CIC = [
    "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
def leer_csv(directorio: Path) -> pd.DataFrame:
    """Lee todos los CSV del directorio y los une en un solo DataFrame."""
    archivos = sorted(directorio.glob("*.csv"))  # orden fijo => resultado reproducible
    if len(archivos) != 8:
        print(f"  Aviso: se esperaban 8 CSV de CIC-IDS2017 y hay {len(archivos)}.")
    partes = []
    for archivo in archivos:
        # latin-1 porque las etiquetas "Web Attack" traen un carácter que no es UTF-8
        parte = pd.read_csv(archivo, encoding="latin-1", low_memory=False)
        print(f"  {archivo.name}: {len(parte):,} filas")
        partes.append(parte)
    return pd.concat(partes, ignore_index=True)


def generar_sintetico(n_filas: int = 100_000) -> pd.DataFrame:
    """
    Genera un dataset de mentira con el mismo esquema que CIC-IDS2017.

    Sirve solo para probar el flujo completo sin los datos reales. Cada familia
    tiene su propia "firma": algunas características con valores típicos distintos.
    Además se le agregan a propósito los mismos defectos que tiene el dataset
    real (espacios en los nombres, inf/NaN, duplicados, columnas constantes y
    etiquetas raras) para ejercitar toda la limpieza.
    """
    rng = np.random.default_rng(SEMILLA)

    # Etiquetas originales de cada familia y proporción de filas de cada una
    variantes = {
        "Benign": ["BENIGN"],
        "DoS": ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"],
        "DDoS": ["DDoS"],
        "PortScan": ["PortScan"],
        "BruteForce": ["FTP-Patator", "SSH-Patator"],
        "Web": ["Web Attack \x96 Brute Force", "Web Attack \x96 XSS", "Web Attack \x96 Sql Injection"],
        "Bot": ["Bot"],
    }
    proporciones = [0.60, 0.12, 0.08, 0.10, 0.04, 0.03, 0.03]

    n_col = len(COLUMNAS_CIC)
    familia = rng.choice(len(FAMILIAS), size=n_filas, p=proporciones)

    # Firma de cada familia: desplaza 12 características elegidas al azar
    base = rng.normal(3.0, 1.0, size=n_col)
    firmas = np.tile(base, (len(FAMILIAS), 1))
    for f in range(1, len(FAMILIAS)):  # Benign (0) se queda con la base
        columnas = rng.choice(n_col, size=12, replace=False)
        firmas[f, columnas] += rng.normal(0.0, 1.0, size=12)

    # Valores positivos y con cola larga, como los contadores y tiempos de red
    valores = np.exp(firmas[familia] + rng.normal(0.0, 1.0, size=(n_filas, n_col)))
    df = pd.DataFrame(np.round(valores, 2), columns=COLUMNAS_CIC)
    df["Label"] = [rng.choice(variantes[FAMILIAS[f]]) for f in familia]

    # --- Defectos a propósito ---
    df[COLUMNAS_CONSTANTES_CIC] = 0                       # columnas constantes
    df["Fwd Header Length.1"] = df["Fwd Header Length"]   # columna repetida del original
    filas_malas = rng.choice(n_filas, size=n_filas // 500, replace=False)
    df.loc[filas_malas[::2], "Flow Bytes/s"] = np.inf     # infinitos
    df.loc[filas_malas[1::2], "Flow Packets/s"] = np.nan  # valores faltantes
    df.loc[rng.choice(n_filas, size=20, replace=False), "Label"] = "Heartbleed"  # fuera de las 7 familias
    df = pd.concat([df, df.sample(n_filas // 100, random_state=SEMILLA)], ignore_index=True)  # duplicados
    df.columns = [" " + c for c in df.columns]            # espacios como en los CSV reales
    return df


# ---------------------------------------------------------------------------
# 2. Limpieza
# ---------------------------------------------------------------------------
def agrupar_etiqueta(etiqueta: str) -> str | None:
    """Convierte una etiqueta original de CIC-IDS2017 en su familia (None = descartar)."""
    e = str(etiqueta).strip().lower()
    if e == "benign":
        return "Benign"
    if e == "ddos":
        return "DDoS"
    if e.startswith("dos"):          # DoS Hulk, GoldenEye, slowloris, Slowhttptest
        return "DoS"
    if e == "portscan":
        return "PortScan"
    if "patator" in e:               # FTP-Patator, SSH-Patator
        return "BruteForce"
    if e.startswith("web attack"):   # Brute Force, XSS, Sql Injection
        return "Web"
    if e == "bot":
        return "Bot"
    return None                      # Infiltration y Heartbleed: muy pocas filas, se descartan


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica toda la limpieza y devuelve características numéricas + columna de clase (0..6)."""
    print(f"  Filas iniciales:                {len(df):>10,}")

    # Nombres de columna sin espacios al principio ni al final
    df.columns = df.columns.str.strip()

    # El CSV original trae "Fwd Header Length" repetida; pandas la renombra con ".1"
    df = df.drop(columns=["Fwd Header Length.1"], errors="ignore")

    # Todas las características a número; lo que no se pueda convertir queda como NaN
    columnas = [c for c in df.columns if c != "Label"]
    df[columnas] = df[columnas].apply(pd.to_numeric, errors="coerce")

    # inf -> NaN, y después se eliminan las filas con algún NaN
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  Sin inf/NaN:                    {len(df):>10,}")

    # Etiquetas originales -> familias (se hace sobre los valores únicos, que son pocos)
    mapa = {etiqueta: agrupar_etiqueta(etiqueta) for etiqueta in df["Label"].unique()}
    descartadas = [e for e, familia in mapa.items() if familia is None]
    df["Label"] = df["Label"].map(mapa)
    df = df.dropna(subset=["Label"])
    print(f"  Sin etiquetas fuera de familia: {len(df):>10,}  (descartadas: {descartadas})")

    # Filas duplicadas
    df = df.drop_duplicates()
    print(f"  Sin duplicados:                 {len(df):>10,}")

    # Columnas constantes: no aportan información
    constantes = [c for c in columnas if df[c].nunique() <= 1]
    df = df.drop(columns=constantes)
    print(f"  Columnas constantes eliminadas: {len(constantes)}  -> quedan {len(columnas) - len(constantes)} características")

    # Familia (texto) -> número de clase según el orden de FAMILIAS
    df[COLUMNA_ETIQUETA] = df["Label"].map({nombre: i for i, nombre in enumerate(FAMILIAS)}).astype("int64")
    return df.drop(columns=["Label"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Normalización
# ---------------------------------------------------------------------------
def comprimir(x: np.ndarray) -> np.ndarray:
    """
    Logaritmo "con signo". Las características van de 0 a cientos de millones;
    el logaritmo comprime esos rangos. Se conserva el signo porque algunas
    columnas traen valores negativos (por ejemplo -1 como "sin dato").
    """
    return np.sign(x) * np.log1p(np.abs(x))


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def preprocesar(fraccion: float = 1.0) -> None:
    # --- 1. Carga ---
    if any(DIR_RAW.glob("*.csv")):
        print(f"Leyendo CSV de {DIR_RAW} ...")
        origen = "CIC-IDS2017"
        df = leer_csv(DIR_RAW)
    else:
        print(f"No hay CSV en {DIR_RAW}: se genera un dataset SINTÉTICO de prueba.")
        origen = "sintetico"
        df = generar_sintetico()

    # --- 2. Limpieza ---
    print("Limpiando ...")
    df = limpiar(df)

    # --- 3. Muestra: se reduce solo el tráfico benigno ---
    # El benigno es ~83% del dataset, así que recortarlo es lo que ahorra tiempo.
    # Los ataques se conservan enteros: si también se recortaran, las familias
    # raras (Web y Bot, ~2.000 filas cada una) quedarían con menos de 200 filas
    # y el modelo no llegaría a aprenderlas.
    if fraccion < 1.0:
        es_benigno = df[COLUMNA_ETIQUETA] == FAMILIAS.index("Benign")
        benigno = df[es_benigno].sample(frac=fraccion, random_state=SEMILLA)
        df = pd.concat([benigno, df[~es_benigno]]).sort_index().reset_index(drop=True)
        print(f"  Con {fraccion:.0%} del benigno:            {len(df):>10,}")

    # --- 4. Split estratificado: train y test con la misma proporción de cada familia ---
    columnas = [c for c in df.columns if c != COLUMNA_ETIQUETA]
    X = df[columnas].to_numpy(dtype=np.float64)
    y = df[COLUMNA_ETIQUETA].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FRACCION_TEST, stratify=y, random_state=SEMILLA
    )

    # --- 5. Normalización: media 0 y desvío 1 ---
    # La media y el desvío se calculan SOLO con train y se aplican a test,
    # así el modelo no "espía" información del conjunto de prueba.
    escalador = StandardScaler()
    X_train = escalador.fit_transform(comprimir(X_train))
    X_test = escalador.transform(comprimir(X_test))

    # --- 6. Guardado ---
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    for ruta, X_parte, y_parte in [(RUTA_TRAIN, X_train, y_train), (RUTA_TEST, X_test, y_test)]:
        parte = pd.DataFrame(X_parte.astype(np.float32), columns=columnas)
        parte[COLUMNA_ETIQUETA] = y_parte
        parte.to_parquet(ruta, index=False)

    # Metadatos: sirven para saber con qué datos se generaron los parquet
    def conteos(etiquetas: np.ndarray) -> dict:
        return {nombre: int((etiquetas == i).sum()) for i, nombre in enumerate(FAMILIAS)}

    metadata = {
        "origen": origen,
        "fraccion": fraccion,
        "semilla": SEMILLA,
        "familias": FAMILIAS,
        "caracteristicas": columnas,
        "filas_train": conteos(y_train),
        "filas_test": conteos(y_test),
        "media": escalador.mean_.tolist(),
        "desvio": escalador.scale_.tolist(),
    }
    RUTA_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    # --- Resumen ---
    print(f"\nListo ({origen}). {len(columnas)} características.")
    print(f"  {'Familia':<12}{'train':>10}{'test':>10}")
    for nombre in FAMILIAS:
        print(f"  {nombre:<12}{metadata['filas_train'][nombre]:>10,}{metadata['filas_test'][nombre]:>10,}")
    print(f"  {'TOTAL':<12}{len(y_train):>10,}{len(y_test):>10,}")
    print(f"Guardado en {DIR_PROCESSED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpieza de CIC-IDS2017")
    parser.add_argument("--sample", type=float, default=1.0,
                        help="fracción del tráfico benigno a usar, entre 0 y 1 (los ataques se conservan todos)")
    args = parser.parse_args()
    if not 0 < args.sample <= 1:
        parser.error("--sample debe estar entre 0 (excluido) y 1")
    preprocesar(args.sample)
