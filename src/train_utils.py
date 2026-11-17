"""
train_utils.py — Funciones compartidas de datos, entrenamiento y evaluación.

Las usan tanto el entrenamiento centralizado como los nodos federados, así
todos los métodos se entrenan y se miden exactamente de la misma forma.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from src.config import (
    COLUMNA_ETIQUETA,
    DIR_MODELOS,
    DISPOSITIVO,
    FAMILIAS,
    RUTA_TEST,
    RUTA_TRAIN,
    SEMILLA,
    TAM_LOTE,
    TASA_APRENDIZAJE,
)
from src.models import MLP, Autoencoder, FlowTransformer


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def cargar_datos() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lee los parquet generados por preprocess.py.
    Devuelve (X_train, y_train, X_test, y_test) como arreglos de NumPy.
    lru_cache hace que el disco se lea una sola vez por proceso.
    """
    if not RUTA_TRAIN.exists():
        raise FileNotFoundError("Faltan los datos procesados. Corré primero: python -m src.preprocess --sample 0.1")
    partes = []
    for ruta in (RUTA_TRAIN, RUTA_TEST):
        df = pd.read_parquet(ruta)
        # copy=True: PyTorch necesita arreglos propios (no vistas de solo lectura de pandas)
        partes.append(df.drop(columns=COLUMNA_ETIQUETA).to_numpy(dtype=np.float32, copy=True))
        partes.append(df[COLUMNA_ETIQUETA].to_numpy(dtype=np.int64, copy=True))
    return tuple(partes)


# ---------------------------------------------------------------------------
# Modelos y pesos
# ---------------------------------------------------------------------------
def crear_modelo(nombre: str, semilla: int = SEMILLA) -> nn.Module:
    """
    Crea un modelo nuevo ("mlp", "transformer" o "autoencoder").
    La semilla fija los pesos iniciales: todos los métodos arrancan del mismo punto.
    """
    n_caracteristicas = cargar_datos()[0].shape[1]
    torch.manual_seed(semilla)
    if nombre == "mlp":
        return MLP(n_caracteristicas)
    if nombre == "transformer":
        return FlowTransformer(n_caracteristicas)
    if nombre == "autoencoder":
        return Autoencoder(n_caracteristicas)
    raise ValueError(f"Modelo desconocido: {nombre}")


def obtener_pesos(modulo: nn.Module) -> list[np.ndarray]:
    """Pesos del módulo como lista de arreglos NumPy (el formato que viaja por la red)."""
    # .copy(): sin copia, el arreglo compartiría memoria con el modelo y cambiaría al entrenar
    return [valor.cpu().numpy().copy() for valor in modulo.state_dict().values()]


def cargar_pesos(modulo: nn.Module, pesos: list[np.ndarray]) -> None:
    """Operación inversa: copia una lista de arreglos NumPy dentro del módulo."""
    claves = modulo.state_dict().keys()
    modulo.load_state_dict({clave: torch.tensor(valor) for clave, valor in zip(claves, pesos)})


def guardar_modelo(modulo: nn.Module, nombre_modelo: str, archivo: str, parte: str = "completo") -> None:
    """
    Guarda los pesos en experiments/resultados/modelos/<archivo>.pt.
    `parte` indica qué contiene: "completo" (encoder + cabeza) o "encoder"
    (en la variante personalizada el servidor solo conoce el encoder).
    """
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    pesos = {clave: valor.cpu() for clave, valor in modulo.state_dict().items()}
    torch.save({"modelo": nombre_modelo, "parte": parte, "pesos": pesos}, DIR_MODELOS / f"{archivo}.pt")


def cargar_modelo(ruta) -> nn.Module:
    """
    Reconstruye un modelo a partir de un archivo .pt guardado con guardar_modelo.
    Sirve para traer a esta máquina un modelo entrenado en otra (p. ej. en Colab).
    """
    guardado = torch.load(ruta, map_location="cpu")  # map_location: funciona aunque se haya entrenado en GPU
    modelo = crear_modelo(guardado["modelo"])
    destino = modelo.encoder if guardado["parte"] == "encoder" else modelo
    destino.load_state_dict(guardado["pesos"])
    if guardado["parte"] == "encoder":
        print("Aviso: el archivo solo contiene el encoder; la cabeza de clasificación queda sin entrenar.")
    return modelo


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def entrenar(modelo: nn.Module, X: np.ndarray, y: np.ndarray | None, epocas: int,
             semilla: int = SEMILLA, mu: float = 0.0) -> None:
    """
    Entrena el modelo `epocas` pasadas sobre (X, y).

    - Si `y` es None se entrena como autoencoder (reconstruir la entrada).
    - Si mu > 0 se agrega el término proximal de FedProx, que penaliza
      alejarse de los pesos con los que el modelo llegó (los globales):
          pérdida += (mu / 2) * ||pesos - pesos_globales||²
    """
    if len(X) == 0:
        return
    modelo.to(DISPOSITIVO).train()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    X_t = torch.from_numpy(X).to(DISPOSITIVO)
    y_t = None if y is None else torch.from_numpy(y).to(DISPOSITIVO)
    generador = torch.Generator().manual_seed(semilla)  # orden de los lotes reproducible
    torch.manual_seed(semilla)                          # dropout reproducible
    globales = [p.detach().clone() for p in modelo.parameters()] if mu > 0 else None

    for _ in range(epocas):
        orden = torch.randperm(len(X_t), generator=generador).to(DISPOSITIVO)
        for inicio in range(0, len(orden), TAM_LOTE):
            lote = orden[inicio:inicio + TAM_LOTE]
            salida = modelo(X_t[lote])
            if y_t is None:
                perdida = F.mse_loss(salida, X_t[lote])
            else:
                perdida = F.cross_entropy(salida, y_t[lote])
            if globales is not None:
                distancia = sum(((p - g) ** 2).sum() for p, g in zip(modelo.parameters(), globales))
                perdida = perdida + (mu / 2) * distancia
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
@torch.no_grad()
def predecir(modelo: nn.Module, X: np.ndarray) -> np.ndarray:
    """Familia predicha para cada flujo (se procesa de a lotes grandes)."""
    modelo.to(DISPOSITIVO).eval()
    partes = []
    for inicio in range(0, len(X), 2048):
        lote = torch.from_numpy(X[inicio:inicio + 2048]).to(DISPOSITIVO)
        partes.append(modelo(lote).argmax(dim=1).cpu().numpy())
    return np.concatenate(partes) if partes else np.empty(0, dtype=np.int64)


def calcular_metricas(y: np.ndarray, pred: np.ndarray) -> dict:
    """
    Métricas a partir de las etiquetas reales y las predichas.

    - f1_macro:  promedio del F1 de cada familia presente en `y`. Todas las
                 familias pesan igual, aunque tengan pocas filas (métrica principal).
    - exactitud: fracción de flujos bien clasificados.
    - fpr:       tasa de falsos positivos = tráfico benigno marcado como ataque.
    - recall_X:    de los flujos de la familia X, fracción clasificada como X.
    - detectado_X: de los flujos de la familia X, fracción marcada como ataque
                   (de cualquier familia). Sirve para ataques nunca vistos.
    """
    if len(y) == 0:
        return {}
    metricas = {
        "f1_macro": float(f1_score(y, pred, labels=np.unique(y), average="macro", zero_division=0)),
        "exactitud": float((y == pred).mean()),
        "fpr": float((pred[y == 0] != 0).mean()) if (y == 0).any() else float("nan"),
    }
    for clase, nombre in enumerate(FAMILIAS):
        es_clase = y == clase
        metricas[f"recall_{nombre}"] = float((pred[es_clase] == clase).mean()) if es_clase.any() else float("nan")
        if clase > 0:
            metricas[f"detectado_{nombre}"] = float((pred[es_clase] != 0).mean()) if es_clase.any() else float("nan")
    return metricas


def evaluar(modelo: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    """Predice sobre X y devuelve las métricas."""
    return calcular_metricas(y, predecir(modelo, X))


def entrenar_y_evaluar(modelo: nn.Module, X: np.ndarray, y: np.ndarray, epocas: int,
                       X_local: np.ndarray | None = None, y_local: np.ndarray | None = None) -> list[dict]:
    """
    Entrena época por época y, después de cada una, mide sobre el test global.
    Si se pasa un test local, agrega "f1_local". Devuelve una fila de métricas por época.
    """
    _, _, X_test, y_test = cargar_datos()
    historial = []
    for epoca in range(1, epocas + 1):
        entrenar(modelo, X, y, epocas=1, semilla=SEMILLA + epoca)
        fila = {"ronda": epoca, **evaluar(modelo, X_test, y_test)}
        if X_local is not None:
            fila["f1_local"] = evaluar(modelo, X_local, y_local).get("f1_macro", float("nan"))
        historial.append(fila)
    return historial


# ---------------------------------------------------------------------------
# Gráficas (estilo común a todo el proyecto)
# ---------------------------------------------------------------------------
COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]  # azul, naranja, verde agua: distinguibles con daltonismo
COLOR_REFERENCIA = "#8a8983"                 # gris para líneas de referencia
COLOR_FONDO = "#fcfcfb"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SUAVE = "#52514e"
COLOR_REJILLA = "#e6e5e0"


def preparar_ejes(ax, titulo: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Aplica el estilo común a unos ejes: fondo claro, rejilla tenue y sin bordes de más."""
    ax.set_facecolor(COLOR_FONDO)
    ax.grid(True, color=COLOR_REJILLA, linewidth=1)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COLOR_REJILLA)
    ax.tick_params(colors=COLOR_TEXTO_SUAVE, length=0)
    ax.set_title(titulo, color=COLOR_TEXTO, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=COLOR_TEXTO_SUAVE)
    ax.set_ylabel(ylabel, color=COLOR_TEXTO_SUAVE)


def graficar_matriz(y: np.ndarray, pred: np.ndarray, titulo: str, ruta) -> None:
    """
    Guarda la matriz de confusión como PNG. Cada fila es una familia real y
    muestra qué porcentaje de sus flujos fue a parar a cada familia predicha.
    """
    import matplotlib
    matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
    import matplotlib.pyplot as plt

    matriz = confusion_matrix(y, pred, labels=range(len(FAMILIAS)))
    porcentaje = matriz / np.maximum(matriz.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=COLOR_FONDO)
    imagen = ax.imshow(porcentaje, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(FAMILIAS)):
        for j in range(len(FAMILIAS)):
            ax.text(j, i, f"{porcentaje[i, j]:.0%}\n{matriz[i, j]:,}", ha="center", va="center", fontsize=7.5,
                    color="white" if porcentaje[i, j] > 0.5 else COLOR_TEXTO)
    ax.set_xticks(range(len(FAMILIAS)), FAMILIAS, rotation=45, ha="right")
    ax.set_yticks(range(len(FAMILIAS)), FAMILIAS)
    ax.tick_params(colors=COLOR_TEXTO_SUAVE, length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xlabel("Familia predicha", color=COLOR_TEXTO_SUAVE)
    ax.set_ylabel("Familia real", color=COLOR_TEXTO_SUAVE)
    ax.set_title(titulo, color=COLOR_TEXTO, fontsize=11, loc="left")
    fig.colorbar(imagen, ax=ax, fraction=0.046, label="Fracción de la familia real").outline.set_visible(False)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
