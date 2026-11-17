# Ciber-Inmunología Swarm

Sistema de detección de intrusiones con aprendizaje federado. Varios nodos
(organizaciones simuladas) entrenan juntos un detector de ataques de red sin
compartir sus datos: solo comparten los pesos del modelo. La red se simula en
una sola máquina con [Flower](https://flower.ai).

## Instalación

Requiere Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verificar_entorno.py
```

## Datos

Los 8 CSV de CIC-IDS2017 (versión *MachineLearningCVE*) van en `data/raw/`. No se
versionan. Si `data/raw/` está vacío, `preprocess.py` genera un dataset sintético
con el mismo esquema para poder probar todo el flujo.

## Estructura

| Archivo | Qué hace |
|---|---|
| `src/config.py` | Rutas, semilla, dispositivo e hiperparámetros |
| `src/preprocess.py` | Limpia el dataset, agrupa las etiquetas en 7 familias, normaliza y separa train/test |
| `src/partition.py` | Reparte los datos entre nodos: IID, por familia de ataque o Dirichlet |
| `src/models.py` | MLP, FlowTransformer y autoencoder |
| `src/train_utils.py` | Carga de datos, entrenamiento, métricas y estilo de gráficas |
| `src/centralized.py` | Líneas base: modelo centralizado, modelos aislados por nodo y autoencoder |
| `src/federated.py` | Simulación federada: FedAvg, FedProx, cabezas locales, Krum, mediana y nodos maliciosos |
| `experiments/` | Un script por experimento; los resultados van a `experiments/resultados/` |

## Uso

Todos los comandos se ejecutan desde la raíz del proyecto, con el entorno activado.

```bash
# 1. Preprocesamiento (--sample 0.1 usa el 10% de los datos, para desarrollar)
python -m src.preprocess --sample 0.1

# 2. Ver cómo queda un reparto entre nodos
python -m src.partition --modo dirichlet --alpha 0.5

# 3. Líneas base
python -m src.centralized --modelo mlp

# 4. Una simulación federada suelta
python -m src.federated --estrategia fedavg --modo dirichlet --alpha 0.5

# 5. Experimentos (cada uno guarda un CSV por ronda y una gráfica PNG)
python -m experiments.e1_comparacion      # centralizado vs. aislado vs. federado
python -m experiments.e2_heterogeneidad   # FedAvg vs. FedProx vs. cabezas locales, alpha = 0.1 / 0.5 / 1.0
python -m experiments.e3_zero_day         # un ataque nuevo que solo ve un nodo
python -m experiments.e4_maliciosos       # 1 a 3 nodos maliciosos: FedAvg vs. Krum vs. mediana
```

Todos los experimentos aceptan `--modelo mlp|transformer` y `--rondas N`. El MLP
es el valor por defecto porque es rápido; el FlowTransformer tarda mucho más.

## Notas

- **Reproducibilidad:** la semilla es fija (`SEMILLA` en `config.py`). Dos corridas
  iguales producen CSV idénticos.
- **Métricas:** la principal es el F1 macro. También se reportan la tasa de falsos
  positivos (tráfico benigno marcado como ataque) y el recall por familia.
- **Dispositivo:** se usa la GPU de Apple (MPS) en Apple Silicon y CPU en el resto.
  Se puede forzar con `CIBER_DISPOSITIVO=mps` o `=cpu`.
- **Simulaciones en paralelo:** no lanzar dos simulaciones federadas a la vez; Ray
  (el motor de Flower) falla con varias instancias simultáneas.
