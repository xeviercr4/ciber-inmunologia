# Ciber-Inmunología Swarm

Sistema de detección de intrusiones con aprendizaje federado. Varios nodos
(organizaciones simuladas) entrenan juntos un detector de ataques de red sin
compartir sus datos: solo comparten los pesos del modelo. La red se simula en
una sola máquina.

## Instalación

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verificar_entorno.py
```

## Datos

Los CSV de CIC-IDS2017 (MachineLearningCVE) van en `data/raw/`. No se versionan.

## Estado

Proyecto en construcción: por ahora están la estructura, las dependencias y la
configuración base.
