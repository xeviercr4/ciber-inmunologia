# Demo con Docker: un ataque nuevo contra tres organizaciones

Muestra el experimento E3 (zero-day) con contenedores en vez de gráficas.

- Tres contenedores (`nodo-b`, `nodo-c`, `nodo-d`) son organizaciones que **nunca vieron** la familia de ataque nueva. Cada uno corre un detector con pesos ya entrenados.
- Un cuarto contenedor (`atacante`) les envía 300 flujos de ese ataque y 300 flujos normales, tomados del conjunto de prueba.
- El envío se repite tres veces, cambiando los pesos de los nodos:
  1. cada organización con su propio modelo, sin federación;
  2. con el modelo global de la federación con promedio simple (FedAvg);
  3. con el modelo global de la federación con máscara de clases.

Acá no se entrena nada: los pesos salen de Colab.

## 1. Los pesos

Ya vienen en el repositorio, en `experiments/resultados/modelos/` (familia PortScan, modelo MLP, 20 rondas), así que no hace falta entrenar nada.

Para generarlos de nuevo, o para otra familia, corré en Colab la celda "Pesos para la demo con Docker" del cuaderno. Ejecuta:

```
python -m experiments.e3_zero_day --familia PortScan --guardar
python -m experiments.e3_zero_day --familia PortScan --guardar --mascara
```

y deja en la carpeta `modelos/` de los resultados tres archivos `aislado_nodo`, `federado`, `federado_mascara` y la muestra `e3_<familia>_flujos.csv`. Copialos a `experiments/resultados/modelos/`.

## 2. Correr la demo

Con Docker Desktop abierto, desde esta carpeta:

```
docker compose up -d --build        # levanta los tres nodos y el panel (la primera vez descarga ~1 GB)
open http://localhost:8080          # abre el panel en el navegador
docker compose down                 # apaga todo al terminar
```

En el panel: elegí una fase (cambia los pesos de los tres nodos) y apretá **Enviar tráfico**. Cada cuadrado es un flujo y se pinta según lo que respondió el contenedor de ese nodo. La tabla de abajo va guardando el resultado de cada fase.

Lo mismo sin navegador, en la terminal: `docker compose run --rm atacante` (Enter pasa a la fase siguiente).

Para ver qué registra cada nodo mientras tanto, en otra terminal:

```
docker compose logs -f
```

Otra familia u otro modelo (tienen que existir sus pesos): `FAMILIA=Web MODELO=mlp docker compose up -d`, y lo mismo delante de `docker compose run`.

## Cómo leer la salida

| Columna | Significado |
|---|---|
| Reconoce PortScan | flujos del ataque que el nodo clasifica como esa familia |
| Lo toma por otro ataque | da la alarma, pero con el nombre de otra familia |
| Lo deja pasar | flujos del ataque clasificados como tráfico normal |
| Falsas alarmas | flujos normales marcados como ataque |

## Qué es real y qué está simplificado

- Los pesos y los flujos son los del experimento: mismos datos, misma partición y misma semilla.
- Los contenedores solo predicen. El entrenamiento federado ocurrió antes, en la simulación de Flower; el paso "el servidor envía el modelo global" es el script cargando en cada nodo el archivo que produjo esa simulación.
- Los flujos ya vienen normalizados del preprocesamiento; no se captura tráfico de red real.
