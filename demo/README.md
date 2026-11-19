# Demo con Docker: un ataque nuevo contra tres organizaciones

Muestra el experimento E3 (zero-day) con contenedores en vez de gráficas.

- Tres contenedores (`nodo-b`, `nodo-c`, `nodo-d`) son organizaciones que **nunca vieron** la familia de ataque nueva. Cada uno corre un detector con pesos ya entrenados.
- Un cuarto contenedor (`atacante`) les envía 300 flujos de ese ataque y 300 flujos normales, tomados del conjunto de prueba.
- El envío se repite tres veces, cambiando los pesos de los nodos:
  1. cada organización con su propio modelo, sin federación;
  2. con el modelo global de la federación con promedio simple (FedAvg);
  3. con el modelo global de la federación con máscara de clases.

Acá no se entrena nada: los pesos salen de Colab.

## 1. Generar los pesos (en Colab, una sola vez)

Corré la celda "Pesos para la demo con Docker" del cuaderno. Ejecuta:

```
python -m experiments.e3_zero_day --familia PortScan --guardar
python -m experiments.e3_zero_day --familia PortScan --guardar --mascara
```

Después descargá el zip de resultados y copiá estos seis archivos de su carpeta `modelos/` a `experiments/resultados/modelos/`:

```
e3_mlp_PortScan_aislado_nodo1.pt     e3_mlp_PortScan_federado.pt
e3_mlp_PortScan_aislado_nodo2.pt     e3_mlp_PortScan_federado_mascara.pt
e3_mlp_PortScan_aislado_nodo3.pt     e3_PortScan_flujos.csv
```

## 2. Correr la demo

Con Docker Desktop abierto, desde esta carpeta:

```
docker compose up -d --build        # levanta los tres nodos (la primera vez descarga ~1 GB)
docker compose run --rm atacante    # envía el tráfico; Enter pasa a la fase siguiente
docker compose down                 # apaga todo al terminar
```

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
