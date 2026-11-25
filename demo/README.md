# Demo con Docker: agentes que cooperan ante un ataque nuevo

Muestra el experimento E3 (zero-day) con contenedores. Cada organización es un **agente**: un contenedor que recibe tráfico, decide qué hacer con él y decide por su cuenta si acepta lo que le ofrecen los demás. No hay servidor que les diga qué modelo usar.

| Agente | Quién es |
|---|---|
| Organización A | La única que tiene la familia de ataque nueva en sus datos. Cuando la reconoce, avisa a sus pares y les ofrece el modelo de la federación. |
| Organizaciones B, C y D | Nunca vieron ese ataque. |
| Organización E | Maliciosa: entrenó con las etiquetas invertidas (como en E4) y ofrece ese modelo. |

Un contenedor más (el panel o `atacante`) les envía 300 flujos del ataque y 300 flujos normales, tomados del conjunto de prueba. El envío se hace en tres fases:

1. **Cada agente por su cuenta.** Nadie se comunica.
2. **Los agentes cooperan.** A reconoce el ataque y ofrece un modelo; B, C y D deciden si lo aceptan.
3. **Cooperan, y hay un agente malicioso.** E también ofrece su modelo.

Acá no se entrena nada: los pesos salen de Colab.

## Cómo se comporta un agente

Está todo en `nodo.py`. Con cada lote de tráfico:

1. **Percibe:** clasifica cada flujo con su modelo.
2. **Decide:** permite el tráfico normal, **bloquea** un ataque si la confianza del modelo es de 0,9 o más, y solo **alerta** si es menor.
3. **Coopera:** si vio 5 o más flujos de un mismo ataque en el lote y tiene un modelo para ofrecer, se lo ofrece a sus pares con un mensaje: «vi N flujos de tal familia, te ofrezco este modelo».

Cuando recibe una oferta:

- **La prueba con sus propios datos** (500 flujos de validación que solo él tiene). La acepta si el modelo ofrecido no acierta menos ni da más falsas alarmas que el suyo, con una tolerancia de 2 puntos.
- **Lleva una reputación de cada par**, que empieza en 0,5. Una oferta buena la sube 0,25; una mala la baja a la mitad. Por debajo de 0,2 ignora a ese par sin probar el modelo.

La reputación cubre lo que el agente no puede comprobar: B no tiene flujos del ataque nuevo, así que no puede verificar que el modelo de A lo detecta; solo puede verificar que no lo perjudica.

## 1. Los pesos

La demo necesita, en `experiments/resultados/modelos/`, los archivos que genera la celda "Pesos para la demo con Docker" del cuaderno de Colab. Ejecuta:

```
python -m experiments.e3_zero_day --familia PortScan --guardar
python -m experiments.e3_zero_day --familia PortScan --guardar --mascara
```

y deja en la carpeta `modelos/` de los resultados:

| Archivo | Qué es |
|---|---|
| `e3_mlp_PortScan_aislado_nodo0.pt` a `nodo3.pt` | modelo propio de A, B, C y D |
| `e3_mlp_PortScan_envenenado.pt` | modelo de E |
| `e3_mlp_PortScan_federado_mascara.pt` | modelo que ofrece A |
| `e3_mlp_PortScan_federado.pt` | federación sin máscara (opcional, ver abajo) |
| `e3_PortScan_validacion_nodo0.csv` a `nodo4.csv` | flujos de validación de cada agente |
| `e3_PortScan_flujos.csv` | tráfico que se envía |

Copialos todos juntos a `experiments/resultados/modelos/`: los flujos de validación y los de la demo se sortean en la misma corrida y no se repiten entre sí.

## 2. Correr la demo

Con Docker Desktop abierto, desde esta carpeta:

```
docker compose up -d --build        # levanta los cinco agentes y el panel (la primera vez descarga ~1 GB)
open http://localhost:8080          # abre el panel en el navegador
docker compose down                 # apaga todo al terminar
```

En el panel: elegí el ataque y la fase, y apretá **Enviar tráfico**. Cada cuadrado es un flujo y se pinta según lo que respondió ese agente. Debajo de cada agente está la confianza que tiene en sus pares, y más abajo la conversación completa: quién ofreció qué, qué decidió el otro y por qué. Cada envío arranca con los agentes como al principio (modelo propio y reputaciones en 0,5).

Lo mismo sin navegador, en la terminal: `docker compose run --rm atacante` (Enter pasa a la fase siguiente).

Para ver qué registra cada agente mientras tanto, en otra terminal:

```
docker compose logs -f
```

Variantes:

- Otro ataque: con los botones **Ataque nuevo** del panel, sin reiniciar nada. Aparecen las familias que tienen archivos en la carpeta de modelos; el repositorio trae las seis.
- Arrancar con otra familia (o usar la versión de terminal con ella): `FAMILIA=Web docker compose up -d` y `FAMILIA=Web docker compose run --rm atacante`.
- Que A ofrezca la federación **sin** máscara: `OFRECE_A='e3_mlp_<familia>_federado.pt' docker compose up -d` (con las comillas). Sirve para mostrar el resultado de E3: sin máscara, los nodos que no vieron el ataque borran lo que aprendió el que sí lo vio.

## Cómo leer la salida

| Columna | Significado |
|---|---|
| Reconoce PortScan | flujos del ataque que el agente clasifica como esa familia |
| Otro ataque / otro nombre | da la alarma, pero con el nombre de otra familia |
| Lo deja pasar | flujos del ataque clasificados como tráfico normal |
| Bloqueados | flujos del ataque que el agente bloquea (ataque con confianza de 0,9 o más) |
| Falsas alarmas | flujos normales marcados como ataque |

El resumen por fase cuenta solo a B, C y D, los agentes que no conocían el ataque.

## Qué es real y qué está simplificado

- Los pesos y los flujos son los del experimento: mismos datos, misma partición y misma semilla.
- Las decisiones de los agentes ocurren en vivo: cada contenedor prueba el modelo que le ofrecen con sus datos y lleva su propia reputación de los demás.
- Los agentes no entrenan ni fusionan pesos. El modelo que ofrece A es el modelo global que produjo antes la simulación federada de Flower; el agente lo ofrece ya calculado.
- Un mensaje lleva el nombre del archivo de pesos, no los pesos: todos los contenedores leen la misma carpeta `/modelos`.
- El agente malicioso no tiene código especial: es el mismo `nodo.py` con un modelo y unos datos de validación envenenados.
- Los flujos ya vienen normalizados del preprocesamiento; no se captura tráfico de red real.
- Los agentes reciben el lote de a uno y A lo recibe al final, para que se vea que el primer lote les pasa a los demás antes del aviso.
