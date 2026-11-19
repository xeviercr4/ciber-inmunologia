# Ciber-Inmunología Swarm — Informe de avance

17 de noviembre de 2026 · Javier

> Versión anterior del informe (primera corrida con datos reales: MLP, 5 rondas).
> La versión vigente es `informe_avance_2_corrida_20_rondas.pdf`.

## Resumen

El sistema está construido de punta a punta y ya corrió una vez con los datos reales de CIC-IDS2017, pero los resultados todavía no son concluyentes.

- **Hecho:** preprocesamiento, reparto entre nodos, tres modelos, cinco estrategias federadas, nodos maliciosos y cuatro experimentos automatizados. Todo corre en Google Colab con GPU y es reproducible con semilla fija.
- **Primera corrida real:** modelo MLP, 5 rondas y una muestra del 10% del dataset (252.074 flujos).
- **Lo que se puede afirmar:** el flujo completo funciona con datos reales, y el modelo centralizado alcanza un F1 macro de 0,924.
- **Lo que todavía no se puede afirmar:** qué estrategia federada es mejor, ni cuánto resisten las agregaciones robustas a nodos maliciosos. La muestra dejó a dos familias de ataque con menos de 200 ejemplos y eso distorsiona la métrica principal.
- **Pendiente:** repetir la corrida con el muestreo ya corregido y 20 rondas, y evaluar el modelo principal (FlowTransformer), que aún no se probó con datos reales.

## El proyecto

Ciber-Inmunología Swarm es un detector de intrusiones de red que varias organizaciones entrenan en conjunto sin compartir su tráfico: solo intercambian los pesos del modelo.

La idea viene del sistema inmune: cuando un nodo aprende a reconocer un ataque, ese conocimiento debería llegar a los demás sin que nadie exponga sus datos. Las organizaciones son nodos simulados en una sola máquina con el framework Flower.

Cada ronda de entrenamiento tiene tres pasos:

1. El servidor envía el modelo global a todos los nodos.
2. Cada nodo lo entrena con sus datos privados y devuelve los pesos nuevos.
3. El servidor combina los pesos recibidos y obtiene el siguiente modelo global.

Las preguntas que el proyecto busca responder son cuatro: cuánto se gana colaborando, qué pasa cuando cada nodo ve tráfico distinto, si un ataque nuevo visto por un solo nodo protege a los demás, y cuánto daño hace un nodo malicioso.

## Qué está construido

Las siete partes planificadas están implementadas y probadas; lo que varía es con qué datos se probó cada una.

| Componente | Qué hace | Estado |
|---|---|---|
| Preprocesamiento | Limpia los 8 CSV, agrupa las etiquetas en 7 familias (Benign, DoS, DDoS, PortScan, BruteForce, Web, Bot), normaliza y separa train/test | Probado con datos reales |
| Reparto entre nodos | IID, por familia de ataque y Dirichlet con alpha configurable | Probado con datos reales (Dirichlet e IID) |
| Modelos | MLP de línea base, FlowTransformer (cada característica es un token; 2 capas, 4 cabezas, d=64) y autoencoder para anomalías | MLP y autoencoder con datos reales; FlowTransformer solo con datos sintéticos |
| Líneas base | Modelo centralizado y un modelo aislado por nodo | Probado con datos reales |
| Federación | FedAvg, FedProx, encoder federado con cabeza local, Krum y mediana | Probado con datos reales |
| Nodos maliciosos | Entrenan con las etiquetas invertidas | Probado con datos reales |
| Experimentos | E1 a E4; cada uno guarda un CSV por ronda y una gráfica | Probados con datos reales, 5 rondas |

El dataset real tiene 2.830.743 flujos; tras eliminar valores inválidos y duplicados quedan 2.520.751, con 69 características. Las etiquetas Infiltration y Heartbleed se descartan porque no pertenecen a ninguna de las 7 familias y suman muy pocas filas.

Las métricas son el F1 macro (principal), la tasa de falsos positivos y el recall por familia. Con la misma semilla, dos corridas en CPU producen resultados idénticos.

## Resultados preliminares

Estos números salen de una sola corrida con MLP, 5 rondas y el 10% de los datos; sirven para mostrar que el sistema funciona, no como conclusiones. La sección siguiente explica por qué.

### E1: centralizado, aislado y federado

El modelo centralizado queda muy por encima; el federado y el aislado empatan. Reparto Dirichlet con alpha = 0,5 entre 5 nodos.

| Método | F1 macro | Falsos positivos |
|---|---|---|
| Centralizado | 0,924 | 0,1% |
| Federado (FedAvg) | 0,708 | 0,1% |
| Aislado (promedio de nodos) | 0,711 | 2,1% |

### E2: estrategias frente a nodos heterogéneos

FedProx queda por debajo de FedAvg en los tres casos, y la cabeza local es la mejor para el tráfico propio de cada nodo cuando los nodos son muy distintos. Cada celda muestra F1 macro en el test global / en el test local de cada nodo.

| Estrategia | alpha = 0,1 | alpha = 0,5 | alpha = 1,0 |
|---|---|---|---|
| FedAvg | 0,691 / 0,729 | 0,708 / 0,747 | 0,926 / 0,900 |
| FedProx | 0,397 / 0,456 | 0,539 / 0,542 | 0,554 / 0,545 |
| Encoder federado + cabeza local | 0,390 / 0,771 | 0,757 / 0,844 | 0,871 / 0,875 |

### E3: zero-day simulado

En 5 rondas, la federación no transmitió el conocimiento del ataque nuevo. Se quitó DDoS del entrenamiento de todos los nodos menos uno.

- El modelo global nunca clasificó un flujo DDoS como DDoS (recall 0%).
- Sí los marcó como algún tipo de ataque en cerca del 65% de los casos, lo mismo que logra un nodo aislado que nunca vio DDoS (62,9%).
- Un autoencoder entrenado solo con el tráfico benigno de ese nodo detectó el 85,5% de los flujos DDoS como anomalía, con 5,1% de falsos positivos.

### E4: nodos maliciosos

No se observa daño del ataque ni ventaja de las agregaciones robustas. Diez nodos con reparto IID; sin atacantes, FedAvg da 0,709.

| Nodos maliciosos | FedAvg | Krum | Mediana |
|---|---|---|---|
| 1 | 0,709 | 0,707 | 0,709 |
| 2 | 0,922 | 0,706 | 0,708 |
| 3 | 0,911 | 0,706 | 0,708 |

Que FedAvg mejore con 2 y 3 atacantes no tiene sentido como efecto real; es una señal del problema de medición que se describe a continuación.

### Réplica independiente

Una compañera repitió la comparación central con otro código y otra copia del dataset, y obtuvo el mismo orden entre métodos: centralizado por encima de federado, y FedProx por debajo de FedAvg.

| Método | Réplica | Este proyecto |
|---|---|---|
| Centralizado | 0,930 | 0,924 |
| Federado (FedAvg) | 0,787 | 0,708 |
| Aislado (promedio de nodos) | 0,636 | 0,711 |
| FedProx | 0,552 (mu = 0,05) | 0,539 (mu = 0,1) |

Las cifras no son comparables una a una, porque la configuración es distinta:

- **Datos:** la versión ya limpia publicada en Kaggle, con 52 características en lugar de 69.
- **Muestreo:** 150.000 flujos benignos y todos los ataques (460.555 flujos de entrenamiento), de modo que Web y Bot conservan unas 2.000 filas cada una.
- **Federación:** 5 nodos con Dirichlet alpha = 0,3, 12 rondas y 3 épocas locales por ronda.

La réplica aporta dos cosas. Que FedProx empeore con dos implementaciones y dos valores de mu indica que el problema está en el parámetro y no en un error de código. Y su FedAvg más alto es compatible con que el muestreo sea la causa del estancamiento en 0,71, aunque no lo demuestra porque cambian varias cosas a la vez.

Su FedAvg también se estanca, en 0,79 desde la ronda 2 y lejos del 0,93 centralizado. La réplica no mide recall por familia, así que no se sabe qué clases falla.

## Qué falta validar

El problema principal es que la muestra del 10% dejó a las familias Web y Bot con muy pocos ejemplos, y eso vuelve inestable el F1 macro.

| Familia | Flujos de entrenamiento | Flujos de prueba |
|---|---|---|
| Benign | 167.605 | 41.901 |
| DoS | 15.499 | 3.875 |
| DDoS | 10.241 | 2.560 |
| PortScan | 7.255 | 1.814 |
| BruteForce | 732 | 183 |
| Web | 171 | 43 |
| Bot | 156 | 39 |

El F1 macro promedia las 7 familias con el mismo peso. Por eso se mueve en escalones: acertar 5 familias y fallar 2 por completo da cerca de 0,71, que es el valor que se repite en E1, E2 y E4. La hipótesis es que las dos familias falladas son Web y Bot; falta confirmarla con el recall por familia, que está guardado en los CSV de resultados.

De ahí se desprenden las demás limitaciones:

- **E4 no es interpretable todavía.** El salto de FedAvg a 0,92 con atacantes se explica mejor porque esa corrida llegó a aprender una familia rara que por un efecto del ataque.
- **Cinco rondas son pocas.** Varias curvas seguían subiendo al terminar; el diseño prevé 20.
- **FedProx parece mal calibrado.** Con el coeficiente proximal actual (mu = 0,1) rinde bastante menos que FedAvg; la réplica independiente muestra lo mismo con mu = 0,05. Falta probar valores menores antes de concluir algo sobre la estrategia.
- **Una sola semilla.** No hay medida de variabilidad, así que diferencias chicas entre métodos no son significativas.
- **El FlowTransformer no se evaluó con datos reales.** Es el modelo principal del proyecto y solo se probó en corridas cortas con datos sintéticos.
- **La normalización es global.** Se calcula con todo el conjunto de entrenamiento antes de repartirlo; en una federación real cada nodo solo conocería sus propios datos.
- **El dataset proviene de una copia pública en Hugging Face**, no de la descarga oficial del Canadian Institute for Cybersecurity. El formato coincide, pero no se verificó que el contenido sea idéntico.

## Próximas etapas

Las mejoras están en orden de prioridad: las tres primeras son necesarias para que cualquier resultado sea válido.

| # | Mejora | Qué valida |
|---|---|---|
| 1 | Confirmar con el recall por familia qué familias falla cada método | Si la hipótesis de Web y Bot es correcta |
| 2 | Repetir la corrida con el muestreo nuevo, ya implementado: reduce solo el tráfico benigno y conserva todos los ataques (Web pasa de 171 a unos 1.700 flujos de entrenamiento) | Que el F1 macro deje de depender de 40 ejemplos de prueba |
| 3 | Repetir E1 a E4 con 20 rondas | Que las curvas lleguen a estabilizarse |
| 4 | Repetir E2 con mu = 0,01, que ya es el valor por defecto, y probar 0,001 | Si el mal resultado es de la estrategia o del parámetro |
| 5 | Correr los experimentos con el FlowTransformer | El aporte del modelo principal frente al MLP |
| 6 | Repetir cada experimento con 3 semillas y reportar promedio y desvío | Qué diferencias entre métodos son reales |
| 7 | Repetir E3 con más rondas y con otras familias como zero-day | Si la federación llega a propagar el ataque nuevo, y en cuántas rondas |
| 8 | Evaluar una pérdida con pesos por clase | Si mejora la detección de las familias minoritarias |

Dos resultados de esta corrida merecen seguimiento aunque sean preliminares. El autoencoder detectó el 85,5% de un ataque que nunca vio, lo que sugiere combinarlo con el clasificador federado para el caso zero-day. Y la cabeza local superó a FedAvg en el tráfico propio de cada nodo cuando los datos son heterogéneos, que es el escenario realista entre organizaciones distintas.

## Cómo reproducirlo

El código es público y la corrida completa se repite desde un notebook, sin instalar nada.

- **Repositorio:** [github.com/xeviercr4/ciber-inmunologia](https://github.com/xeviercr4/ciber-inmunologia)
- **Notebook de Colab:** [abrir en Colab](https://colab.research.google.com/github/xeviercr4/ciber-inmunologia/blob/main/colab.ipynb). Descarga el dataset, corre los cuatro experimentos con GPU y guarda los resultados en Google Drive.
- **Dataset:** CIC-IDS2017, versión MachineLearningCVE, del [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html). Los archivos usados vienen de [una copia en Hugging Face](https://huggingface.co/datasets/c01dsnap/CIC-IDS2017).
- **Entorno de la corrida:** Google Colab con GPU Tesla T4, Python 3.13, PyTorch 2.14 y Flower 1.43.
