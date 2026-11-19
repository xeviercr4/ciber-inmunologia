# Imagen de la demo: un nodo detector (o el script que envía el tráfico).
# Se construye desde la raíz del proyecto (ver docker-compose.yml).
FROM python:3.11-slim

# PyTorch solo para CPU: la imagen pesa mucho menos y para predecir no hace falta GPU
RUN pip install --no-cache-dir numpy==1.26.4 \
    && pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY src/__init__.py src/config.py src/models.py src/
COPY demo/ demo/

# -u: los mensajes aparecen al instante en "docker compose logs"
CMD ["python", "-u", "-m", "demo.nodo"]
