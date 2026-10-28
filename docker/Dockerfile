FROM python:3.11-slim
RUN pip install --no-cache-dir tensorflow-cpu==2.21.0 flwr==1.40.0 numpy
WORKDIR /app
COPY modelo.py servidor.py cliente.py ./
ENV TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONUNBUFFERED=1
