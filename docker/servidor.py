"""Servidor Flower dentro do contêiner: espera os 20 clientes e roda o FedAvg."""
import os

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common import ndarrays_to_parameters

from modelo import construir_modelo

N = int(os.environ["N"])
NT = int(os.environ["NT"])
RODADAS = int(os.environ["RODADAS"])

teste = np.load("/dados/teste.npz")  # o servidor só tem o conjunto de teste
avaliador = construir_modelo()


def avaliar(rodada, pesos, config):
    avaliador.set_weights(pesos)
    loss, acc = avaliador.evaluate(teste["X"], teste["y"], verbose=0, batch_size=1024)
    print(f"rodada {rodada:3d}  acurácia global {acc:.4f}", flush=True)
    return loss, {"accuracy": acc}


tf.keras.utils.set_random_seed(42)
estrategia = fl.server.strategy.FedAvg(
    fraction_fit=N / NT,
    min_fit_clients=N,
    min_available_clients=NT,
    fraction_evaluate=0.0,
    min_evaluate_clients=0,
    evaluate_fn=avaliar,
    on_fit_config_fn=lambda rodada: {"rodada": rodada},
    initial_parameters=ndarrays_to_parameters(construir_modelo().get_weights()),
)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=RODADAS),
    strategy=estrategia,
)
