"""Cliente Flower dentro do contêiner: só enxerga /dados/particao.npz (os próprios dados)."""
import os

import flwr as fl
import numpy as np
import tensorflow as tf

from modelo import construir_modelo

CID = int(os.environ["CID"])
E = int(os.environ["E"])
B = int(os.environ["B"])

dados = np.load("/dados/particao.npz")
x, y = dados["X"], dados["y"]


class Cliente(fl.client.NumPyClient):
    def fit(self, parametros, config):
        tf.keras.backend.clear_session()  # libera o modelo da rodada anterior; sem isso a memória cresce a cada rodada
        tf.keras.utils.set_random_seed(42 + 1000 * config["rodada"] + CID)
        modelo = construir_modelo()
        modelo.set_weights(parametros)
        modelo.fit(x, y, epochs=E, batch_size=B, verbose=0)
        return modelo.get_weights(), len(x), {}


fl.client.start_client(server_address="servidor:8080", client=Cliente().to_client())
