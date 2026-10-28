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


_modelo = {}  # modelo reaproveitado entre rodadas: treinar um modelo novo a cada rodada vaza memória


class Cliente(fl.client.NumPyClient):
    def fit(self, parametros, config):
        tf.keras.utils.set_random_seed(42 + 1000 * config["rodada"] + CID)
        novo = construir_modelo()  # consome o gerador aleatório como antes, para o treino sair idêntico
        modelo = _modelo.setdefault("m", novo)
        modelo.set_weights(parametros)
        for v in modelo.optimizer.variables:  # Adam do zero, como num modelo novo
            if "learning_rate" not in v.path:
                v.assign(tf.zeros_like(v))
        modelo.fit(x, y, epochs=E, batch_size=B, verbose=0)
        return modelo.get_weights(), len(x), {}


fl.client.start_client(server_address="servidor:8080", client=Cliente().to_client())
