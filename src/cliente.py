"""Cliente Flower: treina o modelo global com a sua partição local do KDD-99."""
import flwr as fl
import tensorflow as tf

from comum import SEED, carregar, construir_modelo

_dados = {}  # cache por processo: cada processo do Ray lê o arquivo só uma vez


def _particao(cid, normalizar):
    if normalizar not in _dados:
        _dados[normalizar] = carregar(normalizar)
    d = _dados[normalizar]
    idx = d[f"idx_{cid}"]
    return d["X_train"][idx], d["y_train"][idx]


class ClienteKDD(fl.client.NumPyClient):
    def __init__(self, cid, epocas, batch, normalizar):
        self.cid, self.epocas, self.batch = cid, epocas, batch
        self.x, self.y = _particao(cid, normalizar)

    def fit(self, parametros, config):
        tf.keras.utils.set_random_seed(SEED + 1000 * config["rodada"] + self.cid)
        modelo = construir_modelo()
        modelo.set_weights(parametros)  # parte do modelo global recebido
        modelo.fit(self.x, self.y, epochs=self.epocas, batch_size=self.batch, verbose=0)
        return modelo.get_weights(), len(self.x), {}
