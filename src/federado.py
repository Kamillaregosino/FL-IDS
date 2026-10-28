"""Aprendizado federado com Flower + FedAvg.

Nt = 20 clientes no total; a cada rodada o servidor sorteia N deles; cada um treina E épocas
locais com batch B a partir do modelo global; o servidor agrega com FedAvg (média ponderada
pelo número de amostras) e avalia o modelo global no conjunto de teste.

Exemplo:  python src/federado.py --N 5 --E 1 --B 16 --rodadas 100 --normalizar
"""
import argparse
import csv
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import flwr as fl
import tensorflow as tf
from flwr.common import ndarrays_to_parameters

from comum import AQUI, NT, RESULTADOS, SEED, carregar, construir_modelo

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=5, help="clientes por rodada")
p.add_argument("--E", type=int, default=1, help="épocas locais")
p.add_argument("--B", type=int, default=16, help="batch size")
p.add_argument("--rodadas", type=int, default=100)
p.add_argument("--normalizar", action="store_true")
p.add_argument("--paralelos", type=int, default=5, help="clientes treinando ao mesmo tempo")
a = p.parse_args()


def client_fn(context):
    # roda dentro de um processo do Ray; importa aqui para não serializar os dados
    from cliente import ClienteKDD
    return ClienteKDD(int(context.node_config["partition-id"]), a.E, a.B, a.normalizar).to_client()


d = carregar(a.normalizar)
avaliador = construir_modelo()


def avaliar(rodada, pesos, config):
    avaliador.set_weights(pesos)
    loss, acc = avaliador.evaluate(d["X_test"], d["y_test"], verbose=0, batch_size=1024)
    print(f"rodada {rodada:3d}  acurácia global {acc:.4f}", flush=True)
    return loss, {"accuracy": acc}


tf.keras.utils.set_random_seed(SEED)  # mesmo modelo inicial e mesmo sorteio de clientes
estrategia = fl.server.strategy.FedAvg(
    fraction_fit=a.N / NT,
    min_fit_clients=a.N,
    min_available_clients=NT,
    fraction_evaluate=0.0,  # avaliação feita no servidor, com o conjunto de teste
    min_evaluate_clients=0,
    evaluate_fn=avaliar,
    on_fit_config_fn=lambda rodada: {"rodada": rodada},
    initial_parameters=ndarrays_to_parameters(construir_modelo().get_weights()),
)

t0 = time.time()
historico = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NT,
    config=fl.server.ServerConfig(num_rounds=a.rodadas),
    strategy=estrategia,
    client_resources={"num_cpus": os.cpu_count() / a.paralelos},
    ray_init_args={
        "include_dashboard": False,
        "num_cpus": os.cpu_count(),
        "runtime_env": {"env_vars": {
            "PYTHONPATH": AQUI,
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        }},
    },
)
print(f"tempo total: {time.time() - t0:.1f}s")

nome = f"fl_{'norm' if a.normalizar else 'bruto'}_N{a.N}_E{a.E}_B{a.B}.csv"
os.makedirs(RESULTADOS, exist_ok=True)
with open(os.path.join(RESULTADOS, nome), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rodada", "acuracia"])
    w.writerows(historico.metrics_centralized["accuracy"])
print("salvo em resultados/" + nome)
