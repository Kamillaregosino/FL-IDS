"""Modelo centralizado de referência"""
import argparse
import csv
import os
import time

import tensorflow as tf

from comum import RESULTADOS, SEED, carregar, construir_modelo

p = argparse.ArgumentParser()
p.add_argument("--epocas", type=int, default=100)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--normalizar", action="store_true")
a = p.parse_args()

tf.keras.utils.set_random_seed(SEED)
d = carregar(a.normalizar)
modelo = construir_modelo()


class AcuraciaTeste(tf.keras.callbacks.Callback):
    """Mede a acurácia no teste ao fim de cada época"""

    def __init__(self):
        super().__init__()
        self.linhas = []

    def on_epoch_end(self, epoca, logs=None):
        _, acc = self.model.evaluate(d["X_test"], d["y_test"], verbose=0, batch_size=1024)
        self.linhas.append((epoca + 1, acc))
        print(f"época {epoca + 1:3d}  acurácia teste {acc:.4f}", flush=True)


cb = AcuraciaTeste()
t0 = time.time()
modelo.fit(d["X_train"], d["y_train"], epochs=a.epocas, batch_size=a.batch, verbose=0, callbacks=[cb])
print(f"tempo total: {time.time() - t0:.1f}s")

nome = f"centralizado_{'norm' if a.normalizar else 'bruto'}_B{a.batch}.csv"
os.makedirs(RESULTADOS, exist_ok=True)
with open(os.path.join(RESULTADOS, nome), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoca", "acuracia"])
    w.writerows(cb.linhas)
print("salvo em resultados/" + nome)
