"""Partes compartilhadas da replicação do artigo (Damaceno et al., 2023).

- dados: KDD-99 com as 27 propriedades (dataset/__processed_kdd.csv)
- divisão: 75% treino / 25% teste; treino embaralhado e dividido em NT=20 clientes iguais
- modelo: MLP 27 -> 16 (ReLU) -> 5 (softmax), Adam, sparse categorical crossentropy
"""
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
CSV = os.path.join(RAIZ, "dataset", "__processed_kdd.csv")
RESULTADOS = os.path.join(RAIZ, "resultados")
SEED = 42
NT = 20  # número total de clientes (artigo: Nt = 20)


def _cache(normalizar):
    return os.path.join(RAIZ, "cache", f"dados_{'norm' if normalizar else 'bruto'}.npz")


def preparar(normalizar):
    """Lê o CSV, converte para números e salva a divisão treino/teste e as 20 partições."""
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(CSV, low_memory=False)
    # mesma codificação do SCRIPT_PRINCIPAL_FL.py
    df["_Attack Type"] = df["Attack Type"].map({
        "normal": 0, "dos": 1, "probe": 2, "r2l": 3, "u2r": 4
    }).astype(np.int32)
    df["protocol_type"] = df["protocol_type"].map({"tcp": 0, "udp": 1, "icmp": 2})
    df["flag"] = df["flag"].map({
        "SF": 0, "S1": 1, "REJ": 2, "S2": 3, "S0": 4,
        "S3": 5, "RSTO": 6, "RSTR": 7, "RSTOS0": 8, "OTH": 9, "SH": 10
    })
    X = df.drop(columns=["Attack Type", "_Attack Type"]).astype(np.float32).to_numpy()
    y = df["_Attack Type"].to_numpy()
    assert X.shape[1] == 27

    if normalizar:
        # mesmo pré-processamento usado pelos autores no código DDoS (DDos/client.py)
        X = preprocessing.normalize(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    # "as amostras foram embaralhadas e divididas em 20 conjuntos com a mesma quantidade"
    ordem = np.random.default_rng(SEED).permutation(len(X_train))
    partes = np.array_split(ordem, NT)

    os.makedirs(os.path.dirname(_cache(normalizar)), exist_ok=True)
    np.savez_compressed(
        _cache(normalizar), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        **{f"idx_{i}": p for i, p in enumerate(partes)},
    )


def carregar(normalizar):
    if not os.path.exists(_cache(normalizar)):
        preparar(normalizar)
    return dict(np.load(_cache(normalizar)))


def construir_modelo():
    import tensorflow as tf

    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(27,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])
    m.compile(
        optimizer=tf.keras.optimizers.Adam(),  # taxa padrão 0.001 (o artigo não informa)
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m
