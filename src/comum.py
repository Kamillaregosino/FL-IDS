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



CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]


def relatorio(modelo, X, y, nome):
    """Salva precisão, recall e F1 por classe e a matriz de confusão em resultados/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    pred = np.argmax(modelo.predict(X, batch_size=1024, verbose=0), axis=1)
    txt = classification_report(y, pred, labels=range(5), target_names=CLASSES,
                                digits=4, zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(5))

    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, f"relatorio_{nome}.txt"), "w") as f:
        f.write(txt)
        f.write("\nMatriz de confusão (linhas = classe real, colunas = classe prevista)\n")
        f.write("ordem: " + ", ".join(CLASSES) + "\n" + str(cm) + "\n")
    np.savetxt(os.path.join(RESULTADOS, f"matriz_{nome}.csv"), cm, fmt="%d", delimiter=",")

    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(values_format="d", cmap="Blues")
    plt.title(nome)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTADOS, f"matriz_{nome}.png"), dpi=200)
    plt.close()
    print(txt)
    print(cm)

