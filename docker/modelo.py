import tensorflow as tf


def construir_modelo():
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(27,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m
