import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

DATA_PATH = "./Data/Detailed_Stocks_1/"
EXAMPLE_BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 30
BATCH_SIZE = 32
RNN_UNITS = 1024

files = os.listdir(DATA_PATH)

train_features = []
train_labels = []

for file in files:
    with open(f"{DATA_PATH}{file}", "rt") as inp:
        s = [x.split() for x in inp.readlines()]

        features = [float(x[0]) for x in s]
        labels = [float(x[1]) for x in s]
        features = features[:EXAMPLE_BATCH_SIZE * int(len(features) / EXAMPLE_BATCH_SIZE)]
        labels = labels[:EXAMPLE_BATCH_SIZE * int(len(labels) / EXAMPLE_BATCH_SIZE)]

        train_features += features
        train_labels += labels

train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_features.resize((len(train_features), 1))
train_labels.resize((len(train_labels), 1))

train_data = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(EXAMPLE_BATCH_SIZE,
                                                                                      drop_remainder=True).shuffle(
    SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(train_data)

model = keras.models.Sequential([
    tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),

    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
model.build(input_shape=(BATCH_SIZE, None, 1))

model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy)
print(model.summary())

history = model.fit(train_data, epochs=100)
