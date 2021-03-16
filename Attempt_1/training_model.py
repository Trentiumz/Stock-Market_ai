import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

DATA_PATH = "./Data/Detailed_Stocks_1/"
EXAMPLE_BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 30
BATCH_SIZE = 10
RNN_UNITS = 1024

files = os.listdir(DATA_PATH)

train_features = []

for file in files:
    with open(f"{DATA_PATH}{file}", "rt") as inp:
        s = [x.split() for x in inp.readlines()]

        features = [float(x[0]) for x in s]
        labels = [float(x[1]) for x in s]
        features = features[:EXAMPLE_BATCH_SIZE * int(len(features) / EXAMPLE_BATCH_SIZE)]
        labels = labels[:EXAMPLE_BATCH_SIZE * int(len(labels) / EXAMPLE_BATCH_SIZE)]

        features = np.array(features)
        mag = np.max(features) - np.min(features)
        features -= np.min(features)
        features *= 1/mag

        train_features += [((features[x]), (labels[x])) for x in range(len(features))]
train_features = np.array(train_features)
train_features.resize((len(train_features),2,1))

train_data = tf.data.Dataset.from_tensor_slices(train_features).map(lambda x: (x[0], x[1])).batch(EXAMPLE_BATCH_SIZE, drop_remainder=True)
train_data = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(train_data)

model = keras.models.Sequential([
    tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),

    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
model.build(input_shape=(BATCH_SIZE, None, 1))

def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy)
print(model.summary())

history = model.fit(train_data, epochs=100)
