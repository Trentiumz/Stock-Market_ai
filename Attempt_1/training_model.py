import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

DATA_PATH = "./Data/Detailed_Stocks_1/"
EXAMPLE_BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 30
BATCH_SIZE = 20
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
train_features.resize((int(len(train_features) / EXAMPLE_BATCH_SIZE), EXAMPLE_BATCH_SIZE))
train_labels.resize((int(len(train_labels) / EXAMPLE_BATCH_SIZE), EXAMPLE_BATCH_SIZE))
for i in range(len(train_features)):
    train_features[i,:] -= np.min(train_features[i,:])
    train_features[i,:] *= 1/(np.max(train_features[i]) - np.min(train_features[i]))

train_features.resize((len(train_features), EXAMPLE_BATCH_SIZE, 1))
train_labels.resize((len(train_labels), EXAMPLE_BATCH_SIZE, 1))

print(train_features.shape)

model = keras.models.Sequential([
    tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True, stateful=True, recurrent_initializer="glorot_uniform"),

    keras.layers.Dense(1, activation=keras.activations.sigmoid)
])
model.build(input_shape=(BATCH_SIZE, None, 1))

def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy)
print(model.summary())


history = model.fit(train_features[:BATCH_SIZE], train_labels[:BATCH_SIZE], epochs=100)
