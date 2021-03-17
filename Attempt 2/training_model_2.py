import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

DATA_PATH = "./Data/Detailed_Stocks_1/"
EXAMPLE_BATCH_SIZE = 90
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1
BATCH_SIZE = 100

files = os.listdir(DATA_PATH)

train_features = []
train_labels = []


def complete_features(features, labels):
    global train_features
    global train_labels
    features = features[:EXAMPLE_BATCH_SIZE * (len(features) // EXAMPLE_BATCH_SIZE)]
    labels = labels[:EXAMPLE_BATCH_SIZE * (len(labels) // EXAMPLE_BATCH_SIZE)]

    features -= np.min(features)
    features *= 1 / (np.max(features) - np.min(features))

    train_features += list(features)
    train_labels += list(labels)


for file in files:
    with open(f"{DATA_PATH}{file}", "rt") as inp:
        s = [x.split() for x in inp.readlines() if "n" not in x]

        original_features = np.array([float(x[0]) for x in s])
        original_labels = np.array([float(x[1]) for x in s])

        step = 0.2
        times = 5
        for i in range(times):
            start = int(EXAMPLE_BATCH_SIZE * step * i)
            complete_features(original_features[start:], original_labels[start:])

train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_features.resize((len(train_features) // EXAMPLE_BATCH_SIZE, EXAMPLE_BATCH_SIZE))
train_labels.resize((len(train_labels) // EXAMPLE_BATCH_SIZE, EXAMPLE_BATCH_SIZE))
train_labels = train_labels[:, EXAMPLE_BATCH_SIZE - 1:]

print(train_features.shape)
print(train_labels.shape)

model = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(90,)),
    keras.layers.Dense(64, activation="sigmoid"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

test_features, train_features = train_features[:int(TEST_RATIO * len(train_features))], train_features[int(TEST_RATIO * len(train_features)):]
test_labels, train_labels = train_labels[:int(TEST_RATIO * len(train_labels))], train_labels[int(TEST_RATIO * len(train_labels)):]

model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"])
print(model.summary())

history = model.fit(train_features, train_labels, batch_size=BATCH_SIZE, validation_split=0.1, epochs=10000)
print(model.evaluate(train_features, train_labels, verbose=2))
