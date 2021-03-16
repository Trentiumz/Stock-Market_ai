# import tensorflow as tf
# import tensorflow.keras as keras
import numpy as np
import os

DATA_PATH = "./Data/Detailed_Stocks_1/"
files = os.listdir(DATA_PATH)



for file in files:
    with open(f"{DATA_PATH}{file}", "rt") as inp:
        features = np.array([int(x[0]) for x in inp.readlines()])
        labels = np.array([int(x[1]) for x in inp.readlines()])
