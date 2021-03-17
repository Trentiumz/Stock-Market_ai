import os

STOCK_DIR = "./Data/Detailed_Stocks_1/"
files = os.listdir(STOCK_DIR)

toremove = []
for file in files:
    with open(f"{STOCK_DIR}{file}", "rt") as inp:
        if "n" in inp.read():
            print(file)
            toremove.append(f"{STOCK_DIR}{file}")
for i in toremove:
    os.remove(i)