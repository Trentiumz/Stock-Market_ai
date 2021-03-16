import os

STOCK_DIR = "./Data/Detailed_Stocks_1/"
files = os.listdir(STOCK_DIR)

for file in files:
    with open(f"{STOCK_DIR}{file}", "rt") as inp:
        to_write = [x for x in inp.readlines() if x.split()[0].isnumeric() and x.split()[1].isnumeric()]
    with open(f"{STOCK_DIR}{file}", "wt") as out:
        out.write("\n".join(to_write))