import os

STOCK_DIR = "../Attempt_1/Data/Formatted_Stocks/"
files = os.listdir(STOCK_DIR)

toremove = []
for file in files:
    lines = None
    with open(f"{STOCK_DIR}{file}", "rt") as inp:
        s = inp.read()
        if "n" in s:
            print(file, s.count("n"))
            lines = [x for x in s.split("\n") if "n" not in x]
    if lines:
        with open(f"{STOCK_DIR}{file}", "wt") as out:
            out.write("\n".join(lines))
for i in toremove:
    os.remove(i)