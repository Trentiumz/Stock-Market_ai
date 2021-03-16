import os
from copy import deepcopy

INFO_PATH = "../Data/Stock Info/"
OUTPUT_PATH = "./Data/Formatted_Stocks/"

files = [x.split(".")[0] for x in os.listdir(INFO_PATH)]
stock_info = {i:[] for i in files}

for file in files:
    with open(f"{INFO_PATH}{file}.tck", "rt") as reader:
        lines = reader.readlines()
        for line in lines:
            day, minute, price = map(float, line.split())
            stock_info[file].append(price)

for stock_name in stock_info:
    with open(f"{OUTPUT_PATH}{stock_name}.ftck", "wt") as output:
        output.write("\n".join(map(str, stock_info[stock_name])))