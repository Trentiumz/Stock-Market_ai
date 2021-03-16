import os
from copy import deepcopy

INFO_PATH = "../Stock AI/Data/Stock Info/"
OUTPUT_PATH = "./Data/Formatted_Stocks/"

files = os.listdir(INFO_PATH)
days_template = [[] for _ in range(500)]
stock_info = {i:deepcopy(days_template) for i in files}

for file in files:
    with open(f"{INFO_PATH}{file}", "rt") as reader:
        lines = reader.readlines()
        for line in lines:
            day, minute, price = map(float, line.split())
            stock_info[file][int(day)].append(price)

for stock_name in stock_info:
    for day in range(len(stock_info[stock_name])):
        if len(stock_info[stock_name][day]) == 0:
            continue
        with open(f"{OUTPUT_PATH}{stock_name}_{day}.ftck", "wt") as output:
            output.write("\n".join(map(str, stock_info[stock_name][day])))