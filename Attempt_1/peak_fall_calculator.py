import os
import numpy as np
import matplotlib.pyplot as plt

INFO_PATH = "./Data/Formatted_Stocks/"
OUTPUT_PATH = "./Data/Detailed_Stocks_1/"
files = os.listdir(INFO_PATH)
SEGMENT_LENGTH = 7

info_holder = {}

def plot(x):
    plt.figure()
    plt.scatter([x[0] for x in total_significant if x[1]], [prices[x[0]] for x in total_significant if x[1]],
                c="#dc143c")
    plt.scatter([x[0] for x in total_significant if not x[1]], [prices[x[0]] for x in total_significant if not x[1]],
                c="#00FF00")
    plt.plot(prices)
    plt.show()

# Calculating peaks and falls and putting it into info_holder in ((prices), (peaks), (falls))
for file in files:
    # Get the prices from the file
    with open(f"{INFO_PATH}{file}", "rt") as reader:
        prices = np.array([float(x) for x in reader.readlines()])
        prices = prices[:SEGMENT_LENGTH * int(len(prices) / SEGMENT_LENGTH)]
    peaks = []
    falls = []

    # Segment the data into SEGMENT_LENGTH lengths
    segmented = prices.reshape((-1,SEGMENT_LENGTH))
    segmented_max = [np.max(x) for x in segmented]
    segmented_min = [np.min(x) for x in segmented]

    # If there are two peaks next to each other, we only take one
    for i in range(len(segmented_max)):
        cur = segmented_max[i]
        lef = 0 if i == 0 else segmented_max[i-1]
        rig = 0 if i == len(segmented_max) - 1 else segmented_max[i + 1]
        if cur > lef and cur > rig or cur > lef and cur == rig:
            peaks.append(np.where(prices[i * SEGMENT_LENGTH:(i + 1)*SEGMENT_LENGTH]==cur)[0][0] + i * SEGMENT_LENGTH)

    # If there are two falls next to each other, we only take one
    for i in range(len(segmented_min)):
        cur = segmented_min[i]
        lef = 99999999999 if i == 0 else segmented_min[i-1]
        rig = 99999999999 if i == len(segmented_min) - 1 else segmented_min[i + 1]
        if cur < lef and cur < rig or cur < lef and cur == rig:
            falls.append(np.where(prices[i * SEGMENT_LENGTH:(i + 1)*SEGMENT_LENGTH]==cur)[0][0] + i * SEGMENT_LENGTH)

    # We find all of the significant points, and if both are peaks/falls, then we insert another one inside
    total_significant = [(x, True) for x in peaks] + [(x, False) for x in falls]
    total_significant = list(sorted(total_significant, key=lambda x: x[0]))
    for i in range(1, len(total_significant)):
        if total_significant[i][1] == total_significant[i - 1][1] == True:
            falls.append(np.argmin(prices[total_significant[i - 1][0]: total_significant[i][0]]) + total_significant[i - 1][0])
        if total_significant[i][1] == total_significant[i - 1][1] == False:
            peaks.append(np.argmax(prices[total_significant[i - 1][0]: total_significant[i][0]]) + total_significant[i - 1][0])

    # We add another element so that we fill total_significant
    total_significant = [(x, 1) for x in peaks] + [(x, 0) for x in falls]
    total_significant = list(sorted(total_significant, key=lambda x: x[0]))
    if len(total_significant) == 0:
        continue
    if total_significant[-1][0] != len(prices) - 1:
        total_significant.append((len(prices) - 1, not total_significant[-1][1]))

    # We fill it up with data detailing whether or not the next one is a rise or fall
    nexts = []
    last = -1
    for index, val in total_significant:
        nexts += [val] * (index - last)
        last = index

    # Export this data into a dictionary
    info_holder[file] = (tuple(prices), tuple(nexts))


for file in info_holder:
    with open(f"{OUTPUT_PATH}{file}", "wt") as output:
        output.write("\n".join([f"{info_holder[file][0][x]} {info_holder[file][1][x]}" for x in range(len(info_holder[file][0]))]))