"""
Append all the single-hour traffic statistics summary files to get a
traffic statistics file for one day.
"""

import numpy as np
import os
import pandas as pd

files = os.listdir("results")
files = [x for x in files if x.endswith(".csv")]

def f(x):
    y = x.split(".")[0]
    return str.isdigit(y) and len(y) == 10

files = [x for x in files if f(x)]

# Make sure the files are in temporal order
d = [int(x.split(".")[0]) for x in files]
ii = np.argsort(d)
files = [files[i] for i in ii]

adf = []
for f in files:
    df = pd.read_csv(os.path.join("results", f))
    df = df.iloc[:, 1:]
    adf.append(df)
dx = pd.concat(adf, axis=0)
dx.columns = ["Traffic", "Sources", "UDP", "TCP"]

dx["Minute"] = np.arange(dx.shape[0])
dx["Hour"] = np.floor(dx.Minute / 60)
dx.Minute = dx.Minute % 60
dx.Minute = dx.Minute.astype(np.int)
dx.Hour = dx.Hour.astype(np.int)

dx = dx[["Hour", "Minute", "Traffic", "Sources", "UDP", "TCP"]]

dx.to_csv("traffic_stats.csv", index=None)
