"""
Import this module to get the data (summarized to counts by crime type x day).

Putting this into a module means that we don't have to reload the data
every time we run one of the analysis scripts.
"""

import pandas as pd

cdat = pd.read_csv("cdat.csv.gz")
