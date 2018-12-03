"""
This script produces for data values that can be imported into other scripts:

df : A dataframe containing gene expression data

older, younger : subject ids (corresponding to columns of df) for older and
  younger subjects

women, men : subect ids for women and men
"""

import gzip
import pandas as pd
import numpy as np

fid = gzip.open("GDS4858_full.soft.gz", "rt")

for line in fid:

    if line.startswith("!subset_description = women"):
        line = next(fid)
        women = line.split("= ")[1].rstrip().split(",")
    elif line.startswith("!subset_description = men"):
        line = next(fid)
        men = line.split("= ")[1].rstrip().split(",")
    elif line.startswith("!subset_description = 19 to 28 years"):
        line = next(fid)
        younger = line.split("= ")[1].rstrip().split(",")
    elif line.startswith("!subset_description = 65 to 76 years"):
        line = next(fid)
        older = line.split("= ")[1].rstrip().split(",")
    elif line.startswith("!dataset_table_begin"):
        break

line = next(fid)
columns = line.rstrip().split("\t")

dat = []
for line in fid:
    if line.startswith("!deataset_table_end"):
        break
    dat.append(line.rstrip().split("\t"))

df = pd.DataFrame(dat, columns=columns)

for c in df.columns:
    if c.startswith("GSM"):
        df[c] = df[c].astype(np.float64)
