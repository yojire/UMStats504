"""
Compute the intra-daily variation (IDV) values for the NHANES
actigraphy data, and save them to a file.
"""

import numpy as np
import pandas as pd
import gzip

# Range of periods to consider
L = np.linspace(5, 24*60, 200).astype(np.int)

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

rm = None

out_idv = gzip.open("actigraphy_idv.csv.gz", "wt")

nlags = 60*12

out_idv.write("SEQN,")
out_idv.write(",".join(["%03d" % k for k in L]) + "\n")

nrow = 0
while True:

    try:
        df1 = rdr.get_chunk()
    except StopIteration:
        break

    if rm is not None:
        df1 = pd.concat((rm, df1), axis=0)
        rm = None

    ii = df1.SEQN == df1.SEQN.iloc[0]
    df = df1.loc[ii, :]

    jj = df1.SEQN != df1.SEQN.iloc[0]
    rm = df1.loc[jj, :]

    y = np.asarray(df.PAXINTEN).astype(np.float64)

    iv = []
    for p in L:

        # Average over blocks of width p
        z = y[0:p*(len(y)//p)]
        z -= z.mean()
        z /= z.std()
        z = np.reshape(z, (-1, p)).mean(1)
        d = np.diff(z)

        # Calculate the idv statistic
        iv.append(np.var(d))

    iv = np.asarray(iv)

    nrow += 1
    if nrow % 10 == 0:
        print(nrow)

    out_idv.write("%d," % df.SEQN.iloc[0])
    out_idv.write(",".join([str(z) for z in iv]) + "\n")

out_idv.close()
