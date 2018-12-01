"""
Compute the eigenvalues from a Singular spectrum analysis (SSA) of the NHANES
actigraphy data, and save them to a file.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.linalg import toeplitz
import gzip

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

rm = None

out_ssa = gzip.open("actigraphy_ssa.csv.gz", "wt")

nlags = 60*12

out_ssa.write("SEQN,")
out_ssa.write(",".join(["eig%03d" % k for k in range(nlags)]) + "\n")

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

    a = acf(df.PAXINTEN, nlags=nlags)

    if np.isnan(a).any():
        eig = np.nan*np.ones(c.shape[0])
    else:
        c = toeplitz(a)
        eiv, eig = np.linalg.eig(c)
        ii = np.argsort(-eiv)
        eiv = eiv[ii]
        eig = eig[:, ii]

    nrow += 1
    if nrow % 10 == 0:
        print(nrow)

    out_ssa.write("%d," % df.SEQN.iloc[0])
    out_ssa.write(",".join([str(z) for z in eiv]) + "\n")

out_ssa.close()
