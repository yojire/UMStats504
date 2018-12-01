"""
Small demonstration using intra-daily variation (IDV) with the NHANES
actigraphy data.
"""

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Range of periods to consider
L = np.linspace(5, 24*60, 200).astype(np.int)

pdf = PdfPages("idv.pdf")

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

rm = None

idx = 0
while True:

    df1 = rdr.get_chunk()
    if rm is not None:
        df1 = pd.concat((rm, df1), axis=0)
        rm = None

    ii = df1.SEQN == df1.SEQN.iloc[0]
    df = df1.loc[ii, :]

    jj = df1.SEQN != df1.SEQN.iloc[0]
    rm = df1.loc[jj, :]

    print(df.SEQN.value_counts(), "\n\n")

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
        iv.append([np.log(p), np.var(d)])

    iv = np.asarray(iv)

    plt.clf()
    plt.grid(True)
    plt.plot(iv[:, 0], iv[:, 1])
    plt.xlabel("log(period)", size=16)
    plt.ylabel("Intra-daily variation", size=15)
    pdf.savefig()

    idx += 1
    if idx > 10:
        break

pdf.close()
