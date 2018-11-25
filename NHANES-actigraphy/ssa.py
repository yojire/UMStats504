"""
Singular spectrum analysis (SSA) of the NHANES actigraphy data
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.linalg import toeplitz
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

pdf = PdfPages("ssa.pdf")

rdr = pd.read_csv("paxraw.csv.gz", error_bad_lines=False, chunksize=24*60*7)

rm = None

eivals = []
eigals = []
c0 = 0

while True:

    df1 = rdr.get_chunk()
    if rm is not None:
        df1 = pd.concat((rm, df1), axis=0)
        rm = None

    ii = df1.SEQN == df1.SEQN.iloc[0]
    df = df1.loc[ii, :]

    jj = df1.SEQN != df1.SEQN.iloc[0]
    rm = df1.loc[jj, :]

    a = acf(df.PAXINTEN, nlags=60*12)
    c = toeplitz(a)
    c0 += c

    eiv, eig = np.linalg.eig(c)
    ii = np.argsort(-eiv)
    eiv = eiv[ii]
    eig = eig[:, ii]

    print(df.SEQN.value_counts(), "\n\n")

    eivals.append(eiv)
    eigals.append(eig)
    if len(eivals) > 10:
        break

eivals = np.asarray(eivals)

# Plot the eigenvalues of individual subjects
plt.clf()
plt.grid(True)
lk = np.log10(1 + np.arange(eivals.shape[1]))
for i in range(eivals.shape[0]):
    plt.plot(lk, np.log10(eivals[i, :]), color='grey', alpha=0.5)
plt.xlabel(r"$\log\, k$", size=16)
plt.ylabel(r"$\log\, \lambda_k$", size=16)
pdf.savefig()

# Get the spectrum of the average covariance matrix
c0 /= len(eivals)
eiv, eig = np.linalg.eig(c0)
ii = np.argsort(-eiv)
eiv = eiv[ii]
eig = eig[:, ii]

# Plot the eigenvalues of the average autocovariance matrix
evm = eivals.mean(0)
plt.clf()
plt.grid(True)
lk = np.log10(1 + np.arange(len(eiv)))
plt.plot(lk, np.log10(eiv), color='grey', alpha=0.5)
plt.xlabel(r"$\log\, k$", size=16)
plt.ylabel(r"$\log\, \lambda_k$", size=16)
pdf.savefig()

# Plot the eigenvalues of some individual covariance matrices
for i in range(5):
    plt.clf()
    plt.grid(True)
    plt.title("Subject %d" % (i + 1))
    for j in range(5):
        plt.plot(eigals[i][:, j], label=str(j))
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc="center right")
    leg.draw_frame(False)
    plt.xlabel("Minutes", size=15)
    plt.ylabel("Eigenvalue", size=15)
    pdf.savefig()

# Plot the eigenvalues of the average covariance matrix
plt.clf()
plt.title("Average covariance matrix")
plt.grid(True)
for i in range(5):
    plt.plot(eig[:, i], label=str(i))
ha, lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, loc="center right")
leg.draw_frame(False)
plt.xlabel("Minutes", size=15)
plt.ylabel("Eigenvalue", size=15)
pdf.savefig()

pdf.close()
