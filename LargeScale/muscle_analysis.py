import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from scipy.stats.distributions import t as student_t
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

from muscle_data import df, women, men, older, younger

pdf = PdfPages("muscle_analysis.pdf")

# Restrict to the actual gene expression data
a = list(set(women) | set(men))
dx = df.loc[:, df.columns.isin(a)]

# Build a design matrix for main effects and interactions
xmat = np.zeros((dx.shape[1], 4))
xmat[:, 0] = 1
xmat[dx.columns.isin(women), 1] = 1
xmat[dx.columns.isin(older), 2] = 1
xmat[:, 3] = xmat[:, 1] * xmat[:, 2]

# Factorize the design matrix
# (x'x)^{-1} x' = vs^{-1}u'
u, s, vt = np.linalg.svd(xmat, 0)

# Parameter estimates
params = np.dot(np.dot(dx, u) / s, vt)

# Fitted values
fv = np.dot(params, xmat.T)

# Residuals
rd = dx - fv

# Unexplained variation
uv = (rd**2).sum(1) / (dx.shape[1] - 4)

# (x'x)^{-1} = (vs^2v')^{-1}
xtx = np.dot(vt.T / s**2, vt)

# Standard error for the interaction term
se = np.sqrt(uv * xtx[3, 3])

# Z-scores for the interaction term
zs = params[:, 3] / se
zs = zs.dropna()
zsa = np.abs(zs)

# P-values for the interaction term
pv = student_t.cdf(-np.abs(zs), xmat.shape[0] - xmat.shape[1])

# Bonferroni threshold
bt = norm.ppf(1 - 0.025 / zs.shape[0])

# Calculate the FDR for a range of threshold from 2 to 5.
fdr = []
n = len(zs)
for t in np.linspace(0, 6, 20):
    d = np.sum(zsa > t)
    f = 2*n*norm.cdf(-t) / d
    fdr.append([t, f, d])
fdr = np.asarray(fdr)

# Plots relating to FDR
plt.clf()
plt.grid(True)
plt.plot(fdr[:, 0], fdr[:, 1], '-')
plt.xlabel("Z-score threshold", size=15)
plt.ylabel("FDR", size=15)
pdf.savefig()

for trunc in False, True:
    plt.clf()
    plt.grid(True)
    plt.plot(fdr[:, 0], fdr[:, 2], '-')
    if trunc:
        plt.ylim(0, 200)
    plt.xlabel("Z-score threshold", size=15)
    plt.ylabel("Number of discoveries", size=15)
    pdf.savefig()

for trunc in False, True:
    plt.clf()
    plt.grid(True)
    plt.plot(fdr[:, 1], fdr[:, 2], '-')
    if trunc:
        plt.ylim(0, 200)
    plt.xlabel("FDR", size=15)
    plt.ylabel("Number of discoveries", size=15)
    pdf.savefig()

# Use interpolation to get traditional (tail-based) FDR values
f = interp1d(fdr[:, 0], fdr[:, 1])
bh_fdr = [f(z) for z in zsa]
bh_fdr = np.asarray(bh_fdr)

# Get local FDR values
loc_fdr = sm.stats.local_fdr(zs)

dr = pd.DataFrame({"bh": bh_fdr, "local": loc_fdr.values, "zs": zs}, index=loc_fdr.index)

# Plot local versus traditional FDR, color the positive and negative effects
# differently
drd = dr.dropna()
plt.clf()
plt.grid(True)
ii = drd.zs > 0
plt.plot(drd.bh[ii], drd.local[ii], 'o', rasterized=True)
ii = drd.zs < 0
plt.plot(drd.bh[ii], drd.local[ii], 'o', rasterized=True)
plt.xlabel("BH-FDR", size=15)
plt.ylabel("Local-FDR", size=15)
pdf.savefig()


pdf.close()
