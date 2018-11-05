import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from tax_data import tax
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# County population data
# http://www.nber.org/data/census-intercensal-county-population.html
cp = pd.read_csv("county_population.csv", encoding="latin1")
cp = pd.melt(cp, id_vars="fips", var_name="year", value_name="pop")
ii = cp.year.str.startswith("pop")
cp = cp.loc[ii, :]
cp["year"] = cp.year.apply(lambda x: x[3:])
cp = cp.loc[cp.fips != 0, :]
cp = cp.dropna()
cp["year"] = cp.year.astype(np.int)
cp["pop"] = cp["pop"].astype(np.int)
cp["logpop"] = np.log(cp["pop"])

# Count the number of houses constructed per year in each FIPS region
nb = tax.loc[tax["YEAR BUILT"] >= 1960, :]
nb = nb.groupby(["FIPS CODE", "YEAR BUILT"]).size()
nb = nb.reset_index()
nb = nb.loc[nb["YEAR BUILT"] >= 1960, :]
nb.columns = ["fips", "year", "numbuilt"]
nb["year"] = nb["year"].astype(np.int)

# Add zeros where no houses were constructed
years = np.arange(1960, 2016)
fips = nb["fips"].unique()
ix = pd.MultiIndex.from_product([years, fips], names=["year", "fips"])
nb2 = pd.DataFrame(index=ix).reset_index()
nb3 = pd.merge(nb2, nb, left_on=["year", "fips"], right_on=["year", "fips"], how='left')
nb3.numbuilt = nb3.numbuilt.fillna(0)
nb = nb3.dropna()

# Merge in the county population data
nb = pd.merge(cp, nb, left_on=["fips", "year"], right_on=["fips", "year"])

ydf = 6

pdf = PdfPages("year_built.pdf")

for fm in "", " * logpop":
    qm = []
    for alpha in 2**np.linspace(-3, 3, 10):
        model = sm.GLM.from_formula("numbuilt ~ bs(year, %d)" % ydf + fm, data=nb, family=sm.families.NegativeBinomial(alpha=alpha))
        result = model.fit()
        qm.append([alpha, result.aic])
    qm = np.asarray(qm)
    plt.clf()
    plt.axes([0.16, 0.13, 0.8, 0.8])
    plt.title("Year" if fm == "" else "Year and population")
    plt.plot(qm[:, 0], qm[:, 1])
    plt.xlabel("Alpha", size=16)
    plt.ylabel("AIC", size=16)
    plt.grid(True)
    pdf.savefig()

# Fit Poisson GLMs
model0 = sm.GLM.from_formula("numbuilt ~ bs(year, %s)" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=3))
result0 = model0.fit()
model1 = sm.GLM.from_formula("numbuilt ~ bs(year, %s) * logpop" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=1))
result1 = model1.fit()

for (model, result) in (model0, result0), (model1, result1):

    # Check the mean/variance relationship for the Poisson model
    qt = pd.qcut(result.fittedvalues, 40 if model is model1 else 20)
    dy = pd.DataFrame({"fit": result.fittedvalues, "obs": model.endog})
    qa = dy.groupby(qt).agg({"fit": np.mean, "obs": np.var})

    # log mean / log variance plot
    plt.clf()
    xx = np.log(qa.fit)
    plt.plot(xx, np.log(qa.obs), 'o')
    x = np.linspace(1.05*xx.min(), 1.05*xx.max(), 10)
    plt.plot(x, np.log(np.exp(x) + 0.5*np.exp(2*x)), '-')
    plt.xlabel("Log mean", size=15)
    plt.ylabel("Log variance", size=15)
    plt.grid(True)
    if result is result0:
        plt.title("Adjust for year")
    else:
        plt.title("Adjust for year and population")
    pdf.savefig()


# Fit models controling for year only, or controlling for year and population (with full interactions).
# Also fit using three different GEE covariance structures.

model2 = sm.GEE.from_formula("numbuilt ~ bs(year, %d)" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=1),
                            cov_struct=sm.cov_struct.Independence(), groups="fips")
result2 = model2.fit()

model3 = sm.GEE.from_formula("numbuilt ~ bs(year, %d)" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=1),
                             cov_struct=sm.cov_struct.Stationary(max_lag=15), time=nb.year-1960, groups="fips")
result3 = model3.fit(maxiter=5)

model4 = sm.GEE.from_formula("numbuilt ~ bs(year, %d)" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=1),
                             cov_struct=sm.cov_struct.Exchangeable(), time=nb.year-1960, groups="fips")
result4 = model4.fit(maxiter=5)

model5 = sm.GEE.from_formula("numbuilt ~ bs(year, %d) * logpop" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=3),
                            cov_struct=sm.cov_struct.Independence(), groups="fips")
result5 = model5.fit()

model6 = sm.GEE.from_formula("numbuilt ~ bs(year, %d) * logpop" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=3),
                             cov_struct=sm.cov_struct.Stationary(max_lag=15), time=nb.year-1960, groups="fips")
result6 = model6.fit(maxiter=5)

model7 = sm.GEE.from_formula("numbuilt ~ bs(year, %d) * logpop" % ydf, data=nb, family=sm.families.NegativeBinomial(alpha=3),
                             cov_struct=sm.cov_struct.Exchangeable(), time=nb.year-1960, groups="fips")
result7 = model7.fit(maxiter=5)

# Plot the estimated autocorrelations
for result in result3, result6:
    plt.clf()
    plt.plot(result.cov_struct.dep_params)
    plt.grid(True)
    plt.gca().set_xticks(range(18))
    plt.xlim(0, 15)
    plt.xlabel("Lag (years)", size=15)
    plt.ylabel("Autocorrelation", size=15)
    pdf.savefig()

from statsmodels.sandbox import predict_functional
pred1, cb1, fvals1 = predict_functional.predict_functional(result5, "year", ci_method="simultaneous", values={"logpop": 10})
pred2, cb2, fvals2 = predict_functional.predict_functional(result5, "year", ci_method="simultaneous", values={"logpop": 11})
pred3, cb3, fvals3 = predict_functional.predict_functional(result5, "year", ci_method="simultaneous", values={"logpop": 12})

for k in range(2):
    plt.clf()
    plt.axes([0.15, 0.1, 0.72, 0.86])
    if k == 0:
        # Plot fitted values on the log scale
        plt.plot(fvals1, pred1, '-', label="10")
        plt.plot(fvals2, pred2, '-', label="11")
        plt.plot(fvals3, pred3, '-', label="12")
        plt.fill_between(fvals1, cb1[:, 0], cb1[:, 1], color='grey')
        plt.fill_between(fvals2, cb2[:, 0], cb2[:, 1], color='grey')
        plt.fill_between(fvals3, cb3[:, 0], cb3[:, 1], color='grey')
        plt.ylabel("Housing starts (log scale)", size=15)
    else:
        # Plot fitted values on the scale of the data
        plt.plot(fvals1, 100*np.exp(pred1), '-', label="10")
        plt.plot(fvals2, 100*np.exp(pred2), '-', label="11")
        plt.plot(fvals3, 100*np.exp(pred3), '-', label="12")
        plt.fill_between(fvals1, 100*np.exp(cb1[:, 0]), 100*np.exp(cb1[:, 1]), color='grey')
        plt.fill_between(fvals2, 100*np.exp(cb2[:, 0]), 100*np.exp(cb2[:, 1]), color='grey')
        plt.fill_between(fvals3, 100*np.exp(cb3[:, 0]), 100*np.exp(cb3[:, 1]), color='grey')
        plt.ylabel("Housing starts (US projection)", size=15)
    ha, lb = plt.gca().get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, "center right")
    leg.set_title("log pop")
    leg.draw_frame(False)
    plt.xlabel("Year", size=15)
    pdf.savefig()

# Get the residuals and pivot them to wide form (FIPS in rows, years in columns)
nb["resid"] = result5.resid_pearson
xt = nb.pivot("fips", "year", "resid")
x0 = np.asarray(xt.fillna(value=xt.mean().mean())) # Simple mean imputation
u, s, vt = np.linalg.svd(x0, 0)

plt.clf()
plt.title("Singular values of completed residuals")
plt.plot(1960 + np.arange(len(s)), s)
plt.grid(True)
plt.xlabel("Component", size=15)
plt.ylabel("Singular value", size=15)
pdf.savefig()

plt.clf()
plt.title("Singular vectors of completed residuals")
plt.plot(vt[0,:], label="1")
plt.plot(vt[1,:], label="2")
plt.grid(True)
plt.xlabel("Year", size=15)
plt.ylabel("Singular vector loading", size=15)
ha,lb = plt.gca().get_legend_handles_labels()
leg = plt.figlegend(ha, lb, "center right")
leg.draw_frame(False)
pdf.savefig()

plt.clf()
plt.hist(u[:,0] * s[0])
plt.title("Component 1")
plt.xlabel("FIPS region coefficient", size=15)
plt.ylabel("Frequency", size=15)
pdf.savefig()

plt.clf()
plt.hist(u[:,1] * s[1])
plt.title("Component 2")
plt.xlabel("FIPS region coefficient", size=15)
plt.ylabel("Frequency", size=15)
pdf.savefig()

pdf.close()
