# This script requires statsmodels version 0.9 or above.  If your statsmodels
# is too old, use this shell command to update it:
#     pip install --upgrade --user statsmodels

# If True, do not run the procedure analysis, which requires a lot of memory
# and may hang (in TruncatedSVD)
noproc = True

# The following two lines are needed on flux and some other headless systems
import matplotlib
matplotlib.use('agg')

import patsy
import pandas as pd
import numpy as np
from drug_categories import dcat
import statsmodels.api as sm
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Read the medications file
med_file = "PartD_Prescriber_PUF_NPI_Drug_15.txt.gz"
dfx = pd.read_csv(med_file, delimiter="\t")

# Create a variable indicating which drugs are opioids
dfx["opioid"] = dfx.drug_name.isin(dcat["opioid"])

# Get the total day supply for each provider, for opioids and non-opioids
dr = dfx.groupby(["npi", "opioid"]).agg({"total_day_supply": np.sum})
dr = dr.unstack()
dr = dr.rename(columns={False: "Non_opioids", True: "Opioids"})
dr = dr.fillna(0)
dr = dr.reset_index()
dr.columns = ["npi", "Non_opioids", "Opioids"]

# Merge in the state where the provider operates
ds = dfx.groupby("npi").agg({'nppes_provider_state': "first", "nppes_provider_city": "first"})
ds = ds.reset_index()
dr = pd.merge(dr, ds, left_on="npi", right_on="npi")
dr = dr.rename(columns={"nppes_provider_state": "state", "nppes_provider_city": "city"})

# Drop providers who never prescribe opioids, or who never prescribe any drugs.
dr = dr.loc[dr.Opioids > 0, :]
dr = dr.loc[dr.Non_opioids > 0, :]

# Remove very small states
s = dr.groupby("state").size()
s = pd.DataFrame({"num_providers": s})
dr = pd.merge(dr, s, left_on="state", right_index=True, how='left')
dr = dr.loc[dr.num_providers >= 500]

# Create log transformed variables and z-scores
dr["log_op"] = np.log2(0.1 + dr.Opioids)
dr["log_nonop"] = np.log2(0.1 + dr.Non_opioids)
dr["log_op_z"] = (dr.log_op - dr.log_op.mean()) / dr.log_op.std()
dr["log_nonop_z"] = (dr.log_nonop - dr.log_nonop.mean()) / dr.log_nonop.std()

dr = dr.dropna()

pdf = PdfPages("opioids.pdf")

# Plot overall log/log data
plt.clf()
plt.plot(dr.log_op, dr.log_nonop, 'o', color='orange', alpha=0.3, rasterized=True)
plt.xlabel("Non-opioid days supply")
plt.ylabel("Opioids days supply")
plt.grid(True)
pdf.savefig()

# Basic log/log regression model, looking at how overall prescribing
# predicts opioid prescribing
olsmodel = sm.OLS.from_formula("log_op ~ log_nonop", data=dr)
olsresult = olsmodel.fit()
olsmodelz = sm.OLS.from_formula("log_op_z ~ log_nonop_z", data=dr)
olsresultz = olsmodelz.fit()

# Fit log opioid versus non-opioid volume separately by state
plt.clf()
bxr = np.linspace(-4, 4, 20)
for sn, dz in dr.groupby("state"):
    # sn is the state's name, dz is the data for that state
    model = sm.OLS.from_formula("log_op_z ~ log_nonop_z", data=dz)
    result = model.fit()
    params = result.params.values
    plt.plot(bxr, params[0] + params[1]*bxr, '-', color='grey')
plt.title("Trends by state")
plt.xlabel("Log non-opioid days supply (Z-score)")
plt.ylabel("Log opioid days supply (Z-score)")
pdf.savefig()

pdf.close()

# Fit three different fixed effects models for opioid volume relative to non-opioid volume.

# Allow the intercepts to vary by state (no z-scoring)
femodel = sm.OLS.from_formula("log_op ~ log_nonop + C(state)", data=dr)
feresult = femodel.fit()

# Allow the intercepts to vary by state (with z-scoring)
femodelz = sm.OLS.from_formula("log_op_z ~ log_nonop_z + C(state)", data=dr)
feresultz = femodelz.fit()

# Allow the slopes and intercepts to vary by state (with z-scoring)
femodelsz = sm.OLS.from_formula("log_op_z ~ log_nonop_z * C(state)", data=dr)
feresultsz = femodelsz.fit()

# Pairwise comparison of state intercepts, for every pair of states.
# It's important here to use centered (or Z-scored) data here so the
# intercepts are inside the range of the data.

pa = feresultz.params # Parameter estimates
vc = feresultz.cov_params() # Covariance matrix of parameter estimates

# Positions of the state fixed effects in the parameter vector
ii = [i for i,x in enumerate(pa.index) if x.startswith("C(state")]

# Reduce the parameter estimate vector and covariance matrix to contain
# only the parameters corresponding to state effects
pa = pa.iloc[ii]
vc = vc.iloc[ii, ii]

# Get the contrast for every pair of states
zc = np.zeros((len(pa), len(pa)))
for i1 in range(len(pa)):
    for i2 in range(len(pa)):

        # Create a contrast vector between states i1 and i2
        d = np.zeros(len(pa))
        d[i1] = 1
        d[i2] = -1

        # Get the contrast and standard error (for comparing these two states)
        se = np.sqrt(np.dot(d, np.dot(vc, d)))
        zc[i1, i2] = (pa[i1] - pa[i2]) / se

# Put the results in a dataframe with columns showing the two states
# being compared, and the Z-score for the difference of the state interepts
j1, j2 = np.tril_indices(len(pa), -1) # The lower triangle so we don't repeat pairs
statepair_icept_z = pd.DataFrame({"State1": pa.index.values[j1],
                                  "State2": pa.index.values[j2],
                                  "Diff": zc[j1, j2]})
statepair_icept_z = statepair_icept_z.sort_values(by="Diff")

# Pairwise comparison of state slopes, for every pair of states
# See preceeding section, the code is very simuilar except here we
# work with the slope parameters instead of the intercept parameters

pa = feresultsz.params # All parameter estimates
vc = feresultsz.cov_params() # Covariance matrix for all parameter estiamtes

# Restrict to only the slope parameter estimates
ii = [i for i,x in enumerate(pa.index) if x.startswith("log_nonop_z:C(state")]
pa = pa.iloc[ii]
vc = vc.iloc[ii, ii]

zcx = np.zeros((len(pa), len(pa)))
for i1 in range(len(pa)):
    for i2 in range(len(pa)):

        # Contrast vector
        d = np.zeros(len(pa))
        d[i1] = 1
        d[i2] = -1

        # Standard error of the contrast
        se = np.sqrt(np.dot(d, np.dot(vc, d)))

        # Z-score of the contrast
        zcx[i1, i2] = (pa[i1] - pa[i2]) / se

j1, j2 = np.tril_indices(len(pa), -1)
statepair_slope_z = pd.DataFrame({"State1": pa.index.values[j1],
                                  "State2": pa.index.values[j2],
                                  "Diff": zcx[j1, j2]})
statepair_slope_z = statepair_slope_z.sort_values(by="Diff")

# Mean/variance analysis for basic model
dr["fit"] = olsresult.fittedvalues
dr["bin"] = pd.qcut(dr.fit, 20) # Slice the fitted values into 20 bins
meanvar_base = dr.groupby("bin")["fit", "log_op"].agg({"fit": np.mean, "log_op": np.var})
meanvar_base = meanvar_base.rename(columns={"fit": "mean", "log_op": "var"})

# Mean/variance analysis for state-adjusted model
dr["fit"] = feresult.fittedvalues
dr["bin"] = pd.qcut(dr.fit, 20) # Slice the fitted values into 20 bins
meanvar_states = dr.groupby("bin")["fit", "log_op"].agg({"fit": np.mean, "log_op": np.var})
meanvar_states = meanvar_states.rename(columns={"fit": "mean", "log_op": "var"})

# Basic mixed model with intercepts that vary by state ("random intercepts")
memodel_i = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="state", data=dr)
meresult_i = memodel_i.fit()

# Mixed model with slopes that vary by state ("random slopes")
memodel_is = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="state",
                                     vc_formula={"i": "0 + C(state)", "s": "0+log_nonop_z:C(state)"},
                                     data=dr)
meresult_is = memodel_is.fit()

# Basic mixed model for cities (random intercepts by city)
# too slow to run
#memodel_c = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="city", data=dr)
#meresult_c = memodel_c.fit()

db = pd.read_csv("2015_utilization_reduced.csv.gz")

# Merge provider type
du = db.groupby("npi")["provider_type"].agg("first")
du = pd.DataFrame(du).reset_index()
dr = pd.merge(dr, du, left_on="npi", right_on="npi")

# Use LARS to consider provider type effects.  LARS can't accept a formula, so we
# construct the response vector and design matrix outside of LARS then pass them
# in.
y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + C(provider_type)", data=dr,
                       return_type='dataframe')

# Standardize
xa = np.asarray(x)
xa -= xa.mean(0)
xa /= xa.std(0)
ya = np.asarray(y)[:, 0]
ya -= ya.mean(0)
ya /= ya.std(0)
xnames = x.columns.tolist()

# Run LARS
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)
coefs = coefs[:, 1:] # coefs starts with an extra column of zeros

# Display the first few variables selected by LARS and the correlation
# between fitted and observed values
for k in range(20):

    # k denotes the order in which the variables are added to the model
    # by LARS.  active[k-1] contains the indicators of the variables in
    # the model at stage k

    # Print the variables in the order they are selected, and the correlation
    # coefficients between these fitted values and the observed outcomes.
    f = np.dot(xa, coefs[:, k])
    print("%-60s %6.2f %8.4f" % (xnames[active[k]],
                                 np.corrcoef(ya, f)[0, 1],
                                 coefs[active[k], k]))

# Fixed effects model for provider type
pfemodelz = sm.OLS.from_formula("log_op_z ~ log_nonop_z + C(provider_type)",
                                data=dr)
pferesultz = pfemodelz.fit()

# Basic mixed model (random intercepts by provider)
pmemodelz = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="provider_type", data=dr)
pmeresultz = pmemodelz.fit()

# Merge the BLUPs with the fixed effects estimates
re = pmeresultz.random_effects
re = {k: re[k].provider_type for k in re.keys()}
re = pd.DataFrame({"re": pd.Series(re)})
fe = {}
for k in pferesultz.params.index:
    if k != "Intercept" and k != "log_nonop_z":
        v = k.replace("C(provider_type)[T.", "")
        v = v[0:-1]
        fe[v] = pferesultz.params[k]
fe = pd.Series(fe)
fe = pd.DataFrame(fe, columns=["fe"])
refe = pd.merge(fe, re, left_index=True, right_index=True)

# Mean center to eliminate dependence on the reference category
refe -= refe.mean(0)

# Add in the sample size (per group) so we can see how this relates
# to the degree of shrinkage
n = dr.provider_type.value_counts()
n = pd.DataFrame(n)
n.columns=["n"]
refe = pd.merge(refe, n, left_index=True, right_index=True)

# Mixed model looking at provider types: allow varying intercepts and slopes
# by provider type
pmemodels = sm.MixedLM.from_formula("log_op_z ~ log_nonop_z", groups="provider_type",
                                    vc_formula={"i": "0+C(provider_type)",
                                                "s": "0+log_nonop_z:C(provider_type)"},
                                    data=dr)
pmeresults = pmemodels.fit()

# Use LARS to consider provider effects and state effects together
y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + C(provider_type) + C(state)", data=dr,
                       return_type='dataframe')

# Standardize
xa = np.asarray(x)
xa -= xa.mean(0)
xa /= xa.std(0)
ya = np.asarray(y)[:,0]
ya -= ya.mean(0)
ya /= ya.std(0)
xnames = x.columns.tolist()

# Run LARS for provider and state varying intercepts
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)
coefs = coefs[:, 1:]

# Display the first few variables selected by LARS, and the correlations between
# fitted and observed values for each model selected by LARS.
print("\nLARS path for provider type and state main effects:")
for k in range(20):
    f = np.dot(xa, coefs[:, k])
    print("%-60s %6.2f %8.4f" % (xnames[active[k]],
                                 np.corrcoef(ya, f)[0, 1],
                                 coefs[active[k], k]))

if noproc:
    import sys
    sys.exit(0)

#
# Procedure type analysis
#

# Count the number of times each provider performs each procedure type
du = db.groupby(["npi", "hcpcs_code"])["line_srvc_cnt"].agg(np.sum)
du = du.unstack(fill_value=0)

# Use SVD to form composite variables of the procedure data
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
xr = svd.fit_transform(du.values)
xr = pd.DataFrame(xr, index=du.index)
xr.columns = ["proc_%02d" % k for k in range(xr.shape[1])]

# Remove any existing 'proc' variables from dr so we can merge in the new proc composite
# variables
for c in dr.columns:
    if c.startswith("proc"):
        dr = dr.drop(c, axis=1)
dr = pd.merge(dr, xr, left_on="npi", right_index=True)

# Use LARS to get the solution path for procedure types
procs = " + ".join(xr.columns.tolist())
y, x = patsy.dmatrices("log_op_z ~ 0 + log_nonop_z + %s" % procs,
                       data=dr, return_type='dataframe')

# Standardize the data before running LARS on it
xa = np.asarray(x)
ya = np.asarray(y)[:,0]
xa -= xa.mean(0)
xa /= xa.std(0)
ya -= ya.mean()
ya /= ya.std()
xnames = x.columns.tolist()

# Run LARS on procedure type factors
alphas, active, coefs = linear_model.lars_path(xa, ya, method='lars', verbose=True)

# Display the first few variables selected by LARS, and the
# correlations between fitted and observed values at each step.
print("\nLARS path for procedure types:")
for k in range(1, 10):
    print(xnames[active[k-1]])
    f = np.dot(xa, coefs[:, k])
    print(k, np.corrcoef(ya, f)[0, 1])
