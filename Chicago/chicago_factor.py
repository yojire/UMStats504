import pandas as pd
import numpy as np
import statsmodels.api as sm

# If true, retain counts by space and time, if false aggregate over the spatial units
spacetime = True

df = pd.read_csv("cdat.csv.gz")
df = df.loc[pd.notnull(df.CommunityArea), :]
df.Date = pd.to_datetime(df.Date)

# Get the distinct primary types
pt = df.PrimaryType.unique()

if spacetime:
    dy = pd.pivot_table(df[["PrimaryType", "Date", "CommunityArea", "Num"]], index=["Date", "CommunityArea", "PrimaryType"])
else:
    dy = pd.pivot_table(df[["PrimaryType", "Date", "Num"]], index=["Date", "PrimaryType"])

dy = dy.unstack()
dy.columns = dy.columns.get_level_values(1)
dy = dy.reset_index()

# Convert from integer to float
for x in pt:
    dy[x] = dy.loc[:, x].astype(np.float64)

# Variance stabilize and center
dy.loc[:, pt] = np.sqrt(dy.loc[:, pt])
mn = dy.loc[:, pt].mean(0)
dy.loc[:, pt] -= mn

# Get the PC's
cm = np.dot(dy.loc[:, pt].T, dy.loc[:, pt]) / dy.shape[0]
ev, eg = np.linalg.eig(cm)
ii = np.argsort(-ev)
ev = ev[ii]
eg = eg[:, ii]

# Keep the top 4 PC's
edf = pd.DataFrame(eg[:, 0:4], index=pt)

print("Eigenvalues: ", ev, "\n")
print("Eigenvectors:\n", edf)

# Show how the PC's are related to the mean
mn_norm = mn / np.sqrt(np.sum(mn**2))
print(np.dot(mn_norm, edf))

for k in range(4):
    dy["score_%d" % k] = np.dot(dy.loc[:, pt], eg[:, k])

# Reconstruct some things we need for the regressions
dy["Year"] = dy.Date.dt.year
dy["DayOfYear"] = dy.Date.dt.dayofyear
dy["DayOfWeek"] = dy.Date.dt.dayofweek

# Regress each PC score on space and time variables to see how they
# are related
for k in range(4):

    # Time-only model
    fml0 = "score_%d ~ bs(Year, 4) + bs(DayOfYear, 10) + C(DayOfWeek)" % k
    model0 = sm.OLS.from_formula(fml0, data=dy)
    result0 = model0.fit()

    if spacetime:

        # Space-only model
        fml1 = "score_%d ~ C(CommunityArea)" % k
        model1 = sm.OLS.from_formula(fml1, data=dy)
        result1 = model1.fit()

        # Space/timemodel
        fml2 = "score_%d ~ bs(Year, 4) + bs(DayOfYear, 10) + C(DayOfWeek) + C(CommunityArea)" % k
        model2 = sm.OLS.from_formula(fml2, data=dy)
        result2 = model2.fit()

        # Partial R^2 for space over time
        r0 = result0.rsquared
        r1 = result2.rsquared
        pr = (r1 - r0) / (1 - r0)
        print("Factor %d" % (k + 1))
        print("Partial R^2 for space: %f" % pr)

        # Partial R^2 for time over space
        r0 = result1.rsquared
        pr = (r1 - r0) / (1 - r0)
        print("Partial R^2 for time: %f\n" % pr)
