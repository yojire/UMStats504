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
b, e = np.linalg.eig(cm)
ii = np.argsort(-b)
b = b[ii]
e = e[:, ii]

# Keep the top 3 PC's
edf = pd.DataFrame(e[:, 0:3], index=pt)

# Show how the PC's are related to the mean
mn_norm = mn / np.sqrt(np.sum(mn**2))
print(np.dot(mn_norm, edf))

for k in range(3):
    dy["score_%d" % k] = np.dot(dy.loc[:, pt], e[:, k])

dy["Year"] = dy.Date.dt.year
dy["DayOfYear"] = dy.Date.dt.dayofyear
dy["DayOfWeek"] = dy.Date.dt.dayofweek

for k in range(3):

    fml0 = "score_%d ~ bs(Year, 4) + bs(DayOfYear, 10) + C(DayOfWeek)" % k
    model0 = sm.OLS.from_formula(fml0, data=dy)
    result0 = model0.fit()
    #print(result0.summary())

    if spacetime:
        fml = "score_%d ~ bs(Year, 4) + bs(DayOfYear, 10) + C(DayOfWeek) + C(CommunityArea)" % k
        model = sm.OLS.from_formula(fml, data=dy)
        result = model.fit()
        #print(result.summary())

        r0 = result0.rsquared
        r1 = result.rsquared
        pr = (r1 - r0) / (1 - r0)
        print("Partial R^2 for space: %f" % pr)
