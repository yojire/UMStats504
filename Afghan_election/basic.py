import pandas as pd
import statsmodels.api as sm
from data import df

df = df.dropna()

candidates1 = ['Hilal', 'Abdullah',
              'Rassoul', 'Wardak', 'Karzai', 'Sayyaf', 'Ghani', 'Sultanzoy',
              'Sherzai', 'Naeem', 'Arsala']

candidates2 = ['Abdullah_2', 'Ghani_2']

# Look at the change in the turnout between rounds, in terms of
# the round 1 totals for each candidate.
df["Total_change"] = df.Total_2 - df.Total_1
fml0 = "Total_change ~ " + " + ".join(candidates1)
model0 = sm.OLS.from_formula(fml0, data=df)
result0 = model0.fit()

# Fit models for each round 2 candidate's vote totals in round 2 in terms
# of the round 1 vote totals for all candidates.

fml0 = candidates2[0] + " ~ " + " + ".join(candidates1)
model0 = sm.OLS.from_formula(fml0, data=df)
result0 = model0.fit()

fml1 = candidates2[1] + " ~ " + " + ".join(candidates1)
model1 = sm.OLS.from_formula(fml1, data=df)
result1 = model1.fit()

pa = pd.DataFrame({"Abdullah": result0.params, "Ghani": result1.params})

# Use GEE to look at clustering of PS data within PC's.
mgee = sm.GEE.from_formula(fml0, groups="PC_number", cov_struct=sm.cov_struct.Exchangeable(),
                           data=df)
rgee = mgee.fit()
