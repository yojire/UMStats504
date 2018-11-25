"""
A basic multilevel analysis.  The votes cast for one candidate in one
round of the election are totalled and log transformed, then a variance
decomposition is performed looking at variance contributions from
three nested geographical clusters.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from data import df

df = df.dropna()

# Round 1
candidates1 = ['Hilal', 'Abdullah', 'Rassoul', 'Wardak', 'Karzai',
               'Sayyaf', 'Ghani', 'Sultanzoy', 'Sherzai', 'Naeem',
               'Arsala']

# Round 2
candidates2 = ['Abdullah_2', 'Ghani_2']

# Geographic variables
geo = ["PC_number", "District", "Province"]

for candidate in candidates2 + candidates1:

    dx = df.loc[:, [candidate] + geo]

    vcf = {"District": "0 + C(District)", "PC_number": "0 + C(PC_number)"}
    dx[candidate] = np.log(1 + dx[candidate])
    model = sm.MixedLM.from_formula(candidate + " ~ 1", re_formula="1", groups="Province", vc_formula=vcf, data=dx)
    result = model.fit()

    print(candidate)
    print(result.summary())
