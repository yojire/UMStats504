import sys
sys.path.insert(0, "/afs/umich.edu/user/k/s/kshedden/statsmodels_fork/statsmodels")

import pandas as pd
import statsmodels.api as sm
import numpy as np
import patsy
import statsmodels.stats.knockoff_regeffects as kr

from nhanes_data import dx
dy = dx.copy()

dy.DMDEDUC2 = dy.DMDEDUC2.replace({7: np.nan, 9: np.nan})
dy.RIDAGEYR -= dy.RIDAGEYR.mean()
dy.BMXBMI -= dy.BMXBMI.mean()
dy.RIAGENDR = (dy.RIAGENDR == 2).astype(np.float64)
dy["SomeCollege"] = 1*(dy.DMDEDUC2 >= 4)

yvec, xmat = patsy.dmatrices("BPXSY1 ~ 0 + RIDAGEYR * RIAGENDR * BMXBMI * SomeCollege", data=dy, return_type="dataframe")

yvec = yvec.values[:, 0]
yvec -= yvec.mean()
yvec /= yvec.std()

xcols = xmat.columns.tolist()
xmat = np.asarray(xmat)
xmat -= xmat.mean(0)
xmat /= xmat.std(0)

ko = sm.stats.RegressionFDR(yvec, xmat, kr.ForwardEffects(pursuit=True))


