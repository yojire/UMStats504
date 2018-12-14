"""
Merge some NHANES data files to produce a dataset that can
be used to demonstrate the Kockoff FDR procedure.

wget https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DR1TOT_C.XPT

wget https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/BPX_C.XPT

wget https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/BMX_C.XPT

wget https://wwwn.cdc.gov/Nchs/Nhanes/2003-2004/DEMO_C.XPT
"""

import pandas as pd

dx0 = pd.read_sas("DEMO_C.XPT")
dx0 = dx0[["SEQN", "RIDAGEYR", "RIAGENDR", "DMDEDUC2", "RIDRETH1"]]

dx1 = pd.read_sas("BMX_C.XPT")
dx1 = dx1[["SEQN", "BMXBMI"]]

dx2 = pd.read_sas("BPX_C.XPT")
dx2 = dx2[["SEQN", "BPXSY1"]]

dx = pd.merge(dx0, dx1, left_on="SEQN", right_on="SEQN")
dx = pd.merge(dx, dx2, left_on="SEQN", right_on="SEQN")
