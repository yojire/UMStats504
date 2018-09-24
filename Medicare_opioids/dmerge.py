"""
Create a reduced set of utilization data.  Only providers who are also
present in the part D prescription file are included, and only a subset
of columns are included.

The data are aggregated into provider x procedure cells, counting the number
of times each provider performed each procedure.
"""

import pandas as pd
import numpy as np
import gzip

# The CMS utilization file name
util_file = "Medicare_Provider_Util_Payment_PUF_CY2015.txt.gz"

# The CMS medicare part D file name
med_file = "PartD_Prescriber_PUF_NPI_Drug_15.txt.gz"

# The file name to use for the output file.
out_file = "2015_utilization_reduced.csv.gz"

# Get all the NPI numbers from the utilization file
npi_util = pd.read_csv(util_file, usecols=[0], skiprows=[1], delimiter="\t")
npi_util = npi_util.npi.unique()

# Get all the NPI numbers from the prescription file
npi_drugs = pd.read_csv(med_file, usecols=[0], delimiter="\t")
npi_drugs = npi_drugs.npi.unique()

# Get provider id's for providers who are in both the drug and the procedure
# data files
isect = np.intersect1d(npi_drugs, npi_util)

# Read the whole
usecols = ["npi", "nppes_entity_code", "provider_type", "hcpcs_code", "line_srvc_cnt"]
df = pd.read_csv(util_file, skiprows=[1], delimiter="\t", usecols=usecols)
df = df.loc[df.npi.isin(isect), :]
dx = df.groupby(["npi", "hcpcs_code"]).agg({"nppes_entity_code": "first", "provider_type": "first",
                                            "line_srvc_cnt": np.sum})
dx = dx.reset_index()
dx.to_csv(out_file, compression="gzip")
