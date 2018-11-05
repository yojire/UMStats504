import pandas as pd
import numpy as np

dpath = "/nfs/turbo/arcts-dads-corelogic/Data/tax/0004.gz"

dtp = {"CENSUS TRACT": str} #"SALE DATE (YYYYMMDD)": str}

rdr = pd.read_csv(dpath, delimiter="|", chunksize=200000, low_memory=False,
                  dtype=dtp, error_bad_lines=False)

n = 0
# Loop over sub-chunks
dat = []
while True:

    try:
        df = rdr.get_chunk()
    except StopIteration:
        break

    n += df.shape[0]

    # Keep only single family residences
    df = df.loc[df["PROPERTY INDICATOR"] == 10, :]

    dx = df[['FIPS CODE', 'UNFORMATTED APN', 'YEAR BUILT']]

    dat.append(dx)

tax = pd.concat(dat, axis=0)

# Keep only one record per house
tax = tax.groupby("UNFORMATTED APN").head(1)

# We need to know the year in which the house was built
tax = tax.loc[pd.notnull(tax["YEAR BUILT"]), :]
