import pandas as pd
import numpy as np

dpath = "/nfs/turbo/arcts-dads-corelogic/Data/deed/0004.gz"

dtp = {"SALE DATE (YYYYMMDD)": str}

rdr = pd.read_csv(dpath, delimiter="|", chunksize=200000, low_memory=False,
                  dtype=dtp)

# Loop over sub-chunks
dat = []
while True:

    try:
        df = rdr.get_chunk()
    except StopIteration:
        break

    df["SALE DATE (YYYYMMDD)"] =\
           pd.to_datetime(df["SALE DATE (YYYYMMDD)"], format="%Y%m%d", errors='coerce')

    dx = df[["APN (Parcel Number) (unformatted)", "SALE DATE (YYYYMMDD)", "SALE AMOUNT", "MORTGAGE AMOUNT",
             "RESALE/NEW CONSTRUCTION", "RESIDENTIAL MODEL INDICATOR", "CASH/MORTGAGE PURCHASE",
             "FORECLOSURE", "FIPS", "TRANSACTION TYPE"]]
    ii = pd.notnull(dx[["APN (Parcel Number) (unformatted)", "SALE DATE (YYYYMMDD)", "CASH/MORTGAGE PURCHASE",
                        "TRANSACTION TYPE"]]).all(1)
    dx = dx.loc[ii, :]

    # Only retain records that are for "arms length sales"
    dx = dx.loc[dx["TRANSACTION TYPE"] == 1, :]

    # Convert to a number (days since 1960-01-01)
    dx["SALE DATE"] = dx["SALE DATE (YYYYMMDD)"] - pd.to_datetime("1960-01-01")
    dx["SALE DATE"] = dx["SALE DATE"].dt.days
    dx["SALE DATE"] = dx["SALE DATE"].astype(np.float64)

    # Drop non-residential properties
    dx = dx.loc[dx["RESIDENTIAL MODEL INDICATOR"] == "Y"]

    dx = dx.loc[:, ["APN (Parcel Number) (unformatted)", "SALE DATE", "SALE AMOUNT", "FIPS",
                    "CASH/MORTGAGE PURCHASE", "MORTGAGE AMOUNT"]]

    dat.append(dx)

deed = pd.concat(dat, axis=0)

# Drop properties with only one record
gb = deed.groupby("APN (Parcel Number) (unformatted)")
nr = pd.DataFrame(gb.size())
nr.columns = ["numrecs"]
deed = pd.merge(deed, nr, left_on="APN (Parcel Number) (unformatted)", right_index=True)
deed = deed.loc[deed.numrecs > 1]
