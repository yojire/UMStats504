import pandas as pd
import numpy as np
from datetime import datetime

# It's much faster to read the file if we fix the date/time format.
# Otherwise it has to guess each row's format.
to_datetime = lambda d: datetime.strptime(d, '%m/%d/%Y %H:%M:%S %p')

# Since the file is large, load a limited number of variables.
c = ["Date", "Primary Type", "Community Area"]
df = pd.read_csv("chicago.csv.gz", usecols=c, converters={"Date": to_datetime})
df = df.rename(columns={"Community Area": "CommunityArea", "Primary Type": "PrimaryType"})

# Limit to the 10 most common types of crime
ptype = df.loc[:, "PrimaryType"]
pt = ptype.value_counts()
pt10 = pt[0:10].index
dx = df.loc[ptype.isin(pt10), :]

# Count the number of times each crime type occurs in each community area on each day.
first = lambda x: x.iloc[0]
dx["Date"] = dx.Date.dt.floor('1d')
dy = dx.groupby(["PrimaryType", "Date", "CommunityArea"]).size()
dy.name = "Num"

# Expand the index so that every day has a record
a = dx["PrimaryType"].unique()
b = np.arange(dx.Date.min(), dx.Date.max(), pd.to_timedelta('1d'))
c = dx["CommunityArea"].unique()
ix = pd.MultiIndex.from_product([a, b, c])
dy = dy.reindex(ix)
dy = dy.fillna(0)
dy = dy.astype(np.int)

# Move the hierarchical row index into columns, rename to get more meaningful names.
cdat = dy.reset_index()
cdat = cdat.rename(columns={"level_0": "PrimaryType", "level_1": "Date", "level_2": "CommunityArea"})

# Split date into variables for year, day within year, and day of week.
cdat["DayOfYear"] = cdat.Date.dt.dayofyear
cdat["Year"] = cdat.Date.dt.year
cdat["DayOfWeek"] = cdat.Date.dt.dayofweek

cdat.to_csv("cdat.csv.gz", compression="gzip", index=None)
