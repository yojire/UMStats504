import pandas as pd
import numpy as np
from datetime import datetime

# It's much faster to read the file if we fix the date/time format.
# Otherwise it has to guess each row's format.
to_datetime = lambda d: datetime.strptime(d, '%m/%d/%Y %H:%M:%S %p')

# Since the file is large, load a limited number of variables.
c = ["Date", "Primary Type", "Community Area", "District", "Ward", "Latitude", "Longitude"]
df = pd.read_csv("chicago.csv.gz", usecols=c, converters={"Date": to_datetime})
df = df.rename(columns={"Community Area": "CommunityArea", "Primary Type": "PrimaryType"})

# Limit to the 10 most common types of crime
ptype = df.loc[:, "PrimaryType"]
pt = ptype.value_counts()
pt10 = pt[0:10].index
dx = df.loc[ptype.isin(pt10), :]

# Count the number of times each crime type occurs in each community area on each day.
first = lambda x: x.iloc[0]
dx["DateOnly"] = dx.Date.dt.date
dy = dx.groupby(["PrimaryType", "DateOnly", "CommunityArea"]).agg({"DateOnly": [np.size, lambda x: x.iloc[0]],
                 "District": first, "Ward": first, "Latitude": first, "Longitude": first})

# Expand the index so that every day has a record
a = dx["PrimaryType"].unique()
b = np.arange(dx.DateOnly.min(), dx.DateOnly.max())
c = dx["CommunityArea"].unique()
ix = pd.MultiIndex.from_product([a, b, c])
dy = dy.reindex(ix)

dy.columns = [' '.join(col).strip() for col in dy.columns.values]
dy = dy.rename(columns={'Latitude <lambda>': "Latitude",
                        'Ward <lambda>': "Ward",
                        'DateOnly size': "Num",
                        'DateOnly <lambda>': "DateOnly",
                        'Longitude <lambda>': "Longitude",
                        'District <lambda>': "District"})

dy.Num = dy.Num.fillna(0)

# Move the hierarchical row index into columns, rename to get more meaningul names.
cdat = dy.reset_index()
cdat = cdat.rename(columns={"level_0": "PrimaryType", "level_1": "Date", "level_2": "CommunityArea"})

# Split date into variables for year, day within year, and day of week.
cdat["DayOfYear"] = cdat.Date.dt.dayofyear
cdat["Year"] = cdat.Date.dt.year
cdat["DayOfWeekNum"] = cdat.Date.dt.dayofweek
cdat["DayOfWeek"] = cdat.Date.dt.dayofweek.replace(
    {0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr", 5: "Sa", 6: "Su"})

cdat.to_csv("cdat.csv.gz", compression="gzip")
