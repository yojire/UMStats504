library(data.table)
library(lubridate)
library(R.utils)

# Since the file is large, load a limited number of variables.
c = c("Date", "Primary Type", "Community Area")
df = fread("zcat chicago.csv.gz", select=c, showProgress=TRUE)
setnames(df, old="Community Area", new="CommunityArea")
setnames(df, old="Primary Type", new="PrimaryType")

# Convert the date (we don't need the time)
setnames(df, old="Date", new="DateString")
df[, Datect := as.POSIXct(DateString, format="%m/%d/%Y %H:%M:%S %p", tz="America/Chicago")]
df[, Date := as.Date(Datect)]

# Get rid of columns that we no longer need
df[, Datect := NULL]
df[, DateString := NULL]

# Limit to the 10 most common types of crime
ft = table(df$PrimaryType)
ft = sort(ft, decreasing=TRUE)[1:10]
ft = names(ft)
df = df[PrimaryType %in% ft,]

# Count the number of times each crime type occurs in each community area on each day
setkey(df, "Date", "PrimaryType", "CommunityArea")
df[, Num := .N, by=c("Date", "PrimaryType", "CommunityArea")]
dc = df[, head(.SD, 1), by=c("Date", "PrimaryType", "CommunityArea")]

# Create a complete enumeration of location x date x crime type
da = seq(from=min(dc$Date), to=max(dc$Date), "days")
ix = CJ(Date=da, PrimaryType=unique(dc$PrimaryType), CommunityArea=unique(dc$CommunityArea))

# Include zeros where no crimes were reported
setkey(dc, "Date", "PrimaryType", "CommunityArea")
setkey(ix, "Date", "PrimaryType", "CommunityArea")
dc = dc[ix]
j = which(names(dc) == "Num")
set(dc, which(is.na(dc$Num)), j, 0)

# Construct values for year, day of year, and day of week
dc[, Year := lubridate::year(Date)]
dc[, DayOfYear := lubridate::yday(Date)]
dc[, DayOfWeek := lubridate::wday(Date)]

dc = na.omit(dc)

fwrite(dc, "cdat_R.csv")
gzip("cdat_R.csv", "cdat_R.csv.gz", overwrite=TRUE)
