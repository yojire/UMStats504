library(data.table)
library(splines)
library(ggplot2)

# If true, use both space and time covariates.  Otherwise use only the time covariates.
spacetime = TRUE

cdat = fread("zcat cdat_R.csv.gz")

# It seems that there may be some issues with the first two years.
cdat = cdat[Year >= 2003,]
cdat = cdat[CommunityArea > 0,]

cdat$DayOfWeek = factor(cdat$DayOfWeek, levels=c(1, 2, 3, 4, 5, 6, 7),
                           labels=c("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"))
cdat$CommunityArea = as.factor(cdat$CommunityArea)

primary_types = unique(cdat$PrimaryType)

if (spacetime) {
    pdf("chicago_timetrends_spacetime_R.pdf")
    bins = 10
} else {
    pdf("chicago_timetrends_time_R.pdf")
    bins = 30
}

# Fit a model for each type of crime.
for (pt in primary_types) {

    # The data for one type of crime.
    cda = cdat[cdat$PrimaryType==pt,]

    # In the time-only model, collapse over community areas.
    if (!spacetime) {
        cda = cda[, Num := sum(Num), by=Date]
        cda = cda[, head(.SD, 1), by=Date]
    }

    # Plot a histogram of the response value (number of reports per day/crime type/region).
    plt = ggplot(cda, aes(x=cda$Num)) + geom_histogram(bins=30) + xlab(sprintf("Reports of %s", pt))
    plt = plt + ylab("Frequency") + theme(plot.title=element_text(pt, hjust=0.5))
    print(plt)

    if (spacetime) {
        ft = glm(Num ~ bs(Year, 4) + bs(DayOfYear, 10) + C(DayOfWeek) + C(CommunityArea), family=poisson, data=cda)
    } else {
        ft = glm(Num ~ bs(Year, 4) + bs(DayOfYear, 10) + C(DayOfWeek), family=poisson, data=cda)
    }

	# Estimate the scale parameter to assess whether this is a Poisson
	# or a quasi-Poisson situation.
    sca = mean(resid(ft)^2)
    cat(sprintf("%-20s %10.2f\n", pt, sca))

    # Stratify the by deciles of the fitted responses,
    # calculate empirical mean and variance of the response
    # within each decile
    mf = data.table(fv=fitted.values(ft), y=cda$Num)
    mf[, rank := floor(10*frank(fv)/dim(mf)[1])]
    mf[, va := var(y), by=rank]
    mf[, mn := mean(y), by=rank]
    vm = mf[, head(.SD, 1), by=rank]
    vm[, y := NULL]
    vm[, fv := NULL]
    vm = vm[rank <= 9,]

    # Plot the mean/variance relationship
    vma = c(vm$mn, vm$va)
    range = c(min(vma), max(vma))
    plt = ggplot(vm, aes(x=mn, y=va)) + geom_point() + geom_abline() + xlim(range) + ylim(range)
    plt = plt + theme(plot.title=element_text(pt, hjust=0.5)) + xlab("Mean") + ylab("Variance")
    plt = plt + geom_smooth(method='lm', formula='y~x', se=FALSE)
    print(plt)

    for (tp in c("Year", "DayOfYear")) {
        for (vn in c("DayOfWeek", "CommunityArea")) {

            if ((!spacetime) & (vn == "CommunityArea")) {
                next
            }

            pda = list()
            for (u in unique(cda[[vn]])) {

                cdx = cda[1:100, ]
                cdx$Year = 2015
                cdx$Month = 6
                cdx$Day = 1
                cdx$DayOfYear = 180
                cdx$DayOfWeek = cda$DayOfWeek[1]
                cdx$CommunityArea = cda$CommunityArea[1]

                if (tp == "Year") {
                    cdx$Year = seq(2003, 2018, length.out=100)
                }
                if (tp == "DayOfYear") {
                    cdx$DayOfYear = seq(1, 365, length.out=100)
                }

                if (vn == "DayOfWeek") {
                    cdx$DayOfWeek = u
                }
                if (vn == "CommunityArea") {
                    cdx$CommunityArea = u
                }

                fv = predict(ft, newdata=cdx, se.fit=TRUE)
                pd = data.table(Year=cdx[[tp]], fitnum=fv$fit, se=fv$se*sqrt(sca))
                pd = pd[, lcb := exp(fitnum - 2.8*se)]
                pd = pd[, ucb := exp(fitnum + 2.8*se)]
                pd = pd[, efit := exp(fitnum)]
                pd = pd[, Group := u]

                pda[[length(pda)+1]] = pd
            }

            pd = rbindlist(pda)

            plt = ggplot(data=pd, aes(x=Year, y=efit))
            if (vn == "DayOfWeek") {
                plt = plt + geom_line(data=pd, aes(color=Group))
            } else {
                plt = plt + geom_line(data=pd, aes(color=Group))
            }

            if (vn == "DayOfWeek") {
                plt = plt + geom_ribbon(data=pd, aes(x=Year, ymin=lcb, ymax=ucb, fill=Group), alpha=0.3)
            }

            plt = plt + theme(plot.title=element_text(pt, hjust=0.5)) + xlab(tp)
            plt = plt + ylab(sprintf("Expected number of reports of %s", pt))

            print(plt)
        }
    }
}

dev.off()
