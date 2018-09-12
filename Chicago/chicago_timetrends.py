import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import patsy
from chicago_data import cdat

spacetime = True

cdat = cdat.loc[pd.notnull(cdat.DayOfWeek), :]
cdat = cdat.loc[cdat.Year >= 2003, :]

if not spacetime:
    first = lambda x: x.iloc[0]
    cdat = cdat.groupby(["PrimaryType", "Year", "DayOfYear"]).agg({"Num": np.sum, "DayOfWeek": first})
    cdat = cdat.reset_index()

if spacetime:
    pdf = PdfPages("chicago_timetrends_spacetime.pdf")
    fml = "Num ~ bs(Year, 4) + C(DayOfWeek) + C(CommunityArea) + bs(DayOfYear, 10)"
else:
    pdf = PdfPages("chicago_timetrends_time.pdf")
    fml = "Num ~ bs(Year, 4) + C(DayOfWeek) + bs(DayOfYear, 10)"

opts = {"DayOfWeek": {"lw": 3}, "CommunityArea": {"color": "grey", "lw": 2, "alpha": 0.5}}

# Loop over primary crime types, create a model for each type
for pt, dz in cdat.groupby("PrimaryType"):

    # Create and fit the model
    model = sm.GLM.from_formula(fml, family=sm.families.Poisson(), data=dz)
    result = model.fit(missing='drop')

    # Estimate the scale as if this was a quasi-Poisson model
    scale = np.mean(result.resid_pearson**2)
    print("%-20s %5.1f" % (pt, scale))

    # Get the empirical mean and variance of the response variable
    # in a series of fitted value strata.
    c = pd.qcut(result.fittedvalues, np.linspace(0.1, 0.9, 9))
    dd = pd.DataFrame({"c": c, "y": model.endog})
    mv = []
    for k,v in dd.groupby("c"):
        mv.append([v.y.mean(), v.y.var()])
    mv = np.asarray(mv)

    # Histogram of counts
    plt.clf()
    plt.axes([0.15, 0.1, 0.8, 0.8])
    plt.hist(model.endog)
    plt.xlabel(pt, size=15)
    plt.ylabel("Frequency", size=15)
    pdf.savefig()

    # Plot the empirical mean/variance relationship
    plt.clf()
    plt.title(pt)
    plt.grid(True)
    plt.plot(mv[:, 0], mv[:, 1], 'o', color='orange')
    mx = mv.max()
    mx *= 1.04
    b = np.dot(mv[:, 0], mv[:, 1]) / np.dot(mv[:, 0], mv[:, 0])
    plt.plot([0, mx], [0, b*mx], color='purple')
    plt.plot([0, mx], [0, mx], '-', color='black')
    plt.xlim(0, mx)
    plt.ylim(0, mx)
    plt.xlabel("Mean", size=15)
    plt.ylabel("Variance", size=15)
    pdf.savefig()

    # Plot the fitted means curves for two variables, holding the others fixed.
    for tp in "Year", "DayOfYear":
        for vn in "DayOfWeek", "CommunityArea":

            if vn == "CommunityArea" and not spacetime:
                continue

            p = 100
            dp = cdat.iloc[0:p, :].copy()
            dp.Year = 2015
            dp.DayOfWeek = "Su"
            dp.CommunityArea = 1
            dp.DayOfYear = 180

            if tp == "Year":
                dp.Year = np.linspace(2003, 2018, p)
            elif tp == "DayOfYear":
                dp.DayOfYear = np.linspace(1, 366, p)

            plt.clf()

            if vn == "DayOfWeek":
                plt.axes([0.15, 0.1, 0.72, 0.8])

            plt.grid(True)

            for u in dz[vn].unique():
                dp[vn] = u
                pr = result.predict(exog=dp)
                if vn == "DayOfWeek":
                    plt.plot(dp[tp], pr, '-', label=u, **opts[vn])
                else:
                    plt.plot(dp[tp], pr, '-', **opts[vn])

            plt.xlabel(tp, size=14)
            plt.ylabel("Expected number of reports per day", size=16)
            plt.title(pt + " (by %s)" % vn)

            if vn == "DayOfWeek":
                ha, lb = plt.gca().get_legend_handles_labels()
                leg = plt.figlegend(ha, lb, "center right")
                leg.draw_frame(False)

            pdf.savefig()

            # Plot with confidence bands
            di = model.data.design_info
            cm = scale * result.cov_params()

            if vn == "CommunityArea":
                continue

            # Plot with error bands
            plt.clf()
            plt.axes([0.15, 0.1, 0.72, 0.8])
            plt.grid(True)
            for u in dz[vn].unique():
                dp[vn] = u
                pr = result.predict(exog=dp)
                lpr = np.log(pr)
                xm = patsy.dmatrix(di, dp)
                dm = np.dot(xm, np.dot(cm, xm.T))
                se = np.sqrt(np.diag(dm))

                plt.plot(dp[tp], pr, '-', label=u, **opts[vn])
                lcb = np.exp(lpr - 2.8*se)
                ucb = np.exp(lpr + 2.8*se)
                plt.fill_between(dp[tp], lcb, ucb, color='lightgrey')

            plt.xlabel(tp, size=14)
            plt.ylabel("Expected number of reports per day", size=16)
            plt.title(pt + " (by %s)" % vn)
            ha, lb = plt.gca().get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, "center right")
            leg.draw_frame(False)
            pdf.savefig()

            # Contrast plot with error bands
            plt.clf()
            plt.axes([0.15, 0.1, 0.78, 0.8])
            plt.grid(True)
            lpra, xma = [], []
            for u in ("Su", "Mo"):
                dp[vn] = u
                pr = result.predict(exog=dp)
                lpr = np.log(pr)
                lpra.append(lpr)
                xm = patsy.dmatrix(di, dp)
                xma.append(xm)

            xm = xma[0] - xma[1]
            lpr = lpra[0] - lpra[1]
            dm = np.dot(xm, np.dot(cm, xm.T))
            se = np.sqrt(np.diag(dm))

            lcb = np.exp(lpr - 2.8*se)
            ucb = np.exp(lpr + 2.8*se)
            plt.plot(dp[tp], np.exp(lpr), '-', label=u, **opts[vn])
            plt.fill_between(dp[tp], lcb, ucb, color='lightgrey')

            plt.xlabel(tp, size=14)
            plt.ylabel("Sunday / Monday ratio", size=16)
            plt.title(pt + " (by %s)" % vn)
            pdf.savefig()

pdf.close()

