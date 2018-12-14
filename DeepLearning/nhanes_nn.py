import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from nhanes_data import dx
import patsy

# Standardize all the variables that we will be using
for v in ("BPXSY1", "RIDAGEYR", "BMXBMI"):
    dx[v] -= dx[v].mean()
    dx[v] /= dx[v].std()

# Get a design matrix for the regression
yp, xp = patsy.dmatrices("BPXSY1 ~ 0 + RIDAGEYR + BMXBMI", data=dx, return_type='dataframe')

# Standardize everything
yp -= yp.mean(0)
yp /= yp.std()
xp -= xp.mean(0)
xp /= xp.std(0)

# Fit a least squares regression
model = sm.GLM(yp, xp)
result = model.fit()
result.summary(xname=xp.columns.tolist())

y = torch.tensor(yp.astype(np.float32).values)
x = torch.tensor(xp.astype(np.float32).values)

pdf = PdfPages("nhanes_nn.pdf")

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=xp.shape[1], n_hidden=20, n_output=1)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


for t in range(200):

    prediction = net(x)                # input x and predict based on x
    loss = loss_func(prediction, y)    # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        plt.clf()
        ax = plt.axes()
        plt.grid(True)
        plt.title("Training step %d" % t)
        plt.plot(y.data.numpy(), prediction.data.numpy(), 'o', alpha=0.5, rasterized=True)
        plt.xlabel("Observed SBP-Z", size=16)
        plt.ylabel("Predicted SBP-Z", size=16)
        plt.text(0.6, 0.08, 'Loss=%.4f' % loss.data.numpy(),
            fontdict={'size': 14, 'color': 'black'}, transform=ax.transAxes)
        r = np.corrcoef(y.data.numpy().flat, prediction.data.numpy().flat)[0, 1]
        plt.text(0.6, 0.02, 'Correlation=%.4f' % r,
            fontdict={'size': 14, 'color': 'black'}, transform=ax.transAxes)
        pdf.savefig()


# Make plots of the fitted values against each covariate
for k, na in enumerate(xp.columns):
    plt.clf()
    ax = plt.axes()
    plt.grid(True)
    plt.plot(xp[na], prediction.data.numpy(), 'o', alpha=0.5, rasterized=True)
    plt.xlabel(na, size=16)
    plt.ylabel("Predicted SBP-Z", size=16)
    pdf.savefig()

# Make plot of the fitted neural net function against each covariate,
# holding the other covariate fixed at -1, 0, 1, SD relative to the mean.
for k, na in enumerate(xp.columns):
    plt.clf()
    ax = plt.axes()
    plt.grid(True)
    for b in -1, 0, 1:
        xx = xp.copy().values
        for j in range(xx.shape[1]):
            if j != k:
                xx[:, j] = xx[:, j].mean() + b
            else:
                xx[:, j] = np.linspace(xx[:, j].min(), xx[:, j].max(), xx.shape[0])
        yy = net(torch.tensor(xx.astype(np.float32)))
        yy = yy.detach().numpy()
        plt.plot(xx[:, k], yy, '-', lw=4)
    plt.xlabel(na, size=16)
    plt.ylabel("Predicted SBP-Z", size=16)
    pdf.savefig()

# Make added variable plots for each covariate
from statsmodels.graphics.regressionplots import add_lowess
for k, na in enumerate(xp.columns):
    plt.clf()
    ax = plt.axes(rasterized=True)
    plt.grid(True)
    result.plot_added_variable(na, ax=ax)
    add_lowess(ax)
    pdf.savefig()

pdf.close()