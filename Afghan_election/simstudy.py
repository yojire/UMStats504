"""
This is a simulation study to explore some of the possible results
that could be obtained in a regression analysis of precinct-level vote
totals from an election that is held in two rounds.

The setting here is that we have four candidates, A, B, C and D, who
compete against each other in round 1 of the election. The top two
vote-getters from the first round then run against each other in round
2.  Suppose that these two candidates are A and B.

For simplicity, we use A, B, C, and D to refer to the four candidates
running in round 1, and to their numeric vote totals.  Let A' and B'
denote the number of votes received by candidates A and B in the
second round of the election.

The sample size here is large, so that we can focus on the expected
behavior of the results, not on variation or uncertainty.
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats.distributions import norm

# Number of precincts for which we observe per-candidate vote totals
n = 20000

# Round 1 vote counts per candidate
z = np.random.normal(size=(n, 4))

# Induce some correlations.  Candidate C's support is positively
# correlated with candidate A's, candidate D's support is negatively
# correlated with candidate A's.
r = 0.7
z[:, 2] = r*z[:, 0] + np.sqrt(1 - r**2)*z[:, 2]
z[:, 3] = -r*z[:, 1] + np.sqrt(1 - r**2)*z[:, 3]

# Put the simulated data on a more reasonable scale for vote totals
x = 1000 + 100 * z

"""
Within each precinct, the values of A' and B' are linear combinations
of A, B, C, and D.  Say the coefficients of this linear combination
for A' are b1, b2, b3, b4.  Thus, the value of A' is b1*A + b2*B +
b3*C + b4*D.

There is no reason that the coefficients b1, ..., b4 need to be the
same across all the precincts.  Below we explore what happens when we
allow the coefficients b1, ..., b4 vary from precinct to precinct.

Although the coefficients b1, ..., b4 vary by precinct below, we
continue to fit a model that uses the same coefficients for all
precincts.  What happens in this situation?  What do these
coefficients mean?

The main take-away from what we see below is that the regression
coefficients (calculated directly using linear modeling) do not always
agree with the mean of the precinct-level coefficients.  If the
coefficients vary in a way that is unrelated to the round 1 vote
totals, then the fitted regresion coefficients will be close to the
average of the precinct-level coefficients.  But if the coefficients
vary in a way that is related to the round 1 vote totals, more complex
behavior emerges.

For example, we see that if sgn1 = 1, then b1 is estimated to be much
greater than 1, even through the average true b1 is around 1.  This
happens in situations where the flow from C to A' is greater in
precincts that had larger values of A.
"""

# This parameter determines the randomness of the coefficients
# relating A' to A, B, C, D.  If f = 0, there is no randomness.
# Greater values of f induce greater randomness.
f = 0.5

# This parameter controls the extend to which the coefficients (b1,
# ..., b4) trend with the round 1 vote totals.  If sr is 0, there is
# no relationship.  Larger values of sr lead to b3 being correlated
# with A and b4 being correlated with B.  The directions of these
# correlations are determined by sgn1 and sgn2, defined below.
sr = 0.4

# If sgn1 is positive, precincts with more support for candidate A in
# round 1 will have more flow from C to A' in round 2.
for sgn1 in -1, 0, 1:

    # If sgn2 is positive, precincts with more support for candidate B
    # in round 1 will have more flow from D to A' in round 2.
    for sgn2 in -1, 0, 1:

        print(sgn1, sgn2)
        b = np.random.normal(size=(n, 4))
        b[:, 2] = sgn1*sr*z[:, 0] + np.sqrt(1 - sr**2)*b[:, 2]
        b[:, 3] = sgn2*sr*z[:, 1] + np.sqrt(1 - sr**2)*b[:, 3]

        # Add a mean to the coefficients.  The mean set here creates a
        # large average coefficient for A -> A', and a small average
        # coefficient for B -> A'.  The average coefficients from C
        # and D to A' will be around 0.5.
        b = f*b + np.r_[2, -1, 0, 0]

        # This is a "copula" construction.
        b = norm.cdf(b)

        # The average and SD of the coefficients
        print(b.mean(0))
        print(b.std(0))

        # Generate the round 2 vote totals A'
        y = (x * b).sum(1)

        # Fit the regression and display the results
        model = sm.OLS(y, x)
        results = model.fit()
        print(results.params, "\n")
