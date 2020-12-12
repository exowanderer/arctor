import numpy as np
from arctor.dwt import dwt_chisq

# Compute chi-squared for a given model fitting a data set:
data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])
model = np.ones(8)
params = np.array([1.0, 0.1, 0.1])
chisq = dwt_chisq(model, data, params)
print(chisq)
1693.22308882
# Now, say this is a three-parameter model, with a Gaussian prior
# on the last parameter:
priors = np.array([1.0, 0.2, 0.3])
plow = np.array([0.0, 0.0, 0.1])
pup = np.array([0.0, 0.0, 0.1])
chisq = dwt_chisq(model, data, params, priors, plow, pup)
print(chisq)
