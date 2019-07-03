'''
This script shows how one can map samples that uniformly distributed in [0,1] to be distributed according to some analytical distributions via the inverse CDF (also called percent point function, PPF).

'''

from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# It all begins from MultiNest spitting out samples in [0,1] interval for each parameter. Let's look at how we could convert 1 of these samples. We'll actually do this on 100k samples to show how they distribute:
samples = np.random.random(100000)

# gamma distribution parameters: a is shape, b is rate (1/b is scale)
# in practice, for BSFC's h0 parameters we want to have an exponential prior (a=1) and choose an appropriate "scale" with the b parameter -- this should encourage recognition of small but non-negligible satellite lines
a=1.0
b=0.05   # variance is a/b^2


# Show mapping of samples from [0,1] interval
fig, ax = plt.subplots()

transformed_samples = gamma.ppf(samples, a, loc=0, scale=1.0 / b)
p,x = np.histogram(transformed_samples, bins=1000)

# convert bin edged to centers
x=x[:-1]+(x[1]-x[0])/2

# plot histogram of samples
ax.plot(x,np.array(p, dtype=float)/np.max(p))

# direct random sampling (may be useful for the emcee prior)
#r = gamma.rvs(a, size=1000)

# in case you want to know some distribution parameters:
mean,var,skew, kurt = gamma.stats(a, moments='mvsk', loc=0, scale=1./b)


#~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Nice visualization: plot quantiles:  ####
norm_factor = gamma.pdf(0.0, a,loc=0, scale=1./b)

# 1% and 99% quantiles
xx = np.linspace(gamma.ppf(0.01, a, loc=0, scale=1./b), gamma.ppf(0.99,a, loc=0, scale=1./b), 100)
vals = gamma.pdf(xx, a, loc=0, scale=1./b)
ax.plot(xx, vals/norm_factor, linewidth=2, label='1-99 quantile range')

# 10% to 90% quantiles
xx = np.linspace(gamma.ppf(0.1, a, loc=0, scale=1./b), gamma.ppf(0.9,a, loc=0, scale=1./b), 100)
vals = gamma.pdf(xx, a, loc=0, scale=1./b)
ax.plot(xx, vals/norm_factor, linewidth=2, label='10-90 quantile range')

# 25% to 75% quantiles
xx = np.linspace(gamma.ppf(0.25, a, loc=0, scale=1./b), gamma.ppf(0.75,a, loc=0, scale=1./b), 100)
vals = gamma.pdf(xx, a, loc=0, scale=1./b)
ax.plot(xx, vals/norm_factor, linewidth=2, label='25-75 quantile range')


plt.tight_layout()
ax.legend(fontsize=14).draggable()
