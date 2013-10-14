"""The notation used in this script comes from the following link:
mathworld.wolfram.com/CauchyDistribution.html

Cauchy distribution describes the distribution of horizontal distances at which
a line segment tilted at a random angle cuts the x-axis
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import cauchy
from sampling_common_distribution import sampling_cauchy
from scipy.stats import gamma
from sampling_common_distribution import sampling_exp
from sampling_common_distribution import sampling_erlang
from sampling_common_distribution import sampling_rejection_gamma
from time import time

# Parameter for Gamma distribution, which is the target distribution
alpha = 5.7
beta = 2
n_samples = 1000

# Parameter for Proposal distribution, we use erlang distribution here
# Note that erlang distribution is a special case of Gamma distribution
# If we alpha to be integer, then we can get erlang distribution
k = 5
lambda_rate = 0.5

t0 = time()
# Sampling Gamma using rejection sampling
x_sampling_gamma,ratio_max = sampling_rejection_gamma(alpha,beta,\
        n_samples,dist_proposal='erlang',k=k,lambda_rate=lambda_rate)
t1 = time()
print(["Running Time:",(t1-t0)/60])

x_range = (0.,30.)
x = np.arange(x_range[0],x_range[1],(x_range[1]-x_range[0])/n_samples)
# Draw histogram of x_sampling
y = gamma.pdf(x,alpha,scale=1./beta)
y_proposal = gamma.pdf(x,k,scale=1./lambda_rate)*ratio_max
plt.plot(x,y,'r')
plt.hold
plt.plot(x,y_proposal,'k-')
plt.hist(x_sampling_gamma,bins=n_samples/5,range=x_range,normed=True)
plt.show()

