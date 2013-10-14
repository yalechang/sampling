"""This script is an implementation of the example on Page 818 of Kevin
Murphy's book <Machine Learning-A Probabilistic Perspective>
Rejection sampling of Gamma distribution
Use standard notation here, the parameter for gamma distribution is alpha and
beta.
"""
#print __doc__

import numpy as np
import math
import random
from scipy.stats import gamma
import matplotlib.pyplot as plt

alpha = [2,3,4,5]
beta = [2,2,2,2]
#lamda = 1./beta
#k = math.floor(alpha)

x = np.arange(0,5,0.01)
for i in range(len(alpha)):
    y = gamma.pdf(x,alpha[i],scale=1.0/beta[i])
    plt.plot(x,y,label='alpha='+str(alpha[i])+', beta='+str(beta[i]))
    plt.grid()
    plt.legend()
plt.show()
