import numpy as np
import math
import random
from scipy.stats import beta
import matplotlib.pyplot as plt

alpha = [.1,.5,1,2,5]
beta_v = [2,2,2,2,2]
#lamda = 1./beta
#k = math.floor(alpha)

x = np.arange(0,1,0.01)
for i in range(len(alpha)):
    y = beta.pdf(x,alpha[i],beta_v[i])
    plt.plot(x,y,label='alpha='+str(alpha[i])+', beta='+str(beta_v[i]))
    plt.grid()
    plt.legend()
plt.show()
