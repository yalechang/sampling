import numpy as np
import random
import math
from scipy.stats import gamma

def sampling_cauchy(loc,scale,n_samples):
    """This function takes n_samples samples from the Cauchy distribution
    specified by Cauchy(loc,scale)
    The method used here is to take inverse function of the CDF of Cauchy
    distribution.
    For the details of cauchy distribution, please refer to the following link:
    mathworld.wolfram.com/CauchyDistribution.html

    Parameters
    ----------
    loc: float
        location parameter of cauchy distribution
    scale: float
        scale parameter of cauchy distribution
    n_samples: int
        the number of samples desired from this cauchy distribution

    Returns
    -------
    x_sampling: list, len(n_samples)
        list containing samples from cauchy distribution
    """
    x_sampling = [0]*n_samples
    for i in range(n_samples):
        x_sampling[i] = loc+scale*math.tan((random.random()-0.5)*math.pi)
    return x_sampling

def sampling_exp(lambda_exp,n_samples):
    """This function takes n_samples samples from the exponential distribution
    specified by lambda_exp
    The method used here is to take inverse function of CDF of exponential
    distribution

    Parameters
    ----------
    lambda_exp: float
        rate of change, parameter for exponential distribution
    n_samples: int
        the number of samples desired from the exponential distribution

    Returns
    -------
    x_sampling: list, len(n_samples)
        list containing samples from exponential distribution
    """
    x_sampling = [0]*n_samples
    for i in range(n_samples):
        x_sampling[i] = -math.log(1-random.random())*1./lambda_exp
    return x_sampling

def sampling_erlang(k,lambda_rate,n_samples):
    """This function takes n_samples from erlang distribution, which is a
    special case of Gamma distribution
    The method used here is to take the sum of k iid exponentially distributed
    rv

    Parameters
    ----------
    k: int
        shape parameter, the number of iid exponential rv that should be summed
        to get an erlang rv
    lambda_rate: float
        rate(inverse scale) parameter
    n_samples: int
        the number of samples desired from the erlang distribution

    Returns
    -------
    x_sampling: list, len(n_samples)
        list containing the samples
    """
    x_sampling = [0]*n_samples
    x_sampling_exp = np.zeros((k,n_samples))
    for i in range(k):
        x_sampling_exp[i,:] = sampling_exp(lambda_rate,n_samples)
    for j in range(n_samples):
        x_sampling[j] = sum(x_sampling_exp[:,j])
    return x_sampling

def sampling_rejection_gamma(alpha,beta,n_samples,dist_proposal='erlang',**kwargs):
    """This function take n_samples samples from gamma distribution.
    The method used here is rejection sampling. The most critical issue here is
    the choice of proposal distribution

    Parameters
    ----------
    alpha: float
        shape parameter for gamma distribution
    beta: float
        rate(inverse scale) parameter
    n_samples: int
        the number of samples desired from the erlang distribution
    dist_proposal: str
        the type of proposal distribution used in rejection sampling
        the value could be chosen from the set {'erlang','cauchy'}
    *args: 
        if dist_proposal == 'erlang', the parameters for erlang distribution
            could be provided here in the following keyword arguments form:
            k = val_k, lambda_rate = val_lambda_rate         
        if dist_proposal == 'cauchy', the parameters for cauchy distribution
            could be provided here in the following keyword arguments form:
            m= val_m, b = val_b

    Returns
    -------
    x_sampling: list, len(n_samples)
        list containing the samples from gamma distribution
    """
    assert dist_proposal in ['erlang','cauchy']

    # initialize the output
    x_sampling = []
    n_accept = 0
    if dist_proposal == 'erlang':
        k = kwargs['k']
        lambda_rate = kwargs['lambda_rate']
        x_peak = (alpha-k)*1./(beta-lambda_rate)
        ratio_max = gamma.pdf(x_peak,alpha,scale=1./beta)/gamma.pdf(x_peak,\
                k,scale=1./lambda_rate)
        
        """
        # Batch method
        x_sampling_erlang = sampling_erlang(k,lambda_rate,n_samples*2)
        for i in range(len(x_sampling_erlang)):
            temp = x_sampling_erlang[i]
            ratio = gamma.pdf(temp,alpha,scale=1./beta)/gamma.pdf(temp,k,\
                    scale=1./lambda_rate)
            if random.random() < ratio/ratio_max:
                x_sampling.append(temp)
        """
        
        # One-by-one method
        while n_accept < n_samples:
            temp = sampling_erlang(k,lambda_rate,1)
            ratio = gamma.pdf(temp,alpha,scale=1./beta)/gamma.pdf(temp,k,\
                    scale=1./lambda_rate)
            if random.random() < ratio/ratio_max:
                x_sampling.append(temp[0])
                n_accept += 1
    if dist_proposal == 'cauchy':
        m = kwargs['m']
        b = kwargs['b']
    
    #assert len(x_sampling) == n_accept

    return x_sampling,ratio_max

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import gamma
    import numpy as np
    n_samples = 2000
    x_sampling_erlang = sampling_erlang(5,1,n_samples)
    x_range = (0,20)
    x = np.arange(x_range[0],x_range[1],0.01)
    y = gamma.pdf(x,5,scale=1)
    plt.plot(x,y,'r')
    plt.hold
    plt.hist(x_sampling_erlang,bins=100,range=(0,20),normed=True)
    plt.show()
