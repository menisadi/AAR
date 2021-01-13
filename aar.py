import numpy as np

def exponential_mechanism(data, domain, quality_function, eps, bulk=False, for_sparse=False):
    """Exponential Mechanism
    exponential_mechanism ( data , domain , quality function , privacy parameter )
    :param data: list or array of values
    :param domain: list of possible results
    :param quality_function: function which get as input the data and a domain element and 'qualifies' it
    :param eps: privacy parameter
    :param bulk: in case that we can reduce run-time by evaluating the quality of the whole domain in bulk,
    the procedure will be given a 'bulk' quality function. meaning that instead of one domain element the
    quality function get the whole domain as input
    :param for_sparse: in cases that the domain is a very spared one, namely a big percent of the domain has quality 0,
    there is a special procedure called sparse_domain. That procedure needs, beside that result from the given
    mechanism, the total weight of the domain whose quality is more than 0. If that is the case Exponential-Mechanism
    will return also the P DF before the normalization.
    :return: an element of domain with approximately maximum value of quality function
    """

    # calculate a list of probabilities for each element in the domain D
    # probability of element d in domain proportional to exp(eps*quality(data,d)/2)
    if bulk:
        qualified_domain = quality_function(data, domain)
        domain_pdf = [np.exp(eps * q / 2) for q in qualified_domain]
    else:
        domain_pdf = [np.exp(eps * quality_function(data, d) / 2) for d in domain]
    total_value = float(sum(domain_pdf))
    domain_pdf = [d / total_value for d in domain_pdf]
    normalizer = sum(domain_pdf)
    # for debugging and other reasons: check that domain_cdf indeed defines a distribution
    # use the uniform distribution (from 0 to 1) to pick an elements by the CDF
    if abs(normalizer - 1) > 0.001:
        raise ValueError('ERR: exponential_mechanism, sum(domain_pdf) != 1.')

    # accumulate elements to get the CDF of the exponential distribution
    domain_cdf = np.cumsum(domain_pdf).tolist()
    # pick a uniformly random value on the CDF
    pick = np.random.uniform()

    # return the index corresponding to the pick
    # take the min between the index and  len(D)-1 to prevent returning index out of bound
    result = domain[min(np.searchsorted(domain_cdf, pick), len(domain)-1)]
    # in exponential_mechanism_sparse we need also the total_sum value
    if for_sparse:
        return result, total_value
    return result


def generate_labeled_gaussians(size, dimension, scale=100):
    sample = np.random.normal(0,scale,(size,dimension)).astype(int)
    sample = sample - tuple(sample[:,i].min() for i in range(dimension))
    thresholds = [np.random.uniform(min(sample[:,i]), max(sample[:,i]), 1) for i in range(dimension)]
    positives = np.array([x for x in sample if all(x[i] <= thresholds[i] for i in range(dimension))])
    negatives = np.array([x for x in sample if any(x[i] > thresholds[i] for i in range(dimension))])
    return positives, negatives

def generate_labeled(size, dimension, trials=1000, pvals=0.5):
    ps = pvals*dimension
    sample = np.array([np.random.binomial(t,pvals,size) for t in [trials]*dimension])
    sample = np.column_stack((sample))
    sample = sample - tuple(sample[:,i].min() for i in range(dimension))
    mins = [min(sample[:,i]) for i in range(dimension)]
    maxs = [max(sample[:,i]) for i in range(dimension)]
    thresholds = [np.random.uniform(int(mins[i]/3+2*maxs[i]/3), int(mins[i]/3+2*maxs[i]/3), 1) for i in range(dimension)]
    positives = np.array([x for x in sample if all(x[i] <= thresholds[i] for i in range(dimension))])
    negatives = np.array([x for x in sample if any(x[i] > thresholds[i] for i in range(dimension))])
    return positives, negatives
    
def aar(data, domain,  dimension, margins_size=0, beta=0.1, eps=0.5, delta=0.1, t=0.4, test=False):
    def q(data, x):
        if min(data) <= x <= max(data):
           return 0
        else:
           return min(data[data <= x].shape[0], data[data >= x].shape[0])

    picks = np.zeros(dimension)
    d = data
    if not margins_size:
        margins_size = 150*int(np.log(100*domain**2))
    if test:        
        interiors = [np.array(1) for _ in range(dimension)]
        margins = [np.array(1) for _ in range(dimension)]
        
    for i in range(dimension):
        noisy_margin = int(np.ceil(np.random.laplace(margins_size, margins_size*t, 1)))
        margin = d[np.argpartition(d[:,i],(-1)*noisy_margin)][(-1)*noisy_margin:]
        tnoise = int(t**2 * noisy_margin)
        interior = margin[np.argpartition(margin[:,i],tnoise)][:tnoise]
        pick = exponential_mechanism(interior[:,i], np.arange(0,domain), q, eps)
        new_d = np.array([x for x in d if x[i] < pick])
        d = new_d
        picks[i] = pick
        if test:
            margins[i] = margin
            interiors[i] = interior            
        
    if test:
        return picks, margins, interiors
    else:
        return picks

def false_negatives(positives, points):
    return sum(1 for x in positives if any(x[i] > points[i]
                                           for i in range(len(points))))  / positives.shape[0]


