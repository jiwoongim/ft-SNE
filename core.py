"""
Code taken from https://raw.githubusercontent.com/hma02/thesne/master/model/tsne.py 
And then modified.
"""

import theano.tensor as T
import theano
import numpy as np

import theano.sandbox.rng_mrg as RNG_MRG
import theano.tensor.shared_randomstreams  as RNG_TRG
from theano.tensor.shared_randomstreams import RandomStreams

RNG = np.random.RandomState(0)
MRG = RNG_MRG.MRG_RandomStreams(RNG.randint(2 ** 30))
TRG = RNG_TRG.RandomStreams(seed=1234)

epsilon = 1e-6
floath = np.float32


def sqeuclidean_var(X):

    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)

    return ss.reshape((N, 1)) + ss.reshape((1, N)) - 2*X.dot(X.T)


def discrete_sample(preds, num_sam, temperature=1.0):
    # function to sample an index from a probability array

    probas = TRG.choice(a=np.arange(3), size=[num_sam,], p=preds)
    return np.argmax(probas, axis=1)


def euclidean2_np(X):
    N = X.shape[0]
    ss = np.sum(X**2, axis=1)
    dist = np.reshape(ss, [N, 1]) + np.reshape(ss, [1, N]) - 2*np.dot(X, X.T)
    dist = dist * np.asarray(dist>0,'float32')
    return dist 


def p_Xp_given_X_np(X, sigma, metric, approxF=0):

    N = X.shape[0]
    if metric == 'euclidean':
        sqdistance = euclidean2_np(X)
    elif metric == 'precomputed':
        sqdistance = X**2
    else:
        raise Exception('Invalid metric')
    euc_dist     = np.exp(-sqdistance / (np.reshape(2*(sigma**2), [N, 1])))
    np.fill_diagonal(euc_dist, 0.0 )

    if approxF > 0:
        sorted_euc_dist = euc_dist[:,:]
        np.sort(sorted_euc_dist, axis=1)
        row_sum = np.reshape(np.sum(sorted_euc_dist[:,1:approxF+1], axis=1), [N, 1])
    else:
        row_sum = np.reshape(np.sum(euc_dist, axis=1), [N, 1])

    return euc_dist/row_sum  # Possibly dangerous


def p_Xp_given_X_var(X, sigma, metric):
    N = X.shape[0]

    if metric == 'euclidean':
        sqdistance = sqeuclidean_var(X)
    elif metric == 'precomputed':
        sqdistance = X**2
    else:
        raise Exception('Invalid metric')

    esqdistance = T.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    esqdistance_zd = T.fill_diagonal(esqdistance, 0)

    row_sum = T.sum(esqdistance_zd, axis=1).reshape((N, 1))

    return esqdistance_zd/row_sum  


def p_Xp_X_var(p_Xp_given_X):
    return (p_Xp_given_X + p_Xp_given_X.T) / 2.0


def p_Yp_Y_var(Y):
    N = Y.shape[0]
    sqdistance = sqeuclidean_var(Y)
    one_over = T.fill_diagonal(1/(sqdistance + 1), 0)
    p_Yp_given_Y =  one_over/one_over.sum(axis=1).reshape((N, 1))  
    return p_Yp_given_Y


def p_Yp_Y_var_np(Y):
    N = Y.shape[0]
    sqdistance = euclidean2_np(Y)
    one_over = 1./(sqdistance + 1)
    p_Yp_given_Y =  one_over/one_over.sum(axis=1).reshape((N, 1)) 
    return p_Yp_given_Y


def kl_cost_var(X, Y, sigma, metric):

    p_Xp_given_X = p_Xp_given_X_var(X, sigma, metric)
    PX = p_Xp_X_var(p_Xp_given_X)
    PY = p_Yp_Y_var(Y)

    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    return T.mean(T.sum(PX * T.log(PXc / PYc),-1))  


def reverse_kl_cost_var(X, Y, sigma, metric):

    p_Xp_given_X = p_Xp_given_X_var(X, sigma, metric)
    PX = p_Xp_X_var(p_Xp_given_X)
    PY = p_Yp_Y_var(Y)

    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    return -T.mean(T.sum(PY * T.log(PXc / PYc),-1))  

def js_cost_var(X, Y, sigma, metric):

    return kl_cost_var(X, Y, sigma, metric) * 0.5 + \
            reverse_kl_cost_var(X, Y, sigma, metric) * 0.5


def chi_square_cost_var(X, Y, sigma, metric):

    p_Xp_given_X = p_Xp_given_X_var(X, sigma, metric)
    PX = p_Xp_X_var(p_Xp_given_X)
    PY = p_Yp_Y_var(Y)

    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    return T.mean(T.sum(PY * (PXc / PYc - 1.)**2, -1))  


def hellinger_cost_var(X, Y, sigma, metric):

    p_Xp_given_X = p_Xp_given_X_var(X, sigma, metric)
    PX = p_Xp_X_var(p_Xp_given_X)
    PY = p_Yp_Y_var(Y)

    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    return T.mean(T.sum(PY * (T.sqrt(PXc / PYc) - 1.)**2,-1))  


def find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters,
        metric, verbose=0):
    """Binary search on sigma for a given perplexity."""
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')

    target = np.log(perplexity)

    P = T.maximum(p_Xp_given_X_var(X, sigma, metric), epsilon)

    entropy = -T.sum(P*T.log(P), axis=1)

    # Setting update for binary search interval
    sigmin_shared = theano.shared(np.full(N, np.sqrt(epsilon), dtype=floath))
    sigmax_shared = theano.shared(np.full(N, np.inf, dtype=floath))

    sigmin = T.fvector('sigmin')
    sigmax = T.fvector('sigmax')

    upmin = T.switch(T.lt(entropy, target), sigma, sigmin)
    upmax = T.switch(T.gt(entropy, target), sigma, sigmax)

    givens = {X: X_shared, sigma: sigma_shared, sigmin: sigmin_shared,
            sigmax: sigmax_shared}
    updates = [(sigmin_shared, upmin), (sigmax_shared, upmax)]

    update_intervals = theano.function([], entropy, givens=givens,
            updates=updates)

    # Setting update for sigma according to search interval
    upsigma = T.switch(T.isinf(sigmax), sigma*2, (sigmin + sigmax)/2.)

    givens = {sigma: sigma_shared, sigmin: sigmin_shared,
            sigmax: sigmax_shared}
    updates = [(sigma_shared, upsigma)]

    update_sigma = theano.function([], sigma, givens=givens, updates=updates)

    for i in range(sigma_iters):
        e = update_intervals()
        update_sigma()
        if verbose:
            print('Iteration: {0}.'.format(i+1))
            print('Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(e.min()),
                np.exp(e.max())))

            if np.any(np.isnan(np.exp(e))):
                raise Exception('Invalid sigmas. The perplexity is probably too low.')


def find_sigma_np(X, sigma, N, perplexity, sigma_iters, metric, verbose=1, approxF=0):

    """Binary search on sigma for a given perplexity."""
    target = np.log(perplexity)

    # Setting update for binary search interval
    sigmin = np.full(N, np.sqrt(epsilon), dtype='float32')
    sigmax = np.full(N, np.inf, dtype='float32')

    for i in range(sigma_iters):

        P = np.maximum(p_Xp_given_X_np(X, sigma, metric, approxF), epsilon)
        entropy = -np.sum(P*np.log(P), axis=1)
        minind = np.argwhere(entropy < target).flatten()
        maxind = np.argwhere(entropy > target).flatten()
        sigmin[minind] = sigma[minind]
        sigmax[maxind] = sigma[maxind]

        infmask = np.argwhere(np.isinf(sigmax)).flatten()
        old_sigma = sigma[infmask]
        sigma = (sigmin + sigmax)/2.
        sigma[infmask] = old_sigma*2


        if verbose:
            print('Iteration: {0}.'.format(i+1))
            print('Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(entropy.min()), np.exp(entropy.max())))

            if np.any(np.isnan(np.exp(entropy))):
                raise Exception('Invalid sigmas. The perplexity is probably too low.')


    return sigma

if __name__ == '__main__':
    asdf = discrete_sample(np.asarray([0.3,0.2,0.5]), 1000)
    import pdb; pdb.set_trace()
