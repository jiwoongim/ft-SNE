"""
Code taken from https://github.com/hma02/thesne/blob/master/model/tsne.py
And then modified.
"""
import os, sys
import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from core import    kl_cost_var, reverse_kl_cost_var, js_cost_var, \
                    hellinger_cost_var, chi_square_cost_var, \
                    p_Yp_Y_var_np, floath, find_sigma

from utils import get_epsilon
from utils_sne import precision_K, K_neighbours, neighbour_accuracy_K, plot_map_c, plot_map_news


def tsne(X, perplexity=30, Y=None, output_dims=2, n_epochs=1000,
         initial_lr=1000, final_lr=50, lr_switch=250, init_stdev=1e-3,
         sigma_iters=50, initial_momentum=0.95, final_momentum=0.0, lrDecay=100,\
         momentum_switch=250, metric='euclidean', random_state=None,
         verbose=1, fname=None, color=None, divtype='kl', num_folds=2, datatype='mnist'):
    """Compute projection from a matrix of observations (or distances) using 
    t-SNE.
    
    Parameters
    ----------
    X : array-like, shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        Matrix containing the observations (one per row). If `metric` is 
        'precomputed', pairwise dissimilarity (distance) matrix.
    
    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.
        
    Y : array-like, shape (n_observations, output_dims), optional \
            (default = None)
        Matrix containing the starting position for each point.
    
    output_dims : int, optional (default = 2)
        Target dimension.
        
    n_epochs : int, optional (default = 1000)
        Number of gradient descent iterations.
        
    initial_lr : float, optional (default = 2400)
        The initial learning rate for gradient descent.
        
    final_lr : float, optional (default = 200)
        The final learning rate for gradient descent.
        
    lr_switch : int, optional (default = 250)
        Iteration in which the learning rate changes from initial to final.
        This option effectively subsumes early exaggeration.
        
    init_stdev : float, optional (default = 1e-4)
        Standard deviation for a Gaussian distribution with zero mean from
        which the initial coordinates are sampled.
        
    sigma_iters : int, optional (default = 50)
        Number of binary search iterations for target perplexity.
        
    initial_momentum : float, optional (default = 0.5)
        The initial momentum for gradient descent.
        
    final_momentum : float, optional (default = 0.8)
        The final momentum for gradient descent.
        
    momentum_switch : int, optional (default = 250)
        Iteration in which the momentum changes from initial to final.
        
    metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
        Indicates whether `X` is composed of observations ('euclidean') 
        or distances ('precomputed').
    
    random_state : int or np.RandomState, optional (default = None)
        Integer seed or np.RandomState object used to initialize the
        position of each point. Defaults to a random seed.

    verbose : bool (default = 1)
        Indicates whether progress information should be sent to standard 
        output.
        
    Returns
    -------
    Y : array-like, shape (n_observations, output_dims)
        Matrix representing the projection. Each row (point) corresponds to a
        row (observation or distance to other observations) in the input matrix.
    """
        
    N = X.shape[0]
    X_shared = theano.shared(np.asarray(X, dtype=floath))
    sigma_shared = theano.shared(np.ones(N, dtype=floath))
    find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, metric, verbose)

    sorted_ind_p, pdist = K_neighbours(X, sigma=sigma_shared.get_value(), maxK=10)
    rev_sorted_ind_p, pdist =  K_neighbours(X, maxK=100, revF=True, sigma=sigma_shared.get_value())

    figs_path   = './figs/'+datatype+'/'+divtype
    result_path = './results/'+datatype+'/'+divtype
    embedd_path = './embeddings/'+datatype+'/'+divtype
    if not os.path.exists(figs_path): os.makedirs(figs_path)
    if not os.path.exists(result_path): os.makedirs(result_path)
    if not os.path.exists(embedd_path): os.makedirs(embedd_path)
    np.save(result_path+'/'+datatype+'_probM_seed0_v2_perp'+str(perplexity), pdist)
    np.save(result_path+'/'+datatype+'_data_sorted_v2_seed0_perp'+str(perplexity), sorted_ind_p)

    for i in xrange(1,num_folds):
        print '%s FOLD %d' % (divtype, i)
        random_state = check_random_state(i)

        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
        Y_shared = theano.shared(np.asarray(Y, dtype=floath))

        Y = find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, \
                    n_epochs, initial_lr, final_lr, lr_switch, \
                    init_stdev, initial_momentum, final_momentum, \
                    momentum_switch, metric, sorted_ind_p, \
                    rev_sorted_ind_p, verbose, \
                    fname=fname+'_fold'+str(i), color=color, \
                    divtype=divtype, lrDecay=lrDecay, \
                    datatype=datatype)

    return Y


def find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
           initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
           final_momentum, momentum_switch, metric, sorted_ind_p, rev_sorted_ind_p,\
           verbose=0, fname=None, color=None, divtype='kl', lrDecay=100,\
           visLossF=0, naccuracyF=1, datatype='mnist'):

    """Optimize cost wrt Y"""
    # Optimization hyperparameters
    initial_lr  = np.array(initial_lr, dtype=floath)
    final_lr    = np.array(final_lr, dtype=floath)
    initial_momentum    = np.array(initial_momentum, dtype=floath)
    final_momentum      = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)


    X           = T.fmatrix('X')
    Y           = T.fmatrix('Y')
    Yv          = T.fmatrix('Yv')
    Yv_shared   = theano.shared(np.zeros((N, output_dims), dtype=floath))

    sigma       = T.fvector('sigma')
    momentum    = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Cost
    if divtype == 'kl':
        cost = kl_cost_var(X, Y, sigma, metric)
    elif divtype == 'rkl':
        cost = reverse_kl_cost_var(X, Y, sigma, metric)
    elif divtype == 'js':
        cost = js_cost_var(X, Y, sigma, metric)
    elif divtype == 'hl':
        cost = hellinger_cost_var(X, Y, sigma, metric)
    elif divtype == 'ch':
        cost = chi_square_cost_var(X, Y, sigma, metric)

    # Setting update for Y velocities
    grad_Y  = T.grad(cost, Y)
    norm_gs = abs(grad_Y).sum()
    updates = [(Yv_shared, momentum*Yv - lr*grad_Y)]
    givens = {X: X_shared, sigma: sigma_shared, Y: Y_shared, Yv: Yv_shared}
    update_Yv = theano.function([lr, momentum], [cost, norm_gs], givens=givens, updates=updates)
    Y_len  = T.mean(T.sum(Y**2, axis=1))

    # Setting update for Y
    get_y_i    = theano.function([], Y, givens={Y: Y_shared})
    get_cost_i = theano.function([Y], cost, givens={X: X_shared, sigma: sigma_shared})
    givens     = {Y: Y_shared, Yv: Yv_shared}
    updates    = [(Y_shared, Y + Yv)]
    update_Y   = theano.function([], Y_len, givens=givens, updates=updates)

    loss, gnorms = [], []
    for epoch in range(n_epochs):

        lrY = max(float(get_epsilon(initial_lr, lrDecay, epoch)), min(0.001, initial_lr))
        mom = float(get_epsilon(initial_momentum, lrDecay, epoch)) \
                                if epoch < momentum_switch else 0.85

        c, grad_len = update_Yv(lrY, mom)
        gnorms.append(grad_len)
        y_len = update_Y()
        loss.append(c)

        if verbose:
            projX = np.array(Y_shared.get_value())
            if epoch % 25 == 0  or epoch < 20:
                projX = np.array(Y_shared.get_value())
                np.save('./embeddings/'+fname, projX)

                ffname = './figs/'+fname+'_epoch'+str(epoch)+'.pdf'
                if datatype == 'sbow':
                    color_dict = [\
                      'lightblue', 'darkblue', 
                      'indianred', 'darkred', 'red', 'magenta', 'hotpink',
                      'silver', 'darkgray', 'gray']


                    plot_map_news(projX, color, color_dict, ffname)
                else:
                    plot_map_c(projX, color, ffname)


                print 'Epoch %d, SNE J %f, |GS| %f, |Y| %f, LRY %f, MOM %f' \
                              % (epoch, c, grad_len, y_len, lrY, mom)


    np.save('./results/'+fname+'_loss', np.asarray(loss))
    np.save('./results/'+fname+'_gnorm', np.asarray(gnorms))

    return np.array(Y_shared.get_value())



