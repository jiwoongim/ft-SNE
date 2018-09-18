import os, sys, gzip, pickle, cPickle, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np
from tsne import tsne

from utils import unpickle, plot_map
from utils_sne import precision_K, K_neighbours

from sklearn.decomposition import PCA
RNG = np.random.RandomState(0)

def parse_args():
    desc = "Pytorch implementation of AAE collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--dataset_path', type=str, \
                        default='./data/',\
                        help='Dataset directory')
    parser.add_argument('--divtypet', type=str, default='kl', \
                        choices=['kl','rkl','js','hl', 'ch'],
                        help='Choose your f-divergence')
    parser.add_argument('--perplexity_tsne', type=int, default=100, \
                        help='Perplexity')

    return parser.parse_args()



if __name__ == '__main__':

    args            = parse_args()
    divtypet        = args.divtypet
    dataset_path    = args.dataset_path
    perplexity_tsne = args.perplexity_tsne

    dataset_path = dataset_path 
    data  = np.load(dataset_path+'/data.npy')
    label = np.load(dataset_path+'/label.npy')
    datatype='mydata'

    pca = PCA(n_components=30)
    pcastr = 'pca30'

    data = pca.fit(data).transform(data)
    perm = RNG.permutation(data.shape[0])
    data = data [perm][:6000]
    color= label[perm][:6000]
    initial_momentum=0.5
    n_epochs_tsne=2000; 
    if divtypet=='hl':
        initial_lr_tsne=300
        momentum_switch=200
        lrDecay=100
    elif divtypet=='ch':
        initial_lr_tsne=10;
        momentum_switch=200
        lrDecay=100
    elif divtypet=='rkl':
        initial_lr_tsne=1000; 
        momentum_switch=200
        lrDecay=100
    elif divtypet=='js':
        initial_lr_tsne=1000;
        momentum_switch=200
        lrDecay=100
    else:
        initial_lr_tsne=2500
        momentum_switch=200
        lrDecay=100

    print 'Divtype %s, Perplexity %d' % (divtypet, perplexity_tsne)
    fname = '/'+datatype+'/'+divtypet+'/tsne_'+str(perplexity_tsne)+'perp'+str(n_epochs_tsne)+'epoch_initlr'+str(initial_lr_tsne)+pcastr
    projX = tsne(data, 
                 initial_lr=initial_lr_tsne, \
                 final_lr=initial_lr_tsne,\
                 lrDecay=lrDecay,\
                 initial_momentum=initial_momentum,\
                 momentum_switch=momentum_switch,\
                 perplexity=perplexity_tsne, \
                 n_epochs=n_epochs_tsne, fname=fname, \
                 color=color, divtype=divtypet, datatype=datatype)

    print(fname)
    pass



