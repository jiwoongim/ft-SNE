import os, sys, gzip, pickle, cPickle
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

    parser.add_argument('--datatype', type=str, default='mnist', \
                        choices=['mnist','mnist1','face','news'],
                        help='The name of dataset')
    parser.add_argument('--dataset_path', type=str, \
                        default='/groups/branson/home/imd/Documents/machine_learning_uofg/data/',\
                        help='Dataset directory')
    parser.add_argument('--divtypet', type=str, default='kl', \
                        choices=['kl','rkl','js','hl', 'ch'],
                        help='Choose your f-divergence')
    parser.add_argument('--perplexity_tsne', type=int, default=100, \
                        help='Perplexity')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    return args


if __name__ == '__main__':

    args            = parse_args()
    divtypet        = args.divtypet
    dataset_path    = args.dataset_path
    perplexity_tsne = args.perplexity_tsne

    if args.datatype == 'mnist':
        dataset_path = dataset_path + '/MNIST/mnist.pkl.gz'
        f = gzip.open(dataset_path, 'rb')
        train_set_np, valid_set_np, test_set_np = cPickle.load(f)

        ind0 = np.argwhere(train_set_np[1] == 0).flatten()
        ind1 = np.argwhere(train_set_np[1] == 1).flatten()
        ind2 = np.argwhere(train_set_np[1] == 2).flatten()
        ind3 = np.argwhere(train_set_np[1] == 4).flatten()
        ind4 = np.argwhere(train_set_np[1] == 5).flatten()
        ind  = np.concatenate([ind0, ind1, ind2, ind3, ind4])

        data = train_set_np[0][ind]
        label= train_set_np[1][ind]
        pca = PCA(n_components=30)
        pcastr = 'pca30_5class'

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

    elif args.datatype == 'mnist1':
        dataset_path = dataset_path + '/MNIST/mnist.pkl.gz'
        f = gzip.open(dataset_path, 'rb')
        train_set_np, valid_set_np, test_set_np = cPickle.load(f)

        ind = np.argwhere(train_set_np[1] == 1).flatten()

        data = train_set_np[0][ind]
        label= train_set_np[1][ind]
        pca = PCA(n_components=30)
        pcastr = 'pca30_1class'
        data = pca.fit(data).transform(data)
        perm = RNG.permutation(data.shape[0])
        data = data [perm][:5000]
        color= label[perm][:5000]


        initial_momentum=0.5; momentum_switch=200
        n_epochs_tsne=200; 
        if divtypet=='hl':
            initial_lr_tsne=300
            lrDecay=100
        elif divtypet=='ch':
            initial_lr_tsne=5; 
            momentum_switch=1
            lrDecay=100
        elif divtypet=='rkl':
            initial_lr_tsne=1000; 
            lrDecay=100
        elif divtypet=='js':
            initial_lr_tsne=1000; 
            lrDecay=100
        else:
            initial_lr_tsne=1000 
            lrDecay=100

    elif args.datatype == 'face':
      
        import scipy.io as sio
        mat_contents = sio.loadmat(dataset_path+'/embedding_data/face_data.mat')
        data = mat_contents['images'].T
        light = (mat_contents['lights'].T - mat_contents['lights'].T.min()) / mat_contents['lights'].T.max()
        poses = (mat_contents['poses'].T - mat_contents['poses'].T.min()) / (mat_contents['poses'].T.max() - mat_contents['poses'].T.min())
        color = poses[:,0] 
        n_epochs_tsne=1000; 
        pcastr = 'pose1'
        if divtypet=='hl':
            initial_momentum=0.5
            initial_lr_tsne=100
            momentum_switch=100
            lrDecay=10.0
        elif divtypet=='ch':
            initial_momentum=0.5
            initial_lr_tsne=100
            momentum_switch=100
            lrDecay=10
        elif divtypet=='rkl':
            initial_momentum=0.5
            initial_lr_tsne=1000; 
            momentum_switch=25
            lrDecay=50
        elif divtypet=='js':
            initial_momentum=0.5
            initial_lr_tsne=1000;
            momentum_switch=200
            lrDecay=100
        else:
            initial_momentum=0.5
            initial_lr_tsne=1000 
            momentum_switch=200
            lrDecay=100

    elif args.datatype == 'news':

        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', \
                      'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', \
                      'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
        newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
        vectorizer = TfidfVectorizer()
        data = vectorizer.fit_transform(newsgroups_train.data).todense().astype('float32')

        color = newsgroups_train.target
        pca = PCA(n_components=30)
        pcastr = '_pca30_3hier'

        data = pca.fit(data).transform(data)
        data, color = data[:6000], color[:6000]
        data = data / (data.max()-data.min()) 
        n_epochs_tsne=300; 
        if divtypet=='hl':
            initial_momentum=0.5
            initial_lr_tsne=100
            momentum_switch=200
            lrDecay=5
        elif divtypet=='ch':
            initial_momentum=0.5
            initial_lr_tsne=2000 
            momentum_switch=200
            lrDecay=100
        elif divtypet=='rkl':
            initial_momentum=0.5
            initial_lr_tsne=1000
            momentum_switch=100
            lrDecay=25
        elif divtypet=='js':
            initial_momentum=0.5
            initial_lr_tsne=3000;
            momentum_switch=200
            lrDecay=100
        else:
            initial_momentum=0.5
            initial_lr_tsne=1500
            momentum_switch=200
            lrDecay=100


    print 'Divtype %s, Perplexity %d' % (divtypet, perplexity_tsne)
    fname = args.datatype+'/'+divtypet+'/tsne_'+str(perplexity_tsne)+'perp'+str(n_epochs_tsne)+'epoch_initlr'+str(initial_lr_tsne)+pcastr
    projX = tsne(data, 
                    initial_lr=initial_lr_tsne, \
                    final_lr=initial_lr_tsne,\
                    lrDecay=lrDecay,\
                    initial_momentum=initial_momentum,\
                    momentum_switch=momentum_switch,\
                    perplexity=perplexity_tsne, \
                    n_epochs=n_epochs_tsne, fname=fname, \
                    color=color, divtype=divtypet, args.datatype=args.datatype)

    print(fname)
    pass



