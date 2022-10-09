import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from utils import unpickle
from core import p_Xp_given_X_np, p_Yp_Y_var_np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_map_news(xx, colors, color_dict, fname):

    plt.figure()
    ax = plt.subplot(111)

    area = np.pi * 4 #* (15 * np.random.rand(N))**2  # 0 to 15 point radii

    for i, x  in enumerate(xx):
        plt.scatter(x[0], x[1], s=area, c=color_dict[colors[i]], alpha=0.7, facecolor='0.8', lw = 0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1., box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                  fancybox=True, shadow=True, ncol=3)
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', format='pdf')


def plot_map_c(xx, colors, fname):

    plt.figure()
    ax = plt.subplot(111)

    area = np.pi * 4 #* (15 * np.random.rand(N))**2  # 0 to 15 point radii
    plt.scatter(xx[:,0], xx[:,1], s=area, c=colors, alpha=1.0, cmap=plt.cm.Spectral, \
                    facecolor='0.5', lw = 0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1., box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                  fancybox=True, shadow=True, ncol=3)
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', format='pdf')


def plot1D(xx, colors, fname):

    plt.figure()
    ax = plt.subplot(111)

    area = np.pi * 5 #* (15 * np.random.rand(N))**2  # 0 to 15 point radii
    dummy = np.zeros_like(xx)
    plt.scatter(xx, dummy, s=area, c=colors, alpha=0.9, cmap=plt.cm.Spectral, facecolor='0.5', lw = 0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1., box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                  fancybox=True, shadow=True, ncol=3)

    plt.savefig(fname, bbox_inches='tight', format='pdf')



def plot3D(xx, colors, fname):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    area = np.pi *5 #* (15 * np.random.rand(N))**2  # 0 to 15 point radii
    ax.scatter(xx[:,0], xx[:,1], xx[:,2], c=colors, s=area, alpha=0.5, cmap=plt.cm.Spectral, \
                    facecolor='0.5', lw = 0)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1., box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                  fancybox=True, shadow=True, ncol=3)
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', format='pdf', transparent=True)


def precision_K(p_sorted_ind, q_sorted_ind, Ks, K=3):

    p_sorted_ind = p_sorted_ind[:, :K]
    q_sorted_ind = q_sorted_ind[:, :K]
    N = p_sorted_ind.shape[0]

    accuracy = np.zeros((N,len(Ks)))

    # For each point in x compute the distance of K points in P and Q
    for j,kk in enumerate(Ks):
        for i in range(N):
            for k in range(kk):
                ind_k = q_sorted_ind[i, k]
                tmp_k = np.argwhere(ind_k == p_sorted_ind[i,:kk]).flatten()
                if tmp_k.shape[0] > 0:
                    accuracy[i,j] += 1.0

    # Count the number of correct indices  
    outputs = []
    for jj in  range(len(Ks)):
        outputs += [[np.mean(accuracy[:,jj]), np.std(accuracy[:,jj])]]

    return outputs


def K_neighbours(data, maxK=10, revF=False, sigma=None):

    from utils import dist2hy_np
    if sigma is not None:
        dists = p_Xp_given_X_np(data, sigma, 'euclidean')
    else:
        dists = p_Yp_Y_var_np(data)
    N, _ = dists.shape
    sorted_ind_p = np.zeros((N,maxK), dtype='int32')

    for i in range(N):sorted_ind_p[i,:] = np.argsort(dists[i,:])[1:maxK+1]
    if revF: sorted_ind_p  = sorted_ind_p[:,::-1]

    return sorted_ind_p, dists


def neighbour_accuracy_K(data, labels, Ks, maxK=10):

    N, _ = data.shape

    fractions = []
    for i in range(N):

        ind_sort = data[i,:]
        label           = labels[i]
        neighbor_labels = labels[ind_sort]
        fraction = np.asarray(neighbor_labels == label) * 1.0
        fractions.append(fraction)

    fractions = np.asarray(fractions)
    output = []
    for K in  Ks:
        output += [np.mean(np.sum(fractions[:,:K], axis=1) / K), \
                    np.std(np.sum(fractions[:,:K], axis=1) / K)]

    return output


def get_iris_data():

    data, label = [], []
    f = open('/groups/branson/home/imd/Documents/data/embedding_data/iris.txt', 'r')
    line = f.readline()
    data.append(line[:-1])
    label.append(line[-1])
    while line.strip() != '':
        line = f.readline()
        data.append(line[:-1])
        label.append(line[-1])

    return np.asarray(data), np.asarrya(label)
