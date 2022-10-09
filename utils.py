''' Version 1.000
 Code provided by Daniel Jiwoong Im 
 Permission is granted for anyone to copy, use, modify, or distribute this
 program and accompanying programs and documents for any purpose, provided
 this copyright notice is retained and prominently displayed, along with
 a note saying that the original programs are available from our
 web page.
 The programs and documents are distributed without any warranty, express or
 implied.  As the programs were written for research purposes only, they have
 not been tested to the degree that would be advisable in any important
 application.  All use of these programs is entirely at the user's own risk.'''


import math #, PIL
import matplotlib as mp
import matplotlib.pyplot as plt

import numpy as np
from numpy.lib import stride_tricks

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.rng_mrg as RNG_MRG
#from subnets.layers.utils import rng
rng = np.random.RandomState(0)
MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))


def plot_map(xx, fname):

    plt.figure()
    ax = plt.subplot(111)

    area = np.pi 
    colors = cm.rainbow(np.linspace(0, 1, len(xx)))
    for i, x  in enumerate(xx):
        plt.scatter(x[:,0], x[:,1], s=area, c=colors[i, :], alpha=0.8, cmap=plt.cm.Spectral, \
                    facecolor='0.5', lw = 0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 1., box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                  fancybox=True, shadow=True, ncol=3)

    plt.savefig(fname, bbox_inches='tight', format='pdf')


def divide_by_labels(xx, yy, num_class=10):

    xx_list = []
    for i in range(num_class):

        indice = np.argwhere(yy==i).flatten()
        xx_list.append(xx[indice])

    return xx_list


def prob_K_per(prob, Ks):

    N = prob.shape[0]
    sorted_prob_ind = np.argsort(prob, axis=1)[:,::-1]

    probKs = []
    for k in Ks:
        probK = prob[j, sorted_prob_ind[:,k]]
        probKs.append([np.mean(probK, axis=0), np.std(probK, axis=0)])

    probKs = np.asarray(probKs)
    return probKs


def uncompress_sparseMatrix(matrice):

    data = []; labels = []
    n = matrice.shape[0]
    for i in range(n):
        data.append(matrice[i][0].todense())
        labels.append(np.ones(matrice[i][0].shape[0]) * i)
    data = np.vstack(data).astype('float32')
    perm = rng.permutation(data.shape[0])
    labels = np.hstack(labels)

    return data[perm], labels[perm]

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)



def conv_cond_concat(x, y):
    """ 
    concatenate conditioning vector on feature map axis 
    """
    return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)


def init_conv_weights(W_low, W_high, filter_shape, numpy_rng, rng_dist='normal'):
    """
    initializes the convnet weights.
    """

    if 'uniform' in rng_dist:
        return np.asarray(
            numpy_rng.uniform(low=W_low, high=W_high, size=filter_shape),
                dtype=theano.config.floatX)
    elif rng_dist == 'normal':
        return 0.01 * numpy_rng.normal(size=filter_shape).astype(theano.config.floatX)


def initialize_weight(n_vis, n_hid, W_name, numpy_rng, rng_dist):
    """
    """

    if 'uniform' in rng_dist:
        W = numpy_rng.uniform(low=-np.sqrt(6. / (n_vis + n_hid)),\
                high=np.sqrt(6. / (n_vis + n_hid)),
                size=(n_vis, n_hid)).astype(theano.config.floatX)
    elif rng_dist == 'normal':
        W = 0.01 * numpy_rng.normal(size=(n_vis, n_hid)).astype(theano.config.floatX)
    elif rng_dist == 'ortho': ### Note that this only works for square matrices
        N_ = int(n_vis / float(n_hid))
        sz = np.minimum(n_vis, n_hid)
        W = np.zeros((n_vis, n_hid), dtype=theano.config.floatX)
        for i in range(N_):
            temp = 0.01 * numpy_rng.normal(size=(sz, sz)).astype(theano.config.floatX)
            W[:, i*sz:(i+1)*sz] = sp.linalg.orth(temp)


    return theano.shared(value = np.cast[theano.config.floatX](W), name=W_name)



def init_uniform(shape, rng, scale=0.05, name='W'):

    return theano.shared(rng.uniform(low=-scale, high=scale, size=shape).astype('float32'), name=name)


'''Initialize the bias'''
def initialize_bias(n, b_name):

    return theano.shared(value = np.cast[theano.config.floatX](np.zeros((n,)), \
                dtype=theano.config.floatX), name=b_name)


def init_zero(shape, dtype=theano.config.floatX, name=None):
    return theano.shared(np.zeros(shape, dtype='float32'), name=name)


'''Convolve Gaussian filters'''
def convolve2D(F1,F2, Z): 

    Fz  = (F1.dimshuffle([0,1,2,'x']) *  Z.dimshuffle([0,'x',1,2])).sum(axis=-2)
    FzF = (Fz.dimshuffle([0,1,2,'x']) * F2.dimshuffle([0,'x',1,2])).sum(axis=-2)

    return FzF



def unpickle(path):
    ''' For cifar-10 data, it will return dictionary'''
    #Load the cifar 10
    f = open(path, 'rb')
    data = cPickle.load(f)
    f.close()
    return data 


def share_input(x):
    return theano.shared(np.asarray(x, dtype=theano.config.floatX))


def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    #When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue

    return shared_x, T.cast(shared_y, 'int32')


def repmat_tensor(x,k):

    return T.tile(x.dimshuffle([0,1, 2,'x']), [1,1,1,k])


def repmat_vec(x,k):

    return T.tile(x.dimshuffle([0,'x']), [1,k]).T


def activation_fn_th(X,atype='sigmoid', leak_thrd=0.2):
    '''collection of useful activation functions'''

    if atype == 'softmax':
        return T.nnet.softmax(X)
    elif atype == 'sigmoid':
        return T.nnet.sigmoid(X)
    elif atype == 'tanh':
        return T.tanh(X)
    elif atype == 'softplus':
        return T.nnet.softplus(X)
    elif atype == 'relu':
        return (X + abs(X)) / 2.0
    elif atype == 'linear':
        return X
    elif atype =='leaky':
        f1 = 0.5 * (1 + leak_thrd)
        f2 = 0.5 * (1 - leak_thrd)
        return f1 * X + f2 * abs(X)
    elif atype == 'elu':   
        return T.switch(X > 0, X, T.expm1(X))
    elif atype =='selu':
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*T.where(X>=0.0, X, alpha*T.switch(X > 0, X, T.expm1(X)))




def save_the_weight(x,fname):
    '''save pickled weights'''
    f = file(fname+'.save', 'wb')
    cPickle.dump(x, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print("saved!")
    f.close()
   

def save_the_numpy_params(model,size,rank,epoch, model_path):
    
    tmp  = []

    for param in model.gen_network.params:
        if rank==0: print(param.get_value().shape)
        tmp.append(param.get_value())

    np.save(model_path+'/%d%dgen_params_e%d.npy' % (size, rank, epoch),tmp)

    # comm.Barrier()
    print('saved')
    # exit(0)
 

'''Display the data.
data - n_dim X 3 ,  n_dim = dim_x * dim_y '''
def display_data(data, img_sz, RGB_flag=False, ):
    if RGB_flag:
    	pic = data.reshape(img_sz[0],img_sz[1],3)
    else:
        pic = data.reshape(img_sz[0],img_sz[1])

    plt.figure()
    plt.imshow(pic, cmap='gray')


'''Display dataset as a tiles'''
def display_dataset(data, patch_sz, tile_shape, scale_rows_to_unit_interval=False, \
                                            binary=False, i=1, fname='dataset'):

    x = tile_raster_images(data, img_shape=patch_sz, \
    						tile_shape=tile_shape, tile_spacing=(1,1), output_pixel_vals=False, scale_rows_to_unit_interval=scale_rows_to_unit_interval)
    
    if binary:
    	x[x==1] = 255		

    ## For MNIST
    if fname != None:
        plt.figure()
        plt.imshow(x,cmap='gray')
        plt.axis('off')
        plt.savefig(fname+'.png', bbox_inches='tight')
    else:
        plt.figure()
        plt.imshow(x,cmap='gray')
        plt.axis('off')
        plt.show(block=True)
        

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=True):
    """
Transform an array with one flattened image per row, into an array in
which images are reshaped and layed out like tiles on a floor.

This function is useful for visualizing datasets whose rows are images,
and also columns of matrices for transforming those rows
(such as the first layer of a neural net).

:type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
be 2-D ndarrays or None;
:param X: a 2-D array in which every row is a flattened image.

:type img_shape: tuple; (height, width)
:param img_shape: the original shape of each image

:type tile_shape: tuple; (rows, cols)
:param tile_shape: the number of images to tile (rows, cols)

:param output_pixel_vals: if output should be pixel values (i.e. int8
values) or floats

:param scale_rows_to_unit_interval: if the values need to be scaled before
being plotted to [0,1] or not


:returns: array suitable for viewing as an image.
(See:`PIL.Image.fromarray`.)
:rtype: a 2-d array with same dtype as X.

"""

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    # tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    # tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X[0].dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array

                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array

def corrupt_input(rng, input, corruption_level, ntype='gaussian'):

    return input + rng.normal(size = input.shape, loc = 0.0,
                scale= corruption_level)



def get_corrupted_input(rng, input, corruption_level, ntype='zeromask'):
    ''' depending on requirement, returns input corrupted by zeromask/gaussian/salt&pepper'''
    MRG = RNG_MRG.MRG_RandomStreams(rng.randint(2 ** 30))
    #theano_rng = RandomStreams()
    if corruption_level == 0.0:
        return input

    if ntype=='zeromask':
        return MRG.binomial(size=input.shape, n=1, p=1-corruption_level,dtype=theano.config.floatX) * input
    elif ntype=='gaussian':
        return input + MRG.normal(size = input.shape, avg = 0.0,
                std = corruption_level, dtype = theano.config.floatX)
    elif ntype=='salt_pepper':

        # salt and pepper noise
        print('DAE uses salt and pepper noise')
        a = MRG.binomial(size=input.shape, n=1,\
                p=1-corruption_level,dtype=theano.config.floatX)
        b = MRG.binomial(size=input.shape, n=1,\
                p=corruption_level,dtype=theano.config.floatX)

        c = T.eq(a,0) * b
        return input * a + c

''' improving learning rate'''
def get_epsilon_inc(epsilon, n, i):
    """
    n: total num of epoch:
    i: current epoch num
    """
    return epsilon / ( 1 - i/float(n))

'''decaying learning rate'''
def get_epsilon(epsilon, n, i):
    """
    n: total num of epoch
    i: current epoch num
    """
    return epsilon / ( 1 + i/float(n))

def get_epsilon_decay(i, num_epoch, constant=4): 
    c = np.log(num_epoch/2)/ np.log(constant)
    return 10.**(1-(i-1)/(float(c)))


'''Given tiles of raw data, this function will return training, validation, and test sets.
r_train - ratio of train set
r_valid - ratio of valid set
r_test  - ratio of test set'''
def gen_train_valid_test(raw_data, raw_target, r_train, r_valid, r_test):
    N = raw_data.shape[0]
    perms = np.random.permutation(N)
    raw_data   = raw_data[perms,:]
    raw_target = raw_target[perms]

    tot = float(r_train + r_valid + r_test)  #Denominator
    p_train = r_train / tot  #train data ratio
    p_valid = r_valid / tot  #valid data ratio
    p_test  = r_test / tot	 #test data ratio
    
    n_raw = raw_data.shape[0] #total number of data		
    n_train =int( math.floor(n_raw * p_train)) # number of train
    n_valid =int( math.floor(n_raw * p_valid)) # number of valid
    n_test  =int( math.floor(n_raw * p_test) ) # number of test

    
    train = raw_data[0:n_train, :]
    valid = raw_data[n_train:n_train+n_valid, :]
    test  = raw_data[n_train+n_valid: n_train+n_valid+n_test,:]
    
    train_target = raw_target[0:n_train]
    valid_target = raw_target[n_train:n_train+n_valid]
    test_target  = raw_target[n_train+n_valid: n_train+n_valid+n_test]
    
    print('Among ', n_raw, 'raw data, we generated: ')
    print(train.shape[0], ' training data')
    print(valid.shape[0], ' validation data')
    print(test.shape[0],  ' test data\n')
    
    train_set = [train, train_target]
    valid_set = [valid, valid_target]
    test_set  = [test, test_target]
    return [train_set, valid_set, test_set]


def dist2hy(x,y):
    '''Distance matrix computation
    Hybrid of the two, switches based on dimensionality
    '''

    d = T.dot(x,y.T)
    d *= -2.0
    d += T.sum(x*x, axis=1).dimshuffle(0,'x')
    d += T.sum(y*y, axis=1)

    # Rounding errors occasionally cause negative entries in d
    d = d * T.cast(d>0,theano.config.floatX)

    return d


def dist2hy_np(x,y):
    '''Distance matrix computation
    Hybrid of the two, switches based on dimensionality
    '''

    d = np.dot(x,y.T)
    d *= -2.0
    d += np.sum(x*x, axis=1)[:,None]
    d += np.sum(y*y, axis=1)

    # Rounding errors occasionally cause negative entries in d
    d = d * np.asarray(d>0,'float32')

    return np.sqrt(d)


def save_the_env(dir_to_save, path):
    
    import fnmatch
    import os

    matches = []
    for root, dirnames, filenames in os.walk(dir_to_save):
        for extension in ('*.py', '*.sh'):
            for filename in fnmatch.filter(filenames, extension):
                matches.append(os.path.join(root, filename))
        
    import tarfile
    out = tarfile.open('env.tar', mode='w')
    
    try:
        for f in matches:
            out.add(f)
    except Exception as e:
        raise e

    out.close()
   
    import shutil
    tar_to_save = 'env.tar'
    
    shutil.copy2(tar_to_save, path)
