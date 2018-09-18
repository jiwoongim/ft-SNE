# Stochastic Neighbour Embedding under f-divergence 

Python (Theano) implementation of Stochastic Neighbour Embedding under f-divergence  code provided 
by Daniel Jiwoong Im, Nakul Verma, Kristin Branson 

ft-Stochastic Neighbour Embedding (ft-SNE) is f-divergence based loss criteria for t-SNE.
The main idea is that different f-divergence produce better low-dimensional visualizations 
for different types of structure in data.

For more information, see 
```bibtex
@article{Im2018,
    title={Stochastic Neighbour Embedding under f-divergence},
    author={Im, Daniel Jiwoong and Verma, Nakul and Branson, Kristin},
    year={2018}
}
```
If you use this in your research, we kindly ask that you cite the above arxiv paper.


## Dependencies
Packages
* [numpy](http://www.numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [sklearn](http://scikit-learn.org/stable/install.html/)
* [Theano ('0.9.0.dev-c697eeab84e5b8a74908da654b66ec9eca4f1291')](http://deeplearning.net/software/theano/) 


## How to run
Entry code for MNIST, MNIST1, FACE, NEWS
```
    python ./main.py --datatype mnist  --divtypet kl --perplexity 100
    python ./main.py --datatype mnist1 --divtypet kl --perplexity 100
    python ./main.py --datatype face   --divtypet kl --perplexity 100
    python ./main.py --datatype news   --divtypet kl --perplexity 100
```
Entry code for runninng your own data:
```
    python ./run.py --divtypet kl --perplexity 100 --dataset_path [YOUR OWN DATADIR]
```
Note that the name of data and labels file must be in NumPy array 
(npy) file. Data file name (data.npy) and Label file name (label.npy),
see line 44-45 in run.py for details.


#Illustration 
ft-SNE embeddings obtained with interpolated divergences 
between KL and RKL. The perpleixty for each row corresponds to 10, 100, and 500 respectively:

![Image of cluster embedding](https://github.com/jiwoongim/ft-SNE/blob/master/blob_cropped.jpg)

![Image of manifold embedding](https://github.com/jiwoongim/ft-SNE/blob/master/swiss_cropped.jpg)



