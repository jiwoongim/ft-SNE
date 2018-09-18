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
    journal={http://arxiv.org/abs/1602.05110},
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
Entry code for MNIST 
```
    - ./main.py --datatype mnist  --divtypet kl --perplexity 100
```


