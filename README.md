# Deep Clustering 



This repository stores the models used for experimental computations in the work ["Deep Robust Spectral Clustering"](https://bnaic2023.tudelft.nl/static/media/BNAICBENELEARN_2023_paper_63.01ad1b5e38f534abdaf3.pdf) (link to abstract, request the full paper through [mail](mailto:a.g.rykov@glndwr.ru?subject=Deep%20Robust%20Spectral%20Clustering%3A%20Full%20Paper)).

## Contents

* Models
    * Robust Spectral Map
    * SpectralNet (fork coming soon)
    * Deep K-Means (will be implemented)

* Graph Kernels
    * Nearest Neighbors Kernel
    * Stoachastic Nearest Neighbors Kernel

* Experimental Computations


## Requirements

    numpy==1.26.4
    scikit_learn==1.4.1.post1
    scipy==1.12.0
    torch==2.2.0
    tqdm==4.66.2
    torchvision==0.17.0

## To-do

* Add notebook with experiments
* Add Wiki/Documentation
* Add SpectralNet model (as fork)
* Add Deep K-Means
* Add option for pre-computed kernel (scipy sparse) + fit support
* Add option for dynamic kernel reshaping 