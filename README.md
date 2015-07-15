GamelanPy
=========

GamelanPy is a python implementation of GAMELAN (Genereative Model for Energy LANscape) algorithm.
This algorithm is for learning a sparse Gaussian Mixture Model for structural fluctuations
of a protein, but can also be used for other data where the original data can be assumed 
to follow a mixture of Gaussian distributions.
GamelanPy supports options for sub-sampling methods for scalability and 
nonparanormal distributions for richer family of distributions than Gaussians.
This package contains a pure python library and scripts for the command-line usages.
For more detail on the algorithms, please refer to 
Efficient Learning of Sparse Gaussian Mixture Model of Protein Conformational Substates.



Dependencies
============

GamelanPy is tested to work under Python 2.7 (not tested in Python 3.x)
The required dependencies are:

* numpy >= 1.6.0
* scipy >= 0.13.0
* scikit-learn >= 0.16.0

and the core class and functions in the library are built on top of the scikit-learn's 
GMM class.

GamelanPy is a pure python module, and does not need to be compiled with other compilers.


Install
=======

It is recommended to install GamelanPy with pip:
    
    pip install gamelanpy
   
As this package uses distutils, you also install GamelanPy by downloading the source 
and running:

    cd /path/to/source
    python setup.py install


Code and Repository
===================

The source code is maintained in the github repository:

    https://github.com/yoojioh/gamelanpy

and you can clone the most recent status of the package by running:

    git clone https://github.com/yoojioh/gamelanpy.git



Usages
======

Python Library
--------------

### SparseGMM
SparseGMM class is a main class for a sparse Gaussian Mixture Model. 
The snippet below is a simple code to create SparseGMM object and learn a model:

    from gamelanpy import SparseGMM
    s_gmm = SparseGMM(n_components=3)
    s_gmm.fit(data, 'coreset', 1000)

User must specify the number of components and the type of distributions when
creating the SparseGMM object:
* n_components: Required. int.
* nonparanormal: Optional, default is False. When nonparanoraml is True, 
this object represents a mixture of sparse nonparanormal distributions, and
the learning process takes extra step for estimation of CDF (explained below).

SparseGMM.fit() performs the structure and parameter learning of the model, 
and there are number of options:

* **subsample_method**: Required. 'coreset', 'coreset2', 'uniform', or 'None'
    * 'coreset': Use coreset sampling for cluster identification, 
    and use the original dataset to learn a sparse Gaussian for each cluster.
    * 'coreset2': Use coreset sampling for cluster identification,
    and use **only coreset samples** to learn a sparse Gaussian for each cluster.
    Recommended only if the runtime is the main issue.
    * 'uniform': Use randomly chosen samples for cluster identification, 
    and use the original dataset to learn a sparse Gaussian for each cluster.
    * 'None': Does not use any subsampling method.
* **subsample_size**: Required. Size of the subsamples.
* **l1_penalty_range**: Optional, [min_value, max_value] for l1_penalty parameter search
for each cluster. The default is [0.0001, 10.0].
* **l1_search_depth**: Optional. l1_penalty search is done by binary-search-like
heuristic method using graphical lasso, and it repeats the narrowing the range for
l1_penalty by the given number of steps. The default is 20.
* **l1_search_repeat**: Optional. The heuristic method is not deterministic, so it is
recommended to repeat the search several times and takes the average for the final
l1_penalty parameter. The default is 10.
* **npn_sample_ratio**: Only required when the SparseGMM object is for 
nonparanormal distributions. Estimation of CDF takes extra space complexity, so
users can specify the ratio of the givens samples to use in estimating the CDF.
The value must be in (0.0, 1.0].

Below are some of the examples using a combination of options:

* coreset sampling method with coreset size=500

        s_gmm = gamelan.SparseGMM(n_components=3)
        s_gmm.fit(data, 'coreset', 1000)

* uniform (random) sampling with subsample size=1000

        s_gmm = gamelan.SparseGMM(n_components=3)
        s_gmm.fit(data, 'uniform', 500)
        
* coreset2 sampling method with coreset size=1000 

        s_gmm = gamelan.SparseGMM(n_components=3)
        s_gmm.fit(data, 'coreset2', 1000)
        
* coreset sampling method to estimate a mixture of sparse nonparanormals

        s_gmm = gamelan.SparseGMM(n_components=3, nonparanormal=True)
        s_gmm.fit(data, 'coreset', 1000)
        
* search for the best number for the number of components using BIC score

        n_components_candidates = range(3, 6)
        bic_scores = []
        best_score = -np.inf
        best_model = None
        
        for n_comp in n_components_candidates:
            s_gmm = gamelan.SparseGMM(n_components=n_comp)
        
            # learn using coreset subsampling
            s_gmm.fit(data, 'coreset', 1000)
            bic_score = s_gmm.bic(test)
            bic_scores.append(bic_score)
        
            if bic_score > best_score:
                best_score = bic_score
                best_model = s_gmm



### Prediction and Imputation
Once a model is learned, it can be used for prediction studies. 
The functions **predict_missing_values()** and **sample_missing_values()** can be used
to predict the missing values in a given data and to sample the missing coordinates.

The incomplete data must be ndarray type where **the missing values are NaN's.**

Below is a simple code snippet using both functions:

    from gamelanpy.imputation_util import predict_missing_values
    
    # get the most probable values (mean in the conditional distribution)
    complete_data = predict_missing_values(model, incomplete_data)
    
    # sample the missing values using the conditional distribution
    samples_data = sample_missing_values(model, incomplete_data, n_samples=10)

More example usages are in the example directory in the source.
    

Python Scripts
--------------
This package also contains python scripts that can be used from command-line.
The scripts are installed when the package is installed.

### gamelan_learn_model.py

Usage: 
    
    gamelan_learn_model.py <num_components> <path_to_data> [options]

* num_components: number of components for the model
* path_to_data: path to the csv formatted data

Options for learning: These options are corresponding options in SparseGMM.fit() 
function, and refer to the above section for more explanations.

* --l1-depth: l1_search_depth for l1_penalty parameter search.
* --l1-min and --l1-max: min and max values for l1_parameter search.
* --subsample-method: 'coreset', 'coreset2' or 'uniform'. When not specified, 
it does not use any subsampling method, and uses the whole samples for the learning
* --subsample-size: the size of the subsamples. Required when the --subsample-method
is specified.
* --l1-search-repeat: Number of repeats for l1_penalty parameter search.
* --l1-search-depth: Depth for the binary-search-like heuristic method for 
l1_penalty parameter search.
* --npn-sample-ratio: Sample ratio between 0.0 to 1.0 for estimation step for
learning nonparanormal distributions.

Options for storing the model: It is recommended to dump the whole model after 
the learning, but users can also specify what parameters they want to store.

* --save-model: Path to the file for the whole model. The model is stored in JSON format
where ndarray is stored in base64 encoded values. When the model is stored with this option,
it can be loaded later for gamelan_imputation.py script.
* --save-covars: Path to the file for covariance matrices. 
The covariance matrices 3-dimensional ndarray, so it is reshaped to (n_components * n_vars, n_vars).
* --save-covars: Path to the file for precision matrices. 
The precision matrices 3-dimensional ndarray, so it is reshaped to (n_components * n_vars, n_vars).
* --save-means: Path to the file for mean vectors.
* --save-weights: Path to the file for weights of components


Contribution
============

Any contribution is welcome!
 