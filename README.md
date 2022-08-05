# Adaptive Stratification (ADSS)

This package provides an implementation of the adaptive stratification sampling method to estimate quantities of interest of the form

[<img src="./assets/QoI.png" width="150"/>](./assets/QoI.png)

where the random vector satisfies

[<img src="./assets/QoI_details.png" width="550"/>](./assets/QoI_details.png)

is a given function.

## Using the code
To use the Python code provided here, see the [INSTALL.md](./INSTALL.md) file for further instructions.

Alternatively, if you would like to use the adaptive stratification sampling method without cloning the repository, then you can simply use the `adaptive-stratification` package as part of the Python Package Index (PyPi), which is available [here](https://pypi.org/project/adaptive-stratification/). The PyPi package contains the implementation of the basic adaptive stratification sampling method, but not the visualisation features (cf., [example.ipnyb](./example.ipynb)).

## Using the Sampler

The code snippet below indicates the basic usage of the adaptive stratification sampling method provided by the `AdaptiveStratification` function.

```python
# Import the module containing the sampling routines
from stratification import AdaptiveStratification

# Create a sampler for function func
sampler = AdaptiveStratification(func, d, N_max, N_new_per_stratum, alpha, type='hyperrect')

# Solve
result = sampler.solve()
```
The input arguments for the constructor `AdaptiveStratification` are:

* `func`: implementation of the given function that defines the quantity of interest. It needs to be callable, accepting one m-times-n-dimensional numpy array as input, and returns an m-dimensional numpy array;
* `d`: dimension of the stochastic domain;
* `N_max`: number of total samples to be used;
* `N_new_per_stratum`: targeted average number of samples per stratum, controlling the adaptation;
* `alpha`: number between zero and one, defining the hybrid allocation rule;
* `type`: type of tessellation procedure, i.e., hyper-rectangles (`type='hyperrect'`) or simplices (`type='simplex'`)

The `solve` routine executes the adaptive stratified sampling algorithm. It returns:

* the stratified sampling estimator for the quantity of interest Q;
* a list containing all strata created in the adaptive process;
* the number of strata created;
* an estimate of the variance of the stratified sampling estimator.


### Example
A more complete worked out example is available in the Jupyter Notebook [example.ipnyb](./example.ipynb).

## Documentation
The package comes with a Sphinx documentation that can be generated as described [here](./docs/README.md). Once created, the documentation will be located in the `./doc` folder.

## Citing
If you benefit from this package, please cite:
```
@articlePetterssonKrumscheid22,
author = {P. Pettersson and S. Krumscheid},
title = {Adaptive stratified sampling for non-smooth problems},
journal = {International Journal for Uncertainty Quantification},
year    = {2022},
volume  = {12},
number  = {6},
pages   = {71--99}
}
```
