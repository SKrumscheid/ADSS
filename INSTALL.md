# Installation

## For Usage

To use the code, just run the following command in base directory to install the package:

```
pip install .
```
Alternatively, you can also simply use the corresponding [PyPi package](https://pypi.org/project/adaptive-stratification/) instead.

## For development

For further code developments, it may be convenient to create a designated virtual environment. For example, we provide two possible ways of creating such a virtual environment, which will account for the necessary requirements, are described below. Both options will also install the Adaptive Stratification package.

### Using pip

The  Adaptive Stratification package's dependencies are listed in the [requirements.txt](./requirements.txt) file, which can be used as follows:
```
(python -m) pip install -r requirements.txt
```
For this option, you may want to consider tools for creating isolated virtual python environments, such as [virtualenv](https://pypi.org/project/virtualenv/).

### Using Conda

The dependencies and settings contained in the [environment.yaml](./environment.yaml) can be used to create a Conda environment via:

```
conda env create -f environment.yaml
```

This command and it will create a virtual environment called `stratification-env`.

#### Activating the Conda environment

Once created, you can activate the Conda environment `stratification-env` via:

```
conda activate stratification-env
```

#### Updating the Conda environment

To ensure that you are working with the most update Conda environment, we recommend using the following command whenever the [environment.yaml](./environment.yaml) files has changed.

```
conda env update -f environment.yaml  --prune
```

## Manual installation
In case you do not want to use the environment provided here through the files 
[requirements.txt](./requirements.txt) and [environment.yaml](./environment.yaml), respectively, you can use the following commands to create your own one. 

**pip:**

```
pip install python==3.9 numpy scipy
```

**conda:**

```
conda create -n <env-name> python=3.9 numpy scipy
```

Comments on minimal dependencies:

* the `matplotlib` package is not required for the `AdaptiveStratification` method. In fact, it is only needed when using the accompanying visualization feature (d=1 and d=2 only);
* similarly, the packages `sphinx` and `sphinx-rtd-theme` are also optional, as they are only required for generating the documentation.
