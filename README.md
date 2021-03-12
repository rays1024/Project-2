# DATA 410 Advanced Applied Machine Learning Project 2
In this project, we will compare the following regularization techniques in terms of variable selection and prediction. We will applly the following techniques on both real data and synthetic data.

## General Imports
These imports are the tools for data simulation and regularization technique applications.
```python
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
```

## Simulated Data

Here we wrtite a function to generate simulated data with arbitrary number of observations, features, and Toeplitz correlation matrix.
Input "num_samples" is the number of observations we need. Input "p" is the number of features we want in our simulated data. Input "rho" is used to generate coefficients in the correlation matrix.
```
def make_correlated_features(num_samples,p,rho):
  vcor = [] 
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  X = np.random.multivariate_normal(mu, r, size=num_samples)
  return X
```

## Ridge Regularization


## Least Absolute Shrinkage and Selection Operator (LASSO)


## Elastic Net


## Smoothly Clipped Absolute Deviation (SCAD)


## Square Root LASSO
