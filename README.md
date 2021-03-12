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



## Ridge Regularization


## Least Absolute Shrinkage and Selection Operator (LASSO)


## Elastic Net


## Smoothly Clipped Absolute Deviation (SCAD)


## Square Root LASSO
