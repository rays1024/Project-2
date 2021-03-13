# DATA 410 Advanced Applied Machine Learning Project 2
In this project, we will compare the following regularization techniques in terms of variable selection and prediction. We will applly the following techniques on both real data and synthetic data.

## General Imports
These imports are the tools for data simulation and regularization technique applications.
```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
```

```python
! pip install --upgrade Cython
! pip install --upgrade git+https://github.com/statsmodels/statsmodels
import statsmodels.api as sm
```

```python
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
```

## Simulated Data

Here we wrtite a function to generate simulated data with arbitrary number of observations, features, and Toeplitz correlation matrix.
Input "num_samples" is the number of observations we need. Input "p" is the number of features we want in our simulated data. Input "rho" is used to generate coefficients in the correlation matrix. The function will return a matrix "X" as the synthetic dataset. We will use "X" and other datasets for regularization technique applications.
```python
def make_correlated_features(num_samples,p,rho):
  vcor = [] 
  for i in range(p):
    vcor.append(rho**i)
  r = toeplitz(vcor)
  mu = np.repeat(0,p)
  X = np.random.multivariate_normal(mu, r, size=num_samples)
  return X
```

In our project, we simulated a dataset with 500 observations and four features, and we set the value of rho to be 0.8.

```python
betas =np.array([-5,2,0,6])
betas=betas.reshape(-1,1)
n = 500
sigma = 2
y_data = X_data.dot(betas) + sigma*np.random.normal(0,1,n).reshape(-1,1)
```
Then we created a ground truth of [-5,2,0,6], and used it to generate y with sigma to be 2.

## Ridge Regularization
The Ridge Regression is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}\beta_i^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}\beta_i^2" title="\text{minimize}\, \frac{1}{n}\text{SSR} + K\sum\limits_{i=1}^{n}\beta_i^2" /></a>

where SSR is the squared residual and K is a tuning parameter.

We used the 

## Least Absolute Shrinkage and Selection Operator (LASSO)


## Elastic Net


## Smoothly Clipped Absolute Deviation (SCAD)


## Square Root LASSO
