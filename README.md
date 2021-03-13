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

## Data

### Simulated Data

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
n = 500
p = 4
X_data = make_correlated_features(n,p,0.8)
```

Then we created a ground truth of [-5,2,0,6], and used it to generate y with sigma to be 2.
```python
betas =np.array([-5,2,0,6])
betas=betas.reshape(-1,1)
n = 500
sigma = 2
y_data = X_data.dot(betas) + sigma*np.random.normal(0,1,n).reshape(-1,1)
```

### Real Data
We used the Boston Housing Price dataset to compare different regularization techniques. The x variables include crime, rooms, residential, industrial, nox, older, distance, highway, tax, ptratio, lstat. We used cmedv as the y variable. Then we created training and testing groups for the KFold Validation process.

```python
df = pd.read_csv('/content/Boston Housing Prices.csv')
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df[features])
y = np.array(df['cmedv']).reshape(-1,1)
X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state=1693)
kf = KFold(n_splits=5,shuffle=True,random_state=1234)
```

## Ridge Regularization
The Ridge Regression is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}\beta_i^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}\beta_i^2" title="\text{minimize}\, \frac{1}{n}\text{SSR} + K\sum\limits_{i=1}^{n}\beta_i^2" /></a>

where SSR is the squared residual and K is a tuning parameter.

We used the "statsmodels.api" package and its upgrade from GitHub to calculate the KFold validated MAE. Since the Elastic Net technique is the combination of Ridge and LASSO, we can set the L1 weight to be 0 to obtain Ridge regression results. 

The following function is what we used for Ridge, LASSO, Elastic Net, and Square Root LASSO KFold Validation MAE.
```python
def DoKFold(X,y,alpha,m,w):
  PE = []
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    model = sm.OLS(y_train,X_train)
    if w != -1:
      result = model.fit_regularized(method=m, alpha=alpha)
    else:
      result = model.fit_regularized(method=m, alpha=alpha, L1_wt=w)
    yhat_test = result.predict(X_test)
    PE.append(MAE(y_test,yhat_test))
  return 1000*np.mean(PE)
  ```
  As mentioned above, we set the method to be 'elastic_net' and set the L1_wt to be 0. We also used an iteration process to find the best hyperparameter in a certain range for both real and synthetic data.
  ```python
mae=[]
alpha_val=[]
for i in np.arange(0,1.01,0.01):
  mae.append(DoKFold(X_train,y_train,i,'elastic_net',0))
  alpha_val.append(i)
print(min(mae))
print(alpha_val[mae.index(mae==min(mae))])
```
MAE = $3580.2053332106398

Best alpha value from 0 to 1 = 0.11

```python
alpha_val=[]
L2_norm=[]
for i in np.arange(0,10.01,0.1):
  betahat = sm.OLS(y_data,X_data).fit_regularized(method='elastic_net', alpha=i, L1_wt=0).params
  L2_norm.append(np.sqrt(np.sum((betahat-betas)**2)))
  alpha_val.append(i)
print(min(L2_norm))
print(alpha_val[L2_norm.index(min(L2_norm))])
```
L2 norm = 15.996158036188461

Best alpha value from 1 to 10 = 6.0

## Least Absolute Shrinkage and Selection Operator (LASSO)
The LASSO Regression is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}|\beta_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}|\beta_i|" title="\text{minimize} \frac{1}{n}\text{SSR} + K\sum\limits_{i=1}^{n}|\beta_i|" /></a>

where where SSR is the squared residual and K is a tuning parameter.

We used the "statsmodels.api" package and its upgrade from GitHub to calculate the KFold validated MAE. Since the Elastic Net technique is the combination of Ridge and LASSO, we can set the L1 weight to be 1 to obtain LASSO regression results. 
```python
mae=[]
alpha_val=[]
for i in np.arange(0,1.01,0.01):
  mae.append(DoKFold(X_train,y_train,i,'elastic_net',1))
  alpha_val.append(i)
print(min(mae))
print(alpha_val[mae.index(min(mae))])
```
MAE = $3580.2053332106398
Best alpha value from 0 to 1 = 0.11

## Elastic Net
The Elastic Net is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\left(\alpha\sum\limits_{i=1}^{n}|\beta_i|&plus;(1-\alpha)\sum\limits_{i=1}^{n}\beta_i^2\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\left(\alpha\sum\limits_{i=1}^{n}|\beta_i|&plus;(1-\alpha)\sum\limits_{i=1}^{n}\beta_i^2\right)" title="\text{minimize}\, \frac{1}{n}\text{SSR} + K\left(\alpha\sum\limits_{i=1}^{n}|\beta_i|+(1-\alpha)\sum\limits_{i=1}^{n}\beta_i^2\right)" /></a>


## Smoothly Clipped Absolute Deviation (SCAD)


## Square Root LASSO
