# Automated-Machine-Learning-AutoML-for-time-series-forecasting

![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/Prediction.png)

Time-series forecasting is studied in nearly all fields of science and engineering. For all forecasting tasks, model selection is a necessary step. However, from a set of available models, selecting the most adequate model for a given dataset is still a diﬃcult task. If the hyper-parameters of candidate models are taken into consideration, there would be an enormous number of possible alternatives overall. The wide range of forecasting applications led to growing demand for a system which can automatically select a good model and simultaneously set good parameters for a new task.

The combined algorithm selection and hyper-parameter optimization problem was dubbed CASH. Recently, automated approaches for solving this problem have led to substantial improvements. The Sequential Model-based Algorithm Conﬁguration (SMAC) method showed a successful performance in a high-dimensional, structured and continuous and discrete (Categorical) hybrid parameters space.

This framework, through the application of SMAC, deals with this problem in the context of forecasting. In contrast to previous works on time-series forecasting, the candidate models in my work are machine learning (ML) models from the Scikit-learn regression package. 

In order to improve the performance of ML models, the package TSFRESH is integrated in the framework, which extracts comprehensive and well-established features from time-series. In order to improve the efficiency of the optimization process, a meta-learning method are supported to generate configurations with the likelihood of performing for a new task. This initialization method yields slight improvements to expedite the
optimization.

# Installation
TBD

System requirements
* Linux operating system
* Python (>=3.5)
* SWIG (version 3.0 or later)

install all dependencies manually with:
```
curl https://rxxxxxxxxxxxxxxxxxx requrements.txt
```
then 
```
pip install autots
```
For installing on Windows, please refer to install_document.md

# Running on time-series
In order to train and predict on a time-series, only the following lines of code are required. It frees users from model selection and hyperparameter tuning leveraging advantages in Bayesian optimization. After optimization, users can get the most appropriate model and related parameters to the data.

```python
# import this AutoML4TS tool
import pandas as pd
from autots import AutoForecasting
# load your data
Timeseries = pd.read_csv(path_to_your_data, squeeze=True)
# do the training
autots = AutoForecasting(Timeseries, Forecasting_steps = 20)
autots.optimize()
# do the prediction
autots.predict()
```

The next few paragraphs will give you an overview of this framework.
![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/timeseries_pipeline.png)
In this framework, time-series forecasting problems are reframed as supervised learning problems through sliding window method. To solve such machine learning problems follows this framework a commonly used process, which consists of the following five steps.

* (1) `Data preparation`
* (2) `Feature extraction and selection`
* (3) `Feature processing`
* (4) `Algorithm selection and hyper-parameters configuration`
* (5) `Validation`

For each step, there are a number of methods to choose from. The configuration space for each space are listed as below.

`Data preparation (missing value impute)` : Mean imputation,   Last observation carried forward (LOCF) method, Next observation carried backward (NOCB) method,   Linear interpolation, Spline interpolation,   Cubic interpolation and Moving average.

`Feature extraction and selection` : the package TSFRESH is integrated in this framework, for more details please see [TSFRESH](https://tsfresh.readthedocs.io/en/latest/)

`Feature processing` : Sigmoid transformation, Principal Component Analysis (PCA), Independent Component Analysis(ICA)

`Algorithm selection` : Ridge Regression,  Bayesian Regression,  Stochastic Gradient Descent Regression,  Support Vector Machines Regression, Decision Trees Regression, Random Forest Regression, Bayesian ARD Regression, Xgboost Regression, LightGBM Regression, Catboost Regression

`Validation` : time-series k-fold walk-forward cross-validation as illustrated below. traditional cross-validation method cannot be directly used with time-series data, because it assumes that there is no relationship between observations and that each observation is independent. The time dependency structure of time-series must not be randomly split into groups. Inwalk-forward cross-validation method, the corresponding training set only consists of observations that occurred prior to the observation that forms the test set.
![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/crossvalidation.png)

# Important Features


# Retraining the meta-learner
In order to accelerate the optimization process a meta-learning system are built in this framework which as shown below.
![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/framework.png)
TBD

# Acknowledgement

TBD

