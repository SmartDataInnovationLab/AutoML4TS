# AutoML4TS : Automated-Machine-Learning-AutoML-for-time-series-forecasting
AutoML4TS is a automated tool for forecasting time-series data based on classic regression models. It works well with different time-series, e.g seasonal, non-seasonal, trend, non-trend. When using this tool, the user does not need to make choice between different models and customize it by setting its hyperparameters. This is a project of [SDSC-BW](https://sdsc-bw.de/erfolge).

![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/Prediction.png)

Time-series forecasting is studied in nearly all fields of science and engineering. For all forecasting tasks, model selection is a necessary step. However, from a set of available models, selecting the most adequate model for a given dataset is still a diﬃcult task. If the hyper-parameters of candidate models are taken into consideration, there would be an enormous number of possible alternatives overall. The wide range of forecasting applications led to growing demand for a system which can automatically select a good model and simultaneously set good parameters for a new task.

The combined algorithm selection and hyper-parameter optimization problem was dubbed CASH. Recently, automated approaches for solving this problem have led to substantial improvements. The Sequential Model-based Algorithm Conﬁguration ([SMAC](https://github.com/automl/SMAC3)) method showed a successful performance in a high-dimensional, structured and continuous and discrete (Categorical) hybrid parameters space.

This framework, through the application of SMAC, deals with this problem in the context of forecasting. In contrast to previous works on time-series forecasting, the candidate models in my work are machine learning (ML) models from the Scikit-learn regression package. 

In order to improve the performance of ML models, the package TSFRESH  [reference] is integrated in the framework, which extracts comprehensive and well-established features from time-series. In order to improve the efficiency of the optimization process, a meta-learning method are supported to generate configurations with the likelihood of performing for a new task. This initialization method yields slight improvements to expedite the
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

The next few paragraphs will give you an overview of this framework, which is inspired by [Auto Sklearn](https://github.com/automl/auto-sklearn)
.
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
AutoML4TS is specifically designed for pruduction or commercial business. It is important to know what features have a major impact on the future. Linear models are popular because of their explainability. Getting features' importance from a tree-based model is pretty easy. After training on a time-series, this tool will give important features. 

# Retraining the meta-learner

![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/framework.png)
In order to accelerate the optimization process， a meta-learning system are built in this framework which as shown below. when SMAC starts with a new optimization task, it requires a number of initial evaluations to build the initial surface model. An effective initialization helps SMAC to build a better initial surface model, which can also accelerate the detection of the high-performance region [reference]. Meta-learning is a promising solution, which can suggest good configurations for a new dataset that performed well on previous similar datasets. 

After training a new time-series, the corresponding best model and hyperparameters will be stored in your local database for future use. Now we are trying to build a repository of comprehensive time-seires and corresponding best parameter configurations. After finishing that, we will release these data and provide APIs to access it.

# Acknowledgement

TBD

