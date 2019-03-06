# Automated-Machine-Learning-AutoML-for-time-series-forecasting

![image](https://github.com/SmartDataInnovationLab/AutoML4TS/blob/master/Images/Prediction.png)

Time-series forecasting is studied in nearly all fields of science and engineering.
For all forecasting tasks, model selection is a necessary step. However, from a
set of available models, selecting the most adequate model for a given dataset is
still a diﬃcult task. If the hyper-parameters of candidate models are taken into
consideration, there would be an enormous number of possible alternatives overall.
The wide range of forecasting applications led to growing demand for a system which
can automatically select a good model and simultaneously set good parameters for
a new task.

The combined algorithm selection and hyper-parameter optimization problem was
dubbed CASH. Recently, automated approaches for solving this problem have led
to substantial improvements. The Sequential Model-based Algorithm Conﬁguration
(SMAC) method showed a successful performance in a high-dimensional, structured
and continuous and discrete (Categorical) hybrid parameters space.

This framework, through the application
of SMAC, deals with this problem in the context of forecasting. In contrast to
previous works on time-series forecasting, the candidate models in my work are ma-
chine learning (ML) models from the Scikit-learn regression package. 

In order to improve the performance of ML models, the package TSFRESH is integrated in the 
framework, which extracts comprehensive and well-established features from time
series. In order to improve the efficiency of the optimization process, three
meta-learning methods are supported to generate configurations with the likelihood of perform-
ing for a new task. I conducted an extensive experiment to validate the eﬀect of
meta-learning. This initialization method yields slight improvements to expedite the
optimization.

# Installation

TBD

# Running on timeseries

TBD

# Retraining the meta-learner

TBD

# Acknowledgement

TBD

