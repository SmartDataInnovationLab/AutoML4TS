import sys
sys.path.append("..")

import numpy as np
from sklearn import linear_model
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UnParametrizedHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter

from Forecasting import Automated_ML_Forecasting

class ARD_regression_Forecasting(Automated_ML_Forecasting):
    
    def __init__(self, timeseries, dataname, 
                 #parameter for arg regression
                 n_iter = 300, tol=0.001, 
                 alpha_1=0.000001, alpha_2=0.000001, lambda_1=0.000001, lambda_2=0.000001,
                 threshold_lambda=10000, fit_intercept=True, 
                 #feature extraction parameter
                 Window_size = 20 , Difference = False,
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05): 
        
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.threshold_lambda = float(threshold_lambda)
        self.fit_intercept = fit_intercept
            
        self.estimator = linear_model.ARDRegression(
            n_iter=self.n_iter,
            tol=self.tol,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            threshold_lambda=self.threshold_lambda,
            fit_intercept=self.fit_intercept,
            verbose = True)
        
        super().__init__(timeseries, dataname, Window_size, time_feature, Difference, tsfresh_feature,
                         forecasting_steps, n_splits, max_train_size, NAN_threshold)
        
    def _direct_prediction(self):
        super()._direct_prediction(self.estimator)
        
    def _cross_validation(self):
        return super()._Time_Series_forecasting_cross_validation(self.estimator)
    
    def _cross_validation_visualization(self):
        super()._cross_validation_visualization(self.estimator)
    
    def _cross_and_val(self):
        return super()._cross_and_val(self.estimator)
    
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        Window_size = UniformIntegerHyperparameter(
            name="Window_size", lower=5, upper=50, default_value=20)
			
        Difference = CategoricalHyperparameter(
            name="Difference", choices=["True", "False"], default_value="True")

        tsfresh_feature = CategoricalHyperparameter(
            name="tsfresh_feature", choices=["True", "False"], default_value="True")

        
        n_iter = UnParametrizedHyperparameter("n_iter", value=50)
        
        tol = UniformFloatHyperparameter("tol", 10 ** -5, 10 ** -1,
                                         default_value=10 ** -3, log=True)
        
        alpha_1 = UniformFloatHyperparameter(name="alpha_1", lower=10 ** -10,
                                             upper=10 ** -3, default_value=10 ** -6)
        
        alpha_2 = UniformFloatHyperparameter(name="alpha_2", log=True,
                                             lower=10 ** -10, upper=10 ** -3,
                                             default_value=10 ** -6)
        
        lambda_1 = UniformFloatHyperparameter(name="lambda_1", log=True,
                                              lower=10 ** -10, upper=10 ** -3,
                                              default_value=10 ** -6)
        
        lambda_2 = UniformFloatHyperparameter(name="lambda_2", log=True,
                                              lower=10 ** -10, upper=10 ** -3,
                                              default_value=10 ** -6)
        threshold_lambda = UniformFloatHyperparameter(name="threshold_lambda",
                                                     log=True,
                                                     lower=10 ** 3,
                                                     upper=10 ** 5,
                                                     default_value=10 ** 4)
        
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")

        cs.add_hyperparameters([n_iter, tol, alpha_1, alpha_2, lambda_1, Difference,
                                lambda_2, threshold_lambda, fit_intercept, Window_size, tsfresh_feature])

        return cs
