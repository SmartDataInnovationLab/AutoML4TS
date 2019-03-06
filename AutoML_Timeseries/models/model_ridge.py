import sys
sys.path.append("..")
import numpy as np
from  Forecasting import Automated_ML_Forecasting
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UnParametrizedHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
import sklearn.linear_model


class ridge_forecasting(Automated_ML_Forecasting):
    
    def __init__(self, timeseries, dataname, 
                 # parameter for ridge regression
                 alpha=1, fit_intercept=True, tol=1e-3, 
                 #feature extraction parameter
                 Window_size = 20 , Difference = False,
                 time_feature = True,  tsfresh_feature=False,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):
        
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept 
        self.tol = float(tol)
        self.estimator = sklearn.linear_model.Ridge(alpha=self.alpha,
                                                    fit_intercept=self.fit_intercept,
                                                    tol=self.tol,
                                                    copy_X=True)
        
        super().__init__(timeseries,dataname,Window_size, time_feature, Difference, tsfresh_feature,
                         forecasting_steps, n_splits, max_train_size, NAN_threshold)
        
    def _direct_prediction(self, times=3):
        super()._direct_prediction(self.estimator, times) 

        
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

        tsfresh_feature = CategoricalHyperparameter(
            name="tsfresh_feature", choices=["True", "False"], default_value="True")

        Difference = CategoricalHyperparameter(
            name="Difference", choices=["True", "False"], default_value="True")


        alpha = UniformFloatHyperparameter("alpha", 10 ** -5, 10., log=True, q=0.00001, default_value=1.)

        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")

        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, q=0.00001,log=True)
        cs.add_hyperparameters([alpha, fit_intercept, tol, Window_size, Difference, tsfresh_feature])
        return cs
