import sys
sys.path.append("..")

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
import sklearn.ensemble
import sklearn.tree

from  Forecasting import Automated_ML_Forecasting


class Adaboost_Forecasting(Automated_ML_Forecasting):
    
    def __init__(self, timeseries, dataname, 
                 # parameter for adaboost regression
                 n_estimators=50, learning_rate=0.1, loss="linear", max_depth=3,
                 #feature extraction parameter
                 Window_size = 20 , Difference = False, 
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):
        
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.max_depth = int(max_depth)
        
        base_estimator = sklearn.tree.DecisionTreeRegressor(
            max_depth=self.max_depth)
        
        self.estimator = sklearn.ensemble.AdaBoostRegressor(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            loss=self.loss
        )
        
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

        n_estimators = UniformIntegerHyperparameter(
            name="n_estimators", lower=50, upper=500, default_value=50, log=False)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
        loss = CategoricalHyperparameter(
            name="loss", choices=["linear", "square", "exponential"],
            default_value="linear")
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=10, default_value=3, log=False)

        cs.add_hyperparameters([n_estimators, Difference, learning_rate, loss, max_depth, Window_size, tsfresh_feature])
        return cs

