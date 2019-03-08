import sys
sys.path.append("..")

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

import numpy as np 
import sklearn.neighbors
from Forecasting import Automated_ML_Forecasting

class KNN_forecasting(Automated_ML_Forecasting):
    
    def __init__(self, timeseries, dataname,
                 # parameter for KNN forecasting 
                 n_neighbors=5, weights='uniform', p=2, 
                 #feature extraction parameter
                 Window_size = 20 , Difference = False,
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):
        
        self.n_neighbors = int(n_neighbors)
        if weights not in ("uniform", "distance"):
            raise ValueError("'weights' is not in ('uniform', 'distance'): %s" % weights)
        self.weights = weights
        self.p = int(p)
        
        self.estimator = sklearn.neighbors.KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p)
        
        super().__init__(timeseries,dataname,Window_size, time_feature, Difference,  tsfresh_feature,
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

        tsfresh_feature = CategoricalHyperparameter(
            name="tsfresh_feature", choices=["True", "False"], default_value="True")
			
        Difference = CategoricalHyperparameter(
            name="Difference", choices=["True", "False"], default_value="True")

        n_neighbors = UniformIntegerHyperparameter(
            name="n_neighbors", lower=1, upper=100, log=False, default_value=5)
			
        weights = CategoricalHyperparameter(
            name="weights", choices=["uniform", "distance"], default_value="uniform")
			
        p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)

        cs.add_hyperparameters([n_neighbors, weights, p, Window_size, tsfresh_feature, Difference])

        return cs
