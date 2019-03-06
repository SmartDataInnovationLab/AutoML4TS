import sys
sys.path.append("..")
import xgboost as xgb
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter
from  Forecasting import Automated_ML_Forecasting
import numpy as np


class Xgboost_Forecasting(Automated_ML_Forecasting):
    
    def __init__(self, timeseries, dataname,
                 #xgboost model parameter
                 learning_rate=0.1, n_estimators=100, subsample=1.0,
                 max_depth=3, colsample_bylevel=1, colsample_bytree=1, gamma=0,
                 min_child_weight=1, max_delta_step=0, reg_alpha=0, reg_lambda=1,
                 base_score=0.5, scale_pos_weight=1, nthread=1,
                 random_state=None, verbose=0,
                 #feature extraction parameter
                 Window_size = 20 , Difference = False,
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):
        
        
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.max_depth = int(max_depth)
        self.colsample_bylevel = float(colsample_bylevel)
        self.colsample_bytree = float(colsample_bytree)
        self.gamma = float(gamma)
        self.min_child_weight = int(min_child_weight)
        self.max_delta_step = int(max_delta_step)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.base_score = float(base_score)
        self.scale_pos_weight = float(scale_pos_weight)
        self.nthread = int(nthread)
        
        if verbose:
            self.silent = False
        else:
            self.silent = True
            
        if random_state is None:
            self.seed = np.random.randint(1, 10000, size=1)[0]
        else:
            self.seed = random_state.randint(1, 10000, size=1)[0]
            
        self.objective = 'reg:linear'
        
        self.estimator = xgb.XGBRegressor(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                silent=self.silent,
                objective=self.objective,
                nthread=self.nthread,
                gamma=self.gamma,
                scale_pos_weight=self.scale_pos_weight,
                min_child_weight=self.min_child_weight,
                max_delta_step=self.max_delta_step,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                colsample_bylevel=self.colsample_bylevel,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                base_score=self.base_score,
                seed=self.seed
                )
        
        super().__init__(timeseries,dataname,Window_size, time_feature, Difference, tsfresh_feature,
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

        # Parameterized Hyperparameters
        Window_size = UniformIntegerHyperparameter(
            name="Window_size", lower=5, upper=50, default_value=20)

        tsfresh_feature = CategoricalHyperparameter(
            name="tsfresh_feature", choices=["True", "False"], default_value="True")

        Difference = CategoricalHyperparameter(
            name="Difference", choices=["True", "False"], default_value="True")


        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=1, upper=30, default_value=3)
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=False)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
        subsample = UniformFloatHyperparameter(
            name="subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
        min_child_weight = UniformIntegerHyperparameter(
            name="min_child_weight", lower=1, upper=20, default_value=1, log=False)

        # Unparameterized Hyperparameters


        cs.add_hyperparameters([max_depth, learning_rate, n_estimators, 
                                Difference, subsample, min_child_weight,
                                Window_size, tsfresh_feature ])

        return cs
