import sys
sys.path.append("..")

from sklearn.ensemble import RandomForestRegressor
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
	
from Forecasting import Automated_ML_Forecasting

class Random_forest_Forecasting(Automated_ML_Forecasting):
    
    def __init__(self,timeseries, dataname, 
                 #Random Forest parameter
                 n_estimators=100, criterion='mse', max_features=1.0,
                 max_depth='None', min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0., bootstrap=True, max_leaf_nodes='None',
                 min_impurity_decrease=0.,
                 #feature extraction parameter
                 Window_size = 20 , Difference = False, 
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):
        
        self.n_estimators = int(n_estimators)
        self.criterion = criterion  #str
        self.max_features = float(max_features)
        if max_depth == "None" or max_depth is None:
            self.max_depth = None
        else:
            self.max_depth = int(max_depth)
            
        
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
            
        if max_leaf_nodes == "None" or max_leaf_nodes is None:
            self.max_leaf_nodes = None
            
        self.min_impurity_decrease = float(min_impurity_decrease)
        
        self.estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,                         #100
                criterion=self.criterion,                               #'mse'
                max_features=self.max_features,                         #1.0
                max_depth=self.max_depth,                               #None
                min_samples_split=self.min_samples_split,               #2           
                min_samples_leaf=self.min_samples_leaf,                 #1
                min_weight_fraction_leaf=self.min_weight_fraction_leaf, #0.
                bootstrap=self.bootstrap,                               #True
                max_leaf_nodes=self.max_leaf_nodes,                     #None
                min_impurity_decrease=self.min_impurity_decrease,       #0.
                warm_start=True)

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
        
        n_estimators = UniformIntegerHyperparameter("n_estimators", 10, 200, default_value=100)
        
        criterion = CategoricalHyperparameter("criterion",
                                              ['mse', 'friedman_mse', 'mae'], default_value='mse')
        
        max_features = UniformFloatHyperparameter(
            "max_features", 0.1, 1.0, default_value=1.0)
        
        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 2, 20, default_value=2)
        
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default_value=1)
        
        min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
            
        max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
        
        min_impurity_decrease = UnParametrizedHyperparameter(
            'min_impurity_decrease', 0.0)
        
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default_value="True")

        cs.add_hyperparameters([n_estimators, criterion, max_features, Difference,
                                max_depth, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes,
                                min_impurity_decrease, bootstrap, tsfresh_feature, Window_size])

        return cs
