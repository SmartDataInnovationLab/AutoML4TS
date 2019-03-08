import sys
sys.path.append("..")
from sklearn.svm import SVR
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter
from  Forecasting import Automated_ML_Forecasting
"""{TODO} log = True or False"""
class SVR_Forecasting(Automated_ML_Forecasting):

    def __init__(self, timeseries, dataname,
                 #SVR model parameter
                 C=2.0, cache_size=200, coef0=0.0, epsilon=0.01,
                 tol=0.001, kernel='rbf', shrinking= True, degree=3,
                 gamma=0.1, verbose=False, max_iter=200000, random_state=None,
                 #feature extraction parameter
                 Window_size = 20 , Difference = False, 
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):

        
        self.kernel = kernel
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.tol = float(tol)
        self.shrinking = shrinking
        self.degree = int(degree)
        self.gamma = gamma
        if coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(coef0)
        self.verbose = int(verbose)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        
        self.estimator = SVR(kernel=self.kernel,
                             C=self.C,
                             epsilon=self.epsilon,
                             tol=self.tol,
                             shrinking=self.shrinking,
                             degree=self.degree,
                             gamma=self.gamma,
                             coef0=self.coef0,
                             cache_size=cache_size,
                             verbose=self.verbose,
                             max_iter=self.max_iter)
        
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

        Window_size = UniformIntegerHyperparameter(
            name="Window_size", lower=5, upper=50, default_value=20)

        Difference = CategoricalHyperparameter(
            name="Difference", choices=["True", "False"], default_value="True")

        tsfresh_feature = CategoricalHyperparameter(
            name="tsfresh_feature", choices=["True", "False"], default_value="True")

        C = UniformFloatHyperparameter(
            name="C", lower=0.03125, upper=32768, log=True, default_value=1.0)
        
        epsilon = UniformFloatHyperparameter(name="epsilon", lower=0.001,
                                             upper=1, default_value=0.1,
                                             log=True)

        kernel = CategoricalHyperparameter(
            name="kernel", choices=['linear', 'poly', 'rbf', 'sigmoid'],
            default_value="rbf")

        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=3)

        gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")

        gamma_value = UniformFloatHyperparameter(
            name="gamma_value", lower=0.0001, upper=8, default_value=1)

        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter(
            name="coef0", lower=-1, upper=1, default_value=0)

        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter(
            name="shrinking", choices=["True", "False"], default_value="True")

        tol = UniformFloatHyperparameter(
            name="tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)

        max_iter = UnParametrizedHyperparameter("max_iter", 200000)

        
        cs.add_hyperparameters([Window_size, Difference, tsfresh_feature,C, kernel, degree, gamma, gamma_value, coef0, shrinking,
                               tol, max_iter, epsilon])

        degree_depends_on_kernel = InCondition(child=degree, parent=kernel,
                                               values=["poly"])
        gamma_depends_on_kernel = InCondition(child=gamma, parent=kernel,
                                              values=["rbf", "poly", "sigmoid"])
        coef0_depends_on_kernel = InCondition(child=coef0, parent=kernel,
                                              values=["poly", "sigmoid"])
        gamma_value_depends_on_gamma = InCondition(child=gamma_value, parent=gamma, 
                                              values=["value"])
        cs.add_conditions([degree_depends_on_kernel, gamma_depends_on_kernel,
                           coef0_depends_on_kernel, gamma_value_depends_on_gamma])

        return cs 
        
