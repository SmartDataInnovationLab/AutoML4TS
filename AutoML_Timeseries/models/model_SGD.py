import sys
sys.path.append("..")

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, UnParametrizedHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition, EqualsCondition

from  Forecasting import Automated_ML_Forecasting
from sklearn.linear_model.stochastic_gradient import SGDRegressor

class SGD_Forecasting(Automated_ML_Forecasting):
    def __init__(self, timeseries, dataname,
                 # parameter for SGD regression
                 loss="squared_loss", penalty="l2", alpha=0.0001, fit_intercept=True, tol=1e-3,
                 learning_rate="invscaling", l1_ratio=0.15, epsilon=0.1, max_iter=1000,
                 eta0=0.01, power_t=0.5, average=False,
                 #feature extraction parameter
                 Window_size = 20 , Difference = False, 
                 time_feature = True,  tsfresh_feature=True,
                 forecasting_steps = 25, n_splits = 5,
                 max_train_size = None,  NAN_threshold = 0.05):

        self.loss = loss
        self.penalty = penalty
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.l1_ratio = float(l1_ratio) if l1_ratio is not None else 0.15
        self.epsilon = float(epsilon) if epsilon is not None else 0.1
        self.max_iter = int(max_iter)
        self.eta0 = float(eta0)
        self.power_t = float(power_t) if power_t is not None else 0.25
        self.average = average

        self.estimator = SGDRegressor(loss=self.loss,
                                      penalty=self.penalty,
                                      alpha=self.alpha,
                                      fit_intercept=self.fit_intercept,
                                      max_iter=self.max_iter,
                                      tol=self.tol,
                                      learning_rate=self.learning_rate,
                                      l1_ratio=self.l1_ratio,
                                      epsilon=self.epsilon,
                                      eta0=self.eta0,
                                      power_t=self.power_t,
                                      average=self.average,
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

        tsfresh_feature = CategoricalHyperparameter(
            name="tsfresh_feature", choices=["True", "False"], default_value="True")
			
        Difference = CategoricalHyperparameter(
            name="Difference", choices=["True", "False"], default_value="True")

        loss = CategoricalHyperparameter(
            "loss",
            ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
            default_value="squared_loss")

        penalty = CategoricalHyperparameter(
            "penalty", ["l1", "l2", "elasticnet"], default_value="l2")

        alpha = UniformFloatHyperparameter(
            "alpha", 1e-7, 1e-1, log=True, default_value=0.0001)

        l1_ratio = UniformFloatHyperparameter(
            "l1_ratio", 1e-9, 1., log=True, default_value=0.15)
       
        tol = UniformFloatHyperparameter(
            "tol", 1e-4, 1e-1, default_value=1e-3, log=True)
        epsilon = UniformFloatHyperparameter(
            "epsilon", 1e-5, 1e-1, default_value=0.1, log=True)
        learning_rate = CategoricalHyperparameter(
            "learning_rate", ["optimal", "invscaling", "constant"],
            default_value="invscaling")
        eta0 = UniformFloatHyperparameter(
            "eta0", 1e-7, 1e-1, default_value=0.01)
        power_t = UniformFloatHyperparameter(
            "power_t", 1e-5, 1, default_value=0.25)
        average = CategoricalHyperparameter(
            "average", ["False", "True"], default_value="False")

        # un parametrized parameter
        fit_intercept = UnParametrizedHyperparameter(
            "fit_intercept", "True")

        max_iter = UnParametrizedHyperparameter("max_iter", 1000)

        cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept,
                                tol, epsilon, learning_rate, eta0, Difference,
                                power_t, average, Window_size, tsfresh_feature, max_iter])


        elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
        epsilon_condition = InCondition(epsilon, loss,
            ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"])

        power_t_condition = EqualsCondition(power_t, learning_rate,
                                            "invscaling")

        cs.add_conditions([elasticnet, epsilon_condition, power_t_condition])

        return cs
