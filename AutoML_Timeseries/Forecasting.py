import utility
import numbers
import os
import pickle
import warnings
import tsfresh
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import visualize
warnings.filterwarnings("ignore")


import utility
import numbers
import os
import pickle
import warnings
import tsfresh
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import visualize
warnings.filterwarnings("ignore")


class Automated_ML_Forecasting():
    def __init__(self, 
             timeseries,                     # to be analysed time-seires
             dataname,                       # The name of your time-series
             Window_size = 20 ,              # Hyperparameter to optimize, the size of windows, int ,  in range [5, 100]
             #""" needs to be modified"""
             Time_feature = True,            # Hyperparameter to optimize, whether to use tsfresh features : True or False
             Difference = False,             # Hyperparameter to optimize, whether to difference the time-series : True or False
             Tsfresh_feature=True,           # Hyperparameter to optimize, whether to use tsfresh features : True or False
             Forecasting_steps = 25,         # the step of forecasting
             n_splits = 5,
             max_train_size = None, 
             NAN_threshold = 0.05):


        self._window_size = Window_size
        self._time_feature = Time_feature
        self._forecasting_steps = Forecasting_steps
        self._tsfresh_feature = Tsfresh_feature 
        self._difference = Difference

        self._n_splits = n_splits
        self._max_train_size = max_train_size
        self._nan_threshold = NAN_threshold

        """check all paramaters"""
        self._check_the_input()


        """define the path to save the features"""

        # calculate the features 
        # because of using rolloing window method, windows size varies between 
        self._dir_name = 'feature/{}/'.format(dataname)
        if not os.path.exists(self._dir_name):
            os.makedirs(self._dir_name)



        if self._tsfresh_feature:
            if self._difference:
                # featrues file with tsfresh features of differenced timeseries
                self._features_file_name = '{}_window_size_diff.pkl'.format(Window_size)
            else:
                # featrues file with tsfresh features of raw timeseries
                self._features_file_name = '{}_window_size.pkl'.format(Window_size)
        else :
            if self._difference:
                # fearues file of differenced timeseries without tsfresh features 
                self._features_file_name = '{}_window_size_no_ts_diff.pkl'.format(Window_size)
            else:
                # fearues file without tsfresh features
                self._features_file_name = '{}_window_size_no_ts.pkl'.format(Window_size)
                
        #self._timeseries_median = 0
        #self._timeseries_IQR = 0
                
        if self._difference:
            self._timeseries = timeseries.iloc[self._window_size+1:]
            self._start = timeseries.iloc[self._window_size]
            timeseries = timeseries.diff().iloc[1:]
            index = timeseries.index
            self._scaler = MinMaxScaler(feature_range=(0, 1))
            Y = self._scaler.fit_transform(np.array(timeseries).reshape(-1,1)).reshape(-1)
            timeseries = pd.Series(Y, index= index)
        #else:
        #    self._timeseries_median = timeseries.median()
        #    self._timeseries_IQR = timeseries.quantile(0.75) -timeseries.quantile(0.25)


        if not os.path.exists(self._dir_name+self._features_file_name):
            # if the features is not calculated, make a file dictionary , calculate features , store features

            self._all_features, self._y, self._time_scalers, \
            self._timefeature_columns, self._no_ts_flag = self._extract_features_for_training(timeseries)
            self._transform_the_ts_features()
            self._save_all_features()

        else:

            # if the features is already calculated, load the features
            self._all_features, self._y, self._time_scalers, self._timefeature_columns, \
            self._no_ts_flag, self._median, self._IQR, self._kind_to_fc_parameters, self._ts_features_columns  = self._load_all_features() 
            
            

    def _transform_the_ts_features(self):
        # Transform the tsfresh features to range (0, 1)
        self._median = None 
        self._IQR = None
        self._kind_to_fc_parameters = None
        self._ts_features_columns = None
        if not self._no_ts_flag and self._tsfresh_feature:
            train_x_ts = self._all_features.iloc[:, :-(self._timefeature_columns.shape[0]+self._window_size)]
            train_x_time = self._all_features.iloc[:, -(self._timefeature_columns.shape[0]+self._window_size):]
            
            self._median, self._IQR, transformed_features = utility.feature_sigmoid_transform(train_x_ts)
            if transformed_features.shape[1]>0:
                if transformed_features.isnull().values.any():
                    transformed_features = transformed_features.fillna(transformed_features.mean())

                self._kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(transformed_features)


                self._ts_features_columns = transformed_features.columns
            
                self._all_features = pd.concat([transformed_features,train_x_time], axis = 1)
            else:
                self._no_ts_flag = True
                
        # Transform the raw values features to range (0, 1)
        #if self._timeseries_IQR != 0:
        #    self._all_features.iloc[:, -(self._timefeature_columns.shape[0] + self._window_size):-self._timefeature_columns.shape[0]] = \
        #        (1+np.exp((-self._all_features.iloc[:, -(self._timefeature_columns.shape[0] + self._window_size):-self._timefeature_columns.shape[0]] + self._timeseries_median)/(self._timeseries_IQR*1.35)))**(-1)
                
                

    def _check_the_input(self):

        # check the time-series
        utility.check_the_parameters(self._window_size, 'Window size', numbers.Integral, 5, 99)
        utility.check_the_parameters(self._forecasting_steps, 'forecasting steps', numbers.Integral, 1)
        utility.check_the_parameters(self._n_splits, 'n splits', numbers.Integral, 1)


    def _extract_features_for_training(self, timeseries):
        """
        Features for forcasting consists of three parts: 1. Tsfresh features of the window 
                                                         2. raw values in the window, the number of raw values equals the size of the window
                                                         3. the time index of the predicted steps

        """


        # raw values feature as default 
        values_feature , y = utility.extract_values_in_windows_as_feature(timeseries, self._window_size)

        # time index features
        time_scalers, time_features, timefeature_columns= (None, None, None)
        ts_features , no_ts_flag = (None, False)

        if self._time_feature:
            time_scalers, time_features, timefeature_columns = utility.extract_time_features_DataFrame(values_feature, y)

        # tsfresh features
        if self._tsfresh_feature:
            ts_features, no_ts_flag = utility.extract_tsfresh_features(timeseries, self._window_size, self._nan_threshold)

        all_features = pd.concat([ts_features, values_feature, time_features], axis=1)

        return all_features, y, time_scalers, timefeature_columns, no_ts_flag

    def _save_all_features(self):
        features_info = {'all_features':self._all_features,
                         'y':self._y,
                         'time_scalers':self._time_scalers,
                         'timefeature_columns':self._timefeature_columns,
                         'no_ts_flag':self._no_ts_flag,
                         'median':self._median,
                         'IQR':self._IQR,
                         'kind_to_fc_parameters':self._kind_to_fc_parameters,
                         'ts_features_columns':self._ts_features_columns}

        with open(self._dir_name +self._features_file_name, 'wb') as f:
            pickle.dump(features_info, f, pickle.HIGHEST_PROTOCOL)   


    def _load_all_features(self):
        """
        According to the self._features_dir_name ( features dictionary), if the features have been calculated , load the features 
        """
        with open(self._dir_name +self._features_file_name, 'rb') as f:
            features_info = pickle.load(f)
        return (features_info['all_features'],features_info['y'],features_info['time_scalers'],features_info['timefeature_columns'], 
                features_info['no_ts_flag'], features_info['median'], features_info['IQR'], features_info['kind_to_fc_parameters'],  features_info['ts_features_columns']) 



    def forecasting(self, estimator, index):
        """
        Input: model:  regression model
               index:  the to be forecasted index
               ts_features_columns : tsfresh features
        """
        prediction = list()
        end_index = index
        start_index = end_index - self._window_size
        
        # the beginning sub time series, which attemps to forecast the first value in the future
        sub_timeseries = self._y.iloc[start_index:end_index]
        forecasting_index = self._y[end_index:end_index+self._forecasting_steps].index


        for step in range(0,self._forecasting_steps):

            features = self._extract_features_for_forecasting(sub_timeseries, forecasting_index, step)

            y_pred = estimator.predict(np.array(features).reshape(1, -1)) 

            prediction.append(y_pred)

            
            # form the new sub-timeseires which the last value ist the forecasted values
            sub_timeseries = sub_timeseries.shift(-1)
            sub_timeseries[-1] = y_pred

        return  pd.Series(np.array(prediction).reshape(-1), index= forecasting_index)

    def _extract_features_for_forecasting(self, sub_timeseries, forecasting_index, step):
 

        #if sub_timeseries.isnull().values.any():
            #sub_timeseries.interpolate(method="krogh", inplace=True)
            #print(sub_timeseries)
        #sub_timeseries = sub_timeseries.replace(np.inf,20)
        #sub_timeseries = sub_timeseries.replace(-np.inf,-20)
 
        length = len(sub_timeseries)
        features = pd.DataFrame()
        
        """default values as features"""
        for i in range(1,length+1):
            features['feature_last_{}_value'.format(i)] = [sub_timeseries[-i]]
        features.index = [1]
        # transform the features
        #if self._timeseries_IQR != 0:
        #    features = (1+np.exp((-features + self._timeseries_median)/(self._timeseries_IQR*1.35)))**(-1)
        #    print(features.shape)
        
        """TSFRESH features as features"""
        if self._tsfresh_feature and not self._no_ts_flag and self._kind_to_fc_parameters is not None:

            tsfresh_features = utility.tsfresh_features_extraction_of_sub_time_series(sub_timeseries, self._kind_to_fc_parameters, self._ts_features_columns)

            assert (self._median is not None ), "_median for transformation is None"
            assert (self._IQR is not None),  "_IQR for transformation is None"

            tsfresh_features = (1+np.exp((-tsfresh_features + self._median)/(self._IQR*1.35)))**(-1)
			
            tsfresh_features = tsfresh_features.replace(-np.inf,np.NAN)
            tsfresh_features = tsfresh_features.replace(np.inf,np.NAN)

            # if there is nan in features, replace it with caculated mean
            if tsfresh_features.isnull().values.any():
                features_mean_values = self._median
                position = np.where(np.isnan(tsfresh_features))[1]
                for i in range(0,len(position)):
                    tsfresh_features.iloc[0,position[i]] = features_mean_values[position[i]]
                    
            features = pd.concat([tsfresh_features,features], axis = 1)
    
        
        """Time index as features"""
        if self._time_feature == True:
            
            time_features = utility.add_and_scale_time_features(forecasting_index, step, self._time_scalers, self._timefeature_columns)
            time_features.index = [1]
            features = pd.concat([features,time_features], axis = 1)

        return features
    
    
    
    
    def _cross_and_val(self, estimator):
        performance = list()
        for prediction, groundtruth in self._Time_Series_forecasting_cross_validation(estimator):
            
            # TODO according to metric , choose which score to be calculated
            performance.append(np.mean((prediction-groundtruth)**2))
        return np.mean(performance)    
    
    
    
    def _Time_Series_forecasting_cross_validation(self, estimator):

        
        time_series_spliter = utility.Time_series_forecasting_split(self._n_splits, self._forecasting_steps, self._max_train_size)


        for train_index in time_series_spliter.split(self._y):
            
            # extract the train x and y for sub training
            train_y = self._y.iloc[train_index[0]:train_index[-1]+1]
            train_x = self._all_features.iloc[train_index[0]:train_index[-1]+1]

            estimator.fit(np.array(train_x), np.array(train_y))
            
            if self._difference:
                
                y_predict = self.forecasting(estimator, train_index[-1]+1)
                y_groundtruth = self._timeseries[train_index[-1]+1 : train_index[-1]+1+self._forecasting_steps]
                 # inverse scaler
                y_predict = pd.Series(self._scaler.inverse_transform(y_predict.values.reshape(-1,1)).reshape(-1), index=y_predict.index)
                 # inverse difference
                y_predict = y_predict.cumsum() + self._timeseries.iloc[train_index[-1]]
            else:
                y_predict = self.forecasting(estimator, train_index[-1]+1)
                y_groundtruth = self._y[train_index[-1]+1 : train_index[-1]+1+self._forecasting_steps]
 
            yield y_predict, y_groundtruth    
    
    
    
    
    def _cross_validation_visualization(self, estimator, only_prediction=False):
        
        folders = self._n_splits + 1 
        
        prediction_list = []
        groundtruth_list = []
        
        for prediction, groundtruth in self._Time_Series_forecasting_cross_validation(estimator):
            prediction_list.append(prediction)
            groundtruth_list.append(groundtruth)
            
        
        if self._difference:
            visualize.cross_validation_visualization(folders, self._timeseries, prediction_list, groundtruth_list, only_prediction)
        else:
            visualize.cross_validation_visualization(folders, self._y, prediction_list, groundtruth_list, only_prediction)
        
    

    
    def _direct_prediction(self, estimator, times = 3):
        """ visualize the forecast for the last 3*forecasting steps """
        #split = self._all_features.shape[0] - times*self._forecasting_steps
        
        # split the 
        split = self._all_features.shape[0] - 3*self._forecasting_steps
        split = max(split,int(self._y.shape[0]*3/4))
        if self._max_train_size is None:
            start = 0
            train_x = self._all_features.iloc[:split]
            train_y = self._y.iloc[:split]
        else:
            start = max(0,split-int(self._all_features.shape[0]*self._max_train_size))
            train_x = self._all_features.iloc[start:split]
            train_y = self._y.iloc[start:split]

        original_forecasting_step = self._forecasting_steps
        self._forecasting_steps = self._all_features.shape[0] - split

        estimator.fit(np.array(train_x), np.array(train_y))


        if self._difference:
            # Do the prediction
            prediction = self.forecasting(estimator, split)
            
            # Transform the prediction becuase differencing : inverse scaler and inverse difference
            prediction = pd.Series(self._scaler.inverse_transform(prediction.values.reshape(-1,1)).reshape(-1), index=prediction.index)
            prediction = prediction.cumsum() +  self._timeseries.iloc[split-1]
            
            # Calculate the train prediction 
            train_prediction = estimator.predict(train_x)
            train_prediction = pd.Series(self._scaler.inverse_transform(train_prediction.reshape(-1,1)).reshape(-1), index=train_x.iloc[:split].index)
            timeseries_shift = self._timeseries.iloc[:split].shift(1)
            timeseries_shift.iloc[0] = self._start
            train_prediction = train_prediction + timeseries_shift
            
            
            # Do Visualization
            
            visualize.prediction_visualization(self._timeseries,start, split, prediction, train_prediction)
            self._forecasting_steps = original_forecasting_step

        else: 
            # Do the prediction
            prediction = self.forecasting(estimator, split)
            # training performance . 1 step forecasting
            train_prediction = estimator.predict(np.array(train_x))
            
            train_prediction = pd.Series(train_prediction, index=train_x.index)
            #  Visualisieren
            visualize.prediction_visualization(self._y, start, split, prediction, train_prediction)
            self._forecasting_steps = original_forecasting_step
