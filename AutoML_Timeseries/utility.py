import pandas as pd
import statsmodels as sm
import numpy as np
from tsfresh.feature_selection.significance_tests import target_real_feature_real_test
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
import tsfresh
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh import extract_features, select_features


check_date_number = {"second"   : 60,
                     "minute"   : 60,
                      "hour"    : 24,
                      "week"    : 53,
                      "weekday" : 7,
                      "month"   : 12,
                      "day"     : 31}

def get_nanosecond(time_index):
    """The nanoseconds of the datetime"""
    return np.array(time_index.nanosecond).reshape(-1,1)

def get_second(time_index):
    """The second of the datetime"""
    return np.array(time_index.second).reshape(-1,1)

def get_minute(time_index):
    """The minute of the datetime"""
    return np.array(time_index.minute).reshape(-1,1)

def get_hour(time_index):
    """The hour of the datetime"""
    return np.array(time_index.hour).reshape(-1,1)

def get_week(time_index):
    """The week of the datetime"""
    return np.array(time_index.week).reshape(-1,1)

def get_weekday(time_index):
    """The weekday of the datetime"""
    return np.array(time_index.weekday()).reshape(-1,1)

def get_month(time_index):
    """The month of the datetime"""
    return np.array(time_index.month).reshape(-1,1)

def get_year(time_index):
    """The year of the datetime"""
    return np.array(time_index.year).reshape(-1,1)

def get_day(time_index):
    """The day of the datetime"""
    return np.array(time_index.day).reshape(-1,1)



get_time_index = {"nanosecond": get_nanosecond,
                  "second"    : get_second,
                  "minute"    : get_minute,
                  "hour"      : get_hour,
                  "week"      : get_week,
                  "weekday"   : get_weekday,
                  "month"     : get_month,
                  "day"       : get_day,
                  "year"      : get_year}

scaled_index = ['nanosecond', 'year']

def check_the_parameters(parameter, name, type_of_parameter, min_value=None, max_value=None):
    """ 
    check the type and value of the parameter
    """
    # check the type
    if not isinstance(parameter, type_of_parameter):
        raise ValueError('%s must be of %s type. '
                         '%s of type %s was passed.'
                         % (name ,type_of_parameter,parameter, type(parameter)))
    # check the values
    if min_value:
        if parameter < min_value:
            raise ValueError('%s should be bigger than %s'
                             % (name ,min_value))
    if max_value:
        if parameter > max_value:
            raise ValueError('%s should be smaller than %s'
                             % (name ,max_value))
							 
							 
class Time_series_forecasting_split(object):

    """
    Time Series cross-validator

    Provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets. 
    In each split, test indices must be higher than before, and thus shuffling in cross validator is inappropriate.

    This cross-validation object is a variation of KFold. In the kth split, it returns first k folds as train set and the (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive training sets are supersets of those that come before them.


    """
    def __init__(self, n_splits = 5, forecasting_steps = 30,  max_train_size=None):
        """
        Parameters:	

        n_splits : int, default=5,   Number of splits. Must be at least 1.

        forecasting_steps : int , default=30, the size of test set 

        max_train_size : float, optional , Maximum size for a single training set : in range [0,1]
        """
        self._n_splits = n_splits
        self._forecasting_steps = forecasting_steps
        self._max_train_size = max_train_size

    def split(self, X):
        #     x : timeseries which is used to split
        n_samples = X.shape[0] - self._forecasting_steps 
        #     split beginn at 20% of the length
        base = int(n_samples*0.2)
        rest = n_samples - base
        n_splits = self._n_splits
        n_folds = n_splits + 1
        
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (rest // n_folds)
        test_starts = range(base + test_size + rest % n_folds,
                            n_samples+1, test_size)
        #     a list of index range
        result = list()
        if self._max_train_size:
            max_train_size = int(self._max_train_size * n_samples)
        else:
            max_train_size = None

        for test_start in test_starts:
            if max_train_size and max_train_size < test_start:
                result.append(indices[test_start - max_train_size:test_start])
            else:
                result.append(indices[:test_start])
        return result



def extract_values_in_windows_as_feature(timeseries, window_size):

    values_feature = pd.DataFrame(timeseries.values, index=timeseries.index, columns=["data"])
    for i in range(window_size, 0 ,-1):
        values_feature['feature_last_{}_value'.format(i)] = timeseries.shift(i)  #Move Downward
    y = values_feature.iloc[window_size:, 0]
    values_feature = values_feature.iloc[window_size:,1:]
    return values_feature, y
	

def frequency(timeindex):
    """
    periodicity feature extraction

    (1)Find r_k = Corr( Y_t , Y_(t−k))(autocorrelation function) for all lags up to 1/3 of series length, then look for peaks and troughs in autocorrelation function.
    (2)Frequency is the first peak satisfying the following conditions: 
        a) there is also a trough before it; 
        b) the difference between peak and trough is at least 0.1; 
        c) the peak corresponds to positive correlation.
    (3)If no such peak is found, frequency is set to 1 (equivalent to non-seasonal).

    """
    # step 1
    autocorrelation = sm.tsa.stattools.acf(timeindex, nlags=int(len(timeindex/3)))
    autocorrelation = np.array(autocorrelation)
    # step 2
    peaks = (autocorrelation > np.roll(autocorrelation,1)) & (autocorrelation > np.roll(autocorrelation,-1))
    troughs = (autocorrelation < np.roll(autocorrelation,1)) & (autocorrelation < np.roll(autocorrelation,-1))
    peaks_index = np.where(peaks==True)[0]
    peaks_index = np.stack((peaks_index,np.ones(peaks_index.shape).astype(np.int)), axis=1)
    troughs_index = np.where(troughs==True)[0]
    troughs_index = np.stack((troughs_index,np.zeros(troughs_index.shape).astype(np.int)), axis=1)
    peak_and_trough = np.vstack((troughs_index,peaks_index))
    peak_and_trough = peak_and_trough[peak_and_trough[:,0].argsort()]
    freq = 1
    try:
        if peak_and_trough.shape[0]>2:
            for i in range(2,peak_and_trough.shape[0]):
                if peak_and_trough[i,1]==1 and peak_and_trough[i-1,1]==0 and \
                autocorrelation[peak_and_trough[i,0]]-autocorrelation[peak_and_trough[i-1,0]]>0.1 and \
                autocorrelation[peak_and_trough[i,0]]>0.2 and \
                autocorrelation[peak_and_trough[i-1,0]]<0:
                    freq = peak_and_trough[i,0]
                    break
    except:
        freq=1
    
    return freq


def extract_time_features_DataFrame(Features_DataFrame, timeseries):
    
    # timeseries : according to the window size , extracted y values
    # Features_DataFrame : corresponding last values.
    # these two should have same time index
    
    time_features = pd.DataFrame(index=Features_DataFrame.index)
    time_features["nanosecond"] = Features_DataFrame.index.nanosecond
    time_features["second"] =Features_DataFrame.index.second
    time_features["minute"] =Features_DataFrame.index.minute
    time_features["hour"] =Features_DataFrame.index.hour
    time_features["week"] =Features_DataFrame.index.week
    time_features["weekday"] =Features_DataFrame.index.weekday
    time_features["month"] =Features_DataFrame.index.month
    time_features["year"] =Features_DataFrame.index.year
    time_features["day"] =Features_DataFrame.index.day
    
    scaler_index =[] # for year, nanosecond
    
    # drop the unique features
    time_features = time_features.loc[:, time_features.apply(pd.Series.nunique) != 1]
    
    # drop irrelevant features
    columns = time_features.columns
    for column in columns:
        p = target_real_feature_real_test(time_features[column], timeseries)
        if p > 0.05:
            del time_features[column]
    
    
    for column in time_features.columns:
        if column not in scaled_index:
            period = check_date_number[column] / (2 * np.pi)
            time_features[column] = time_features[column] / period
            time_features['{}_sin'.format(column)] = np.around((np.sin(time_features[column])+1)/2,decimals=3)
            time_features['{}_cos'.format(column)] = np.around((np.cos(time_features[column])+1)/2,decimals=3)
            del time_features[column]
        
        


    scalers = OrderedDict()
    for column in time_features.columns:
        if column in scaled_index:
            array = np.array(time_features[column])
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled = scaler.fit_transform(array.reshape(-1,1)).reshape(-1)
            time_features[column] = scaled
            scalers[column] = scaler

    return scalers, time_features, time_features.columns
	
def extract_tsfresh_features(timeseries, window_size, threshold):

    # tsfresh make_forecasting_frame rolling window
    df_shift, y = make_forecasting_frame(timeseries, kind="x", 
                                         max_timeshift = window_size, 
                                         rolling_direction=1)

    settings_original = EfficientFCParameters()
    
    # caculate all features
    All_features = extract_features(df_shift, column_id="id", 
                                    column_sort="time", column_value="value", 
                                    default_fc_parameters=settings_original,
                                    impute_function=None, disable_progressbar=True,
                                    show_warnings=False, n_jobs=8)
    
    # drop the the first window size values
    All_features = All_features.iloc[window_size-1:]
    y = y.iloc[window_size-1:]

    # tsfresh fileter out relevant featrues through significant test
    #kind_to_fc_parameters =  filter_features(All_features, y, threshold)
    
    
    #drop columns witch are all nan
    All_features = All_features.dropna(axis=1,how='all')

    # nan percentage
    nan_percentage = (All_features.shape[0] - All_features.count())/All_features.shape[0]
    index = nan_percentage.index
    for i in range(0,len(nan_percentage)):
        if nan_percentage[i]>threshold:
            del All_features[index[i]]

    # drop constant features
    All_features = All_features.loc[:, All_features.apply(pd.Series.nunique) != 1] 
    All_features.replace([np.inf, -np.inf], np.nan)
    if All_features.isnull().values.any():
        All_features = All_features.fillna(All_features.mean())

    #filter out not important features
    All_features = select_features(All_features, y)

    
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(All_features)


    if len(kind_to_fc_parameters) > 0:
        temp = extract_features(df_shift.iloc[:3,:], column_id="id", 
                                column_sort="time", column_value="value", kind_to_fc_parameters=kind_to_fc_parameters,
                                impute_function=None, disable_progressbar=True,
                                show_warnings=False, n_jobs=8)

        All_features = All_features[temp.columns & All_features.columns]

      
        return All_features, False
    
    else:
        
        return None, True
		
		
def feature_sigmoid_transform(Featrues_DataFrame):
    """
    transform the operations into range(0,1)
    return transformed feature Frames , the median and IQR for each feature(columns) for new feature transformation
    this is excuted only when there are tsfresh fearues
    """

    # Interquartile range for each features
    Interquartile0_25 = np.array(Featrues_DataFrame.quantile(0.25))
    Interquartile0_75 = np.array(Featrues_DataFrame.quantile(0.75))
    IQR = Interquartile0_75 - Interquartile0_25

    # Median for each features
    median = np.array(Featrues_DataFrame.median())

    feature_columns = Featrues_DataFrame.columns
    # drop the features which has fast 0.0 Interquartile
    position = np.where( IQR == 0 )[0]

    Featrues_DataFrame.drop(Featrues_DataFrame.columns[position], axis=1, inplace = True)

    median = np.delete(median,position)
    IQR = np.delete(IQR,position)

    # sigmoid transformation
    transformed_features = (1+np.exp((-Featrues_DataFrame + median)/(IQR*1.35)))**(-1)

    return median, IQR, transformed_features
	
	
	
def add_and_scale_time_features(forecasting_index, step, time_scalers, time_feature_columns):
    # the time index of the to be forecasted step
    time_index = forecasting_index[step]
    
    # according to existing time features, make the time feature dataframe. columns name must be the same
    features = pd.DataFrame(np.zeros(len(time_feature_columns)).reshape(1,-1),columns=time_feature_columns)


    for column in features.columns:
        if "_" in column:
            name = column.split("_")[0]
            method = column.split("_")[1]

            index = get_time_index[name](time_index)
            if method == "sin":
                features[column] = np.around((np.sin(index*2*np.pi/check_date_number[name])+1)/2,decimals=3)
            else :
                features[column] = np.around((np.cos(index*2*np.pi/check_date_number[name])+1)/2,decimals=3)
        else:
            index = get_time_index[column](time_index)
            features[column] = time_scalers[column].transform(index).reshape(-1)

    return features
	
	
	
	
def tsfresh_features_extraction_of_sub_time_series(DataSeries, kind_to_fc_parameters, ts_features_columns):

    DataFrame = pd.DataFrame(DataSeries)
    DataFrame.reset_index(inplace=True)
    DataFrame.columns = ["time", "value"]
    DataFrame["id"] = 1

    Features = extract_features(DataFrame, column_id="id", column_sort="time", 
                                kind_to_fc_parameters=kind_to_fc_parameters,
                                column_value="value", impute_function=None, disable_progressbar=True,
                                show_warnings=False, n_jobs=0)
    Features = Features[ts_features_columns & Features.columns]
    return Features
