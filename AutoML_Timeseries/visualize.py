import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def cross_validation_visualization(folders, timeseries, prediction_list, groundtruth_list, only_prediction):

    # TODO legend and shadow area

    plt.figure(figsize=(18,folders*3))
    subpolt_position = 1
    
    if only_prediction:
        for prediction, groundtruth in zip(prediction_list, groundtruth_list):

            plt.subplot(folders,1,subpolt_position)
            prediction.plot()
            groundtruth.plot(color='r')
            subpolt_position=subpolt_position+1

        plt.show()
    else :
        for prediction, groundtruth in zip(prediction_list, groundtruth_list):
            
            train = pd.Series(index=timeseries.index)
            test_date = prediction.index[0]
            train[:test_date] = timeseries[:test_date]
            train[-1] = train[test_date]
            train[test_date] = np.NAN
            # because of using date as index sourcing , the test_data is also inculuded. 
            # it is different from normal range slicing.
            # so here, it has to set test_data as nan

            plt.subplot(folders,1,subpolt_position)
            train.plot()
            prediction.plot()
            groundtruth.plot(color='r')
            subpolt_position=subpolt_position+1

        plt.show()    
		
		
def prediction_visualization(timeseries, start, split, prediction, train_prediction):
    
    ax = timeseries.iloc[start:split].plot(label='observed', figsize=(15, 5))
    train_prediction.plot(ax=ax,label='train prediction')
    prediction.plot(ax=ax, label='Forecast')
    timeseries.iloc[split:].plot(ax=ax,label='forecasting groundtruth' )

    ax.fill_betweenx(ax.get_ylim(), prediction.index[0], prediction.index[-1],
                     alpha=.1, zorder=-1)

    plt.legend()
    plt.show()