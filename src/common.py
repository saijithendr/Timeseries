import os
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesData:
    
    def __init__(self, path: str, start: str= None, end: str=None) -> pd.DataFrame:
        self.start = start
        self.end = end
        self.path = path
        
        try:
            df = pd.read_csv(self.path, sep=',', parse_dates=True)
        except:
            print('Invalid Path !!')
    
        df.set_index('Zeitstempel (UTC)', inplace=True)
        self.df = df
            
    def __len__(self):
        return self.df.shape
        
    
    def __call__(self):
        return self.df
    
    def series_data(self, target_col: str):
                
        if self.start and self.end is not None:
            df = self.df.loc[
                self.start : self.end,
                           [target_col]
            ]
            return df
        else:
            return self.df
        
class Config:
    
    def __init__(self) -> None:
        self.data_path = os.path.join('data', 'window', 'turbine_data.csv')
        self.batch_size = 32
        self.epochs = 1

def split_data(train_data, sequence_len):
    
    '''splits the train data into x_train and y_train '''
    
    # Splitting the data into x_train, y_train 
    x_train = []
    y_train = []

    for i in range(sequence_len, len(train_data)):
        x_train.append(train_data[i-sequence_len:i , 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    return x_train, y_train


def test_data( scaled_data, train_data_len, sequence_len ):
    
    test_data = scaled_data[train_data_len-sequence_len: , : ]
    
    # create x_test
    x_test = []

    for i in range(sequence_len , len(test_data)):
        
        x_test.append(test_data[i-sequence_len:i, 0])
        
    x_test = np.array(x_test)
    
    return x_test