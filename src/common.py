import os
import pandas as pd
import numpy as np

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
        self.epochs = 10
