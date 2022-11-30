import os
import pandas as pd
import numpy as np

class TimeSeriesData:
    
    def __init__(self, path: str, start: str= None, end: str=None) -> pd.DataFrame:
        self.start = start
        self.end = end
        self.path = path
        
        try:
            df = pd.read_csv(self.path, sep=',')
        except:
            print('Invalid Path !!')
        
        df['Zeitstempel (UTC)'] = pd.to_datetime(df['Zeitstempel (UTC)'])
        df.set_index('Zeitstempel (UTC)', inplace=True)
        self.df = df
            
    def __len__(self):
        return self.df.shape
        
    
    def __call__(self):
        return self.df
    
    def series_data(self, target_col: str, start, end):
                
        if self.start and self.end is not None:
            df = self.df.loc[
                self.start : self.end,
                           [target_col]
            ]
            return df
        else:
            return self.df