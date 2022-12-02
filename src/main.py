from common import Config, TimeSeriesData

config = Config()

# HyperParameters
data_path = config.data_path
epochs = config.epochs
batch_size = config.batch_size

# TimeFrame 
start_time = '01.01.2021 00:00'
end_time = '30.11.2021 23:50'
target_col = 'ws_2'

data = TimeSeriesData(path=data_path, start=start_time , end=end_time)
df = data.series_data(target_col=target_col)

#Pre-process
df = df.fillna(df.median())

# Splitting the data
df1 = df.values


