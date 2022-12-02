import os
from common import Config, TimeSeriesData, split_data, test_data
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import math

config = Config()

# Building the LSTM Model
def LSTM_model():
    model = Sequential()
    model.add(LSTM(5, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(5, return_sequences=False))
    model.add(Dense(3))
    model.add(Dense(1))
    model.compile(optimizer= 'adam', loss = 'mean_squared_error')
    return model
    


# HyperParameters
data_path = config.data_path
epochs = config.epochs
batch_size = config.batch_size
train_size = 0.8
sequence_len = 5

# TimeFrame 
start_time = '01.01.2021 00:00'
end_time = '30.11.2021 23:50'
target_col = 'ws_2'

data = TimeSeriesData(path=data_path, start=start_time , end=end_time)
df = data.series_data(target_col=target_col)

#Pre-process
df = df.fillna(df.median())

df1 = df.values

# Training data
train_data_len = math.ceil(len(df1) * train_size)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df1)

# splitting the data
x_train, y_train = split_data(
    train_data=df1, sequence_len= sequence_len
)

x_test = test_data(
    scaled_data = scaled_data, train_data_len = train_data_len, sequence_len = sequence_len)

y_test = df1[train_data_len:, :]

# Reshaping
x_train = np.reshape( x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# Train the model
model = LSTM_model()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

model_dir = './saved_model'

model.save(os.path.join(model_dir, 'my_model'))