# Beware troglodytes! Utilizing this tool for the sole purpose 
# of forecasting asset prices and subsequently participating 
# in speculative ventures is strongly discouraged. At 
# best, this may serve as a supplementary indicator within 
# your broader investment strategy.
# -cbrwx - https://github.com/cbrwx

# ---------------------------------

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Configuring matplotlib styling
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['text.color'] = 'white'

# Read the CSV file, using data from yahoo, so it needs to feature data into a time series dataset
df = pd.read_csv('http://lillegaardtannklinikk.no/1/BTC-USD.csv', header=0, parse_dates=['Date'])

# Sort the data by the 'Date' column
df = df.sort_values(by='Date')

# Select the 'Close', 'Open', 'High', 'Low', and 'Volume' columns as the features to be predicted
feature = df[['Close', 'Open', 'High', 'Low', 'Volume']]

# Scale the feature data to be in the range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
feature = scaler.fit_transform(feature.values)

# Split the data into training, validation and test sets
train_size = int(len(feature) * 0.7)
val_size = int(len(feature) * 0.15)
train_feature, val_feature, test_feature = feature[0:train_size], feature[train_size:train_size+val_size], feature[train_size+val_size:]

# Convert the feature data into a time series dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# Define the look_back value and convert the feature data into time series dataset
look_back = 30
X_train, y_train = create_dataset(train_feature, 30)
X_val, y_val = create_dataset(val_feature, 30)
X_test, y_test = create_dataset(test_feature, 30)

# Reshape the data for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

# Create an LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 5)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

# Fit the model to the training data (edit for optimization of your model)
model.fit(X_train, y_train, epochs=55, batch_size=1, verbose=3)

# Make predictions for the next 30 days
predictions = []
for i in range(25):
    # predict the next day's closing price
    yhat = model.predict(X_test[i].reshape(1, -1, 5))[0]
    predictions.append(yhat)
    # update the input data for the next prediction
    if i < 29:
        yhat = np.reshape(yhat, (1, 1, yhat.shape[-1]))
        if i < X_test.shape[0]-1:
            X_test[i+1, :-1, :] = X_test[i, 1:, :]
            X_test[i+1, -1, :] = yhat

# Plot the predictions
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(2000/80, 400/80), dpi=80)

# Set background color to black
fig.set_facecolor('black')

# Set line color of actual price curve to signal green
ax1.plot(y_test[:24], label="Price factor", color='#9aff9a')
ax1.legend(loc="upper left")
ax1.text(0.8, 0.2, 'cbrwx trend prediction', fontsize=10, transform=plt.gcf().transFigure)

# Set line color of predicted price trend to signal green
ax2.plot(predictions, label="Predicted Price trend next 25 days", color='#9aff9a', linestyle='dashed')
ax2.legend(loc="upper left")
ax2.text(0.8, 0.2, 'cbrwx trend prediction', fontsize=10, transform=plt.gcf().transFigure)
plt.show()
