import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['text.color'] = 'white'

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Read the CSV file, using data from yahoo, so it needs to feature data into a time series dataset
df = pd.read_csv('whatever_coin_you_like.csv', header=0, parse_dates=['Date'])

# Sort the data by the 'Date' column
df = df.sort_values(by='Date')

# Select the 'Open', 'High', 'Low', 'Close', 'Volume' columns as the features to be predicted
features = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Scale the feature data to be in the range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features.values)

# Split the data into training and test sets
train_size = int(len(features) * 0.8)
train_features, test_features = features[0:train_size], features[train_size:]
# Convert the feature data into a time series dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 3])
    return np.array(dataX), np.array(dataY)

X_train, y_train = create_dataset(train_features, 30)
X_test, y_test = create_dataset(test_features, 30)

# Reshape the data for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
# Create an LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 5)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
# Make predictions for the next 1 day
predictions = []
for i in range(24):
    # predict the next hour's closing price
    yhat = model.predict(np.reshape(X_test[i], (1, 30, 5)))[0][0]
    predictions.append(yhat)
    # update the input data for the next prediction
    if i < 23:
        yhat = np.reshape(yhat, (1,1))
        X_test[i+1][-1] = yhat

# Plot the predictions
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(1600/80, 600/80), dpi=80)

# Set background color to black
fig.set_facecolor('black')

# Set line color of actual price curve to signal green
ax1.plot(y_test[:24], label="Actual Price Curve", color='#9aff9a')
ax1.legend(loc="upper left")
ax1.text(0.8, 0.2, 'cbrwx trend prediction', fontsize=10, transform=plt.gcf().transFigure)

# Set line color of predicted price trend to signal green
ax2.plot(predictions, label="Predicted Price trend next 24 hours", color='#9aff9a')
ax2.legend(loc="upper left")
ax2.text(0.8, 0.2, 'cbrwx trend prediction', fontsize=10, transform=plt.gcf().transFigure)

plt.show()

