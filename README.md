LSTM Bitcoin Price Prediction

- This repository contains a script for training an LSTM model to predict 
the future closing prices of Bitcoin using historical data. The model is 
trained on data of Bitcoin prices obtained from Yahoo Finance.

Prerequisites
- Python 3.6 or higher
- Tensorflow 2.0 or higher
- Numpy
- Pandas
- Sklearn
- Matplotlib
- Keras

 Installing
- You can install all prerequisites by running
  !pip install pandas numpy scikit-learn keras matplotlib

Running the script
- To run the script, simply execute the following command in your terminal:
  python lstm_crypto_price_prediction.py (or whatever your filename is)

How the code works
- The script first reads in historical data of Bitcoin prices from a CSV file,
and sorts the data by date. It then selects a subset of the data as the features 
to be used for prediction, and scales the data to be in the range of 0 to 1.

- The script then splits the data into training and test sets, and converts the 
data into a time series dataset. An LSTM model is then created and trained on 
the training data, and predictions are made for the next 24 hours.

- Finally, the script plots the actual and predicted price trends, and displays 
the plot. Im also aware of the mislabeling of graphs, but as with all prediction 
its a trend not a set value, so mislabel it as you wish.

There are several ways to improve this code:
- Data preprocessing: The code uses only the 'Open', 'High', 'Low', 'Close', 'Volume' columns as features to be predicted. However, you may want to consider adding other relevant features, such as technical indicators or macroeconomic data, to improve the model's performance.

- Model architecture: The code uses a single LSTM layer with 50 units. You may want to experiment with different architectures, such as adding more LSTM layers or using a combination of LSTM and other types of layers, to see if it improves the model's performance.

- Hyperparameter tuning: The code uses a batch size of 1 and trains for x epochs. You may want to experiment with different batch sizes and number of epochs to see if it improves the model's performance.

- Evaluation and visualization: The code only plots the predictions and actual prices for the next 24 hours. You may want to evaluate the model's performance using metrics such as mean squared error and R2 score, and plot the predictions for a longer period of time to get a better sense of the model's performance.

- Data augmentation : LSTM is sensitive to the temporal ordering of the data. Try to use different window lookback values to augment your dataset and check if it improves the performance

- Testing with different optimization algorithm : Try testing the model with different optimization algorithm and see if it improves the performance.

- You can use KFold cross-validation in scikit-learn to evaluate the model and determine the best hyperparameters for the LSTM network. Here is an example code that implements KFold cross-validation: 
```
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score

def build_model():
    model = Sequential()
    model.add(LSTM(150, input_shape=(X_train.shape[1], 5)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Wrap the LSTM model in a KerasClassifier object
model = KerasClassifier(build_fn=build_model, epochs=100, batch_size=1, verbose=1)

# Define the number of folds for the KFold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=7)

# Calculate the cross-validation scores using KFold
cv_results = cross_val_score(model, X_train, y_train, cv=kfold)

# Print the mean and standard deviation of the cross-validation scores
print("Accuracy: %.2f%% (%.2f%%)" % (cv_results.mean()*100, cv_results.std()*100))
```

- Do not use this for any commercial ideas, it will surely ruin you!
cbrwx
