# lstm_crypto_predictor_
This repository contains a script for training an LSTM model to predict the future closing prices of Crypto using historical data.

LSTM Bitcoin Price Prediction_
This repository contains a script for training an LSTM model to predict the future closing prices of Bitcoin using historical data.

Prerequisites_

Python 3
pandas
numpy
scikit-learn
keras
matplotlib

Installing_

You can install all prerequisites by running
  !pip install pandas numpy scikit-learn keras matplotlib

Running the script_

To run the script, simply execute the following command in your terminal:
  python lstm_crypto_price_prediction.py (or whatever your filename is)

How the code works_

The script first reads in historical data of Bitcoin prices from a CSV file, and sorts the data by date. It then selects a subset of the data as the features to be used for prediction, and scales the data to be in the range of 0 to 1.

The script then splits the data into training and test sets, and converts the data into a time series dataset. An LSTM model is then created and trained on the training data, and predictions are made for the next 24 hours.

Finally, the script plots the actual and predicted price trends, and displays the plot. Im also aware of the mislabeling of graphs, but as with all prediction 
its a trend not a set value, so mislabel it as you wish.

     /\_/\     
    ( o.o )    
     > ^ <
