Overview

- This tool is an implementation of LSTM (Long Short-Term Memory) network to forecast asset prices. The tool uses historical daily data of 'Close', 'Open', 'High', 'Low', and 'Volume' values from a CSV file to train the model and make predictions. The tool is written in Python and uses libraries such as pandas, numpy, scikit-learn, matplotlib and TensorFlow. The tool is not intended for use as a sole indicator for investment decisions, but may serve as a supplementary indicator within a broader investment strategy.

Requirements
- pandas
- numpy
- scikit-learn
- matplotlib
- TensorFlow
- Keras

Usage
- Clone the repository to your local machine.
- Install the required libraries.
- Run the code using a Python environment.

Data
- The tool uses daily data of 'Close', 'Open', 'High', 'Low', and 'Volume' values from a CSV file. The data is sorted by the 'Date' column and scaled to be in the range of (0, 1). The data is then split into training, validation, and test sets. The look_back value is set to 30, which means that the model will use the last 30 days of data to make predictions for the next day.

Model
- The model is a sequential LSTM network with four LSTM layers and one dense output layer. The model is optimized using the mean squared error loss function and the Adam optimizer. The model is trained on the training data and evaluated on the validation data. The model is then used to make predictions on the test data.

Plot
- The tool outputs a plot of the predicted asset prices for the next 30 days. The plot shows the actual asset prices in blue and the predicted asset prices in orange.

Detailed
- This script uses the LSTM (Long Short-Term Memory) algorithm in the Keras library to predict asset prices. The asset prices used in this script are Bitcoin prices, which are obtained from a CSV file.

- The script starts by importing the necessary libraries, including pandas, numpy, scikit-learn, matplotlib, tensorflow, and Keras. Then, it sets up the styling of the matplotlib plot by configuring various parameters such as the background color and text color.

- The CSV file containing the asset prices is then read and sorted by the 'Date' column. The 'Close', 'Open', 'High', 'Low', and 'Volume' columns are selected as the features to be predicted. The feature data is then scaled to be in the range (0, 1) using the MinMaxScaler from scikit-learn.

- The feature data is then split into training, validation, and test sets. The feature data is then converted into a time series dataset using the create_dataset function. The look_back value is set to 30, which means that the model will use the previous 30 days of data to predict the next day's closing price.

- The feature data is then reshaped to be used in the LSTM model. The LSTM model is created using the Sequential class from Keras and is compiled using the mean_squared_error loss function and the Adam optimizer. The model is then fit to the training data using the fit method.

- Finally, the model is used to make predictions for the next 25 days.

Disclaimer
- The tool is provided for educational purposes only and is not intended for use as a sole indicator for investment decisions. Use the tool at your own risk. The author of the tool does not assume any responsibility for any investment decisions made based on the use of the tool.

.cbrwx
