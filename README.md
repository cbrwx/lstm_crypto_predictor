Overview
-This tool is an implementation of LSTM (Long Short-Term Memory) network to forecast asset prices. The tool uses historical daily data of 'Close', 'Open', 'High', 'Low', and 'Volume' values from a CSV file to train the model and make predictions. The tool is written in Python and uses libraries such as pandas, numpy, scikit-learn, matplotlib and TensorFlow. The tool is not intended for use as a sole indicator for investment decisions, but may serve as a supplementary indicator within a broader investment strategy.

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
The tool will output a plot of the predicted asset prices for the next 30 days.
Data
- The tool uses daily data of 'Close', 'Open', 'High', 'Low', and 'Volume' values from a CSV file. The data is sorted by the 'Date' column and scaled to be in the range of (0, 1). The data is then split into training, validation, and test sets. The look_back value is set to 30, which means that the model will use the last 30 days of data to make predictions for the next day.

Model
- The model is a sequential LSTM network with four LSTM layers and one dense output layer. The model is optimized using the mean squared error loss function and the Adam optimizer. The model is trained on the training data and evaluated on the validation data. The model is then used to make predictions on the test data.

Plot
- The tool outputs a plot of the predicted asset prices for the next 30 days. The plot shows the actual asset prices in blue and the predicted asset prices in orange.

Disclaimer
- The tool is provided for educational purposes only and is not intended for use as a sole indicator for investment decisions. Use the tool at your own risk. The author of the tool does not assume any responsibility for any investment decisions made based on the use of the tool.
