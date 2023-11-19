# LSTM-Stock-Prediction-Model-Web-App

A LSTM Stock Prediction Model on Google Stock Prices &amp; A Streamlit Web App For Any Given Stock

## Setting Up Dependencies

The code utilizes the following libraries for data manipulation, visualization, and machine learning:

- Pandas
- Pandas DataReader
- NumPy
- Matplotlib
- yfinance
- Scikit-Learn's MinMaxScaler
- TensorFlow and Keras for building and training the LSTM model

## Data Collection and Preprocessing

### Set Datetime From 2012 Until Yesterday

The code starts by setting up the datetime range from 2012 to yesterday using the `yfinance` library to download stock data.

### Time Series Sequence Generation

The data is split into training and test sets, and the training data is then scaled using Min-Max scaling. Sequences of length 100 are generated from the scaled training data to create input-output pairs for the LSTM model.

## LSTM Model Construction and Training

The LSTM model is constructed with multiple layers, including LSTM layers with varying numbers of units and dropout layers for regularization. The model is compiled using the Adam optimizer and mean squared error loss function. It is then trained on the generated input-output pairs for 50 epochs.

## Testing the Model on Test Data

The last 100 days of training data are extracted and combined with the test data. The combined test data is scaled, and input-output pairs are generated for testing. The model predicts the stock prices for the test data, and the predictions are inverse-scaled to obtain the actual predicted prices. Finally, the predicted and original prices are plotted for visual assessment.

## Saving the Model

The trained LSTM model is saved for future use in the Streamlit web application.

---

# The App

## Setting Up Dependencies

The file app.py utilizes the following libraries for data manipulation, visualization, and creating the Streamlit web application:

- NumPy
- Datetime
- Pandas
- Matplotlib
- yfinance
- Scikit-Learn's MinMaxScaler
- Keras for loading the saved LSTM model
- Streamlit for building the web application

## Loading the Pretrained Model

The saved LSTM model is loaded using Keras's `load_model` function.

## Streamlit Web Application

The Streamlit web application allows the user to input a stock symbol, and it retrieves stock data from Yahoo Finance. Moving averages and predicted stock prices are visualized in the web application using Matplotlib and Streamlit.

