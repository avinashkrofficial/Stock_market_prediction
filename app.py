from nselib import capital_market
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Define date range for data extraction
from_date_str = '01-01-2016'
to_date_str = '01-08-2024'

# Streamlit App Title
st.title('Stock Trend Prediction')

# Fetching Nifty 50 data
input_field = st.text_input("Enter your input:")
index_data = capital_market.index_data(index=input_field, from_date=from_date_str, to_date=to_date_str)
index_data['TIMESTAMP'] = pd.to_datetime(index_data['TIMESTAMP'], format='%d-%m-%Y')
index_data = index_data.sort_values('TIMESTAMP').reset_index(drop=True)

st.subheader('Data from 2016 to 2024')
st.write(index_data)

# Visualization: Closing Price vs Time
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(index_data['TIMESTAMP'], index_data['CLOSE_INDEX_VAL'], label='Closing Price')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
plt.legend()
st.pyplot(fig)

# Add Moving Averages (100 EMA and 200 EMA)
moving_avg_100 = index_data['CLOSE_INDEX_VAL'].ewm(span=100, adjust=False).mean()
moving_avg_200 = index_data['CLOSE_INDEX_VAL'].ewm(span=200, adjust=False).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(index_data['TIMESTAMP'], index_data['CLOSE_INDEX_VAL'], label='Closing Price')
plt.plot(index_data['TIMESTAMP'], moving_avg_100, label='100 EMA', color='orange')
plt.plot(index_data['TIMESTAMP'], moving_avg_200, label='200 EMA', color='red')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time with 100 EMA and 200 EMA')
plt.legend()
st.pyplot(fig)

# Prepare data for model
df = index_data

# Splitting data into training and testing sets (70-30 split)
data_training = pd.DataFrame(df['CLOSE_INDEX_VAL'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['CLOSE_INDEX_VAL'][int(len(df) * 0.70):])
past_100_days = data_training.tail(100)

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare test data
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Load pre-trained model
model = load_model('keras_model.h5')

# Predictions
y_predicted = model.predict(x_test)

# Inverse transform to original scale
y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Visualization: Predictions vs Original
test_dates = df['TIMESTAMP'][len(data_training):].reset_index(drop=True)

st.subheader('Prediction v/s Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, 'b', label='Original Price')
plt.plot(test_dates, y_predicted, 'r', label='Predicted Price')
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.title('Prediction vs Original Closing Price Over Time')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)
