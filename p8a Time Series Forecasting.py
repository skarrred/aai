#p8a
# A)Building a deep learning model for time series forecasting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Step 3: Load and Preprocess Data
data = pd.read_csv('/content/synthetic_time_series.csv')  # Replace with your actual file path
data = data['value'].values  # Replace 'value' with your actual column name

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]

# Step 4: Create Dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Prepare datasets
time_step = 10
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 5: Build the LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 6: Train the Model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Step 7: Make Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE
train_score = np.sqrt(np.mean((train_predict - scaler.inverse_transform(y_train.reshape(-1,1)))**2))
test_score = np.sqrt(np.mean((test_predict - scaler.inverse_transform(y_test.reshape(-1,1)))**2))

print(f'Train Score: {train_score:.2f} RMSE')
print(f'Test Score: {test_score:.2f} RMSE')

# Step 8: Visualize the Results
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(data), label='Original Data')
plt.plot(np.arange(time_step, time_step + len(train_predict)), train_predict, label='Train Predictions')
plt.plot(np.arange(len(train_predict) + (time_step * 2), len(train_predict) + (time_step * 2) + len(test_predict)), test_predict, label='Test Predictions')
plt.legend()
plt.show()
