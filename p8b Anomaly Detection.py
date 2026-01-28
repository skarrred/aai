#p8b
# B)Building a deep learning model for anomaly detection 

# pip install tensorflow pandas scikit-learn matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv("/content/creditcard.csv")
print(df.head())
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Prepare features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separate normal and fraud data
X_normal = X_scaled[y == 0]
X_fraud = X_scaled[y == 1]

# Split normal data into train and test sets
X_train, X_test = train_test_split(X_normal, test_size=0.2, random_state=42)
print("\nTraining samples:", X_train.shape)
print("Test samples:", X_test.shape)

# Build Autoencoder model
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))

encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
bottleneck = Dense(8, activation="relu")(encoder)

decoder = Dense(16, activation="relu")(bottleneck)
decoder = Dense(32, activation="relu")(decoder)
output_layer = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=256,
    validation_split=0.1,
    shuffle=True
)

# Predict reconstruction on test set
reconstructions = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructions), axis=1)

# Set threshold as 95th percentile of reconstruction error on normal test data
threshold = np.percentile(reconstruction_error, 95)
print("\nAnomaly Threshold:", threshold)

# Evaluate on combined test data (normal + fraud)
X_all = np.vstack([X_test, X_fraud])
y_all = np.hstack([np.zeros(len(X_test)), np.ones(len(X_fraud))])

reconstructions_all = autoencoder.predict(X_all)
errors_all = np.mean(np.square(X_all - reconstructions_all), axis=1)

y_pred = (errors_all > threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_all, y_pred))

print("\nClassification Report:")
print(classification_report(y_all, y_pred))

# Plot reconstruction errors
plt.figure(figsize=(8, 4))
plt.hist(errors_all[y_all == 0], bins=50, alpha=0.6, label="Normal")
plt.hist(errors_all[y_all == 1], bins=50, alpha=0.6, label="Fraud")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Anomaly Detection using Autoencoder")
plt.legend()
plt.show()
