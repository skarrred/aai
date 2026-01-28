#p4
# Developing a recommendation system using collaborative filtering or deep learning approaches.
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the MovieLens 100k dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(url, sep='\t', names=columns)

# Step 2: Encode user and item IDs
user_enc = LabelEncoder()
item_enc = LabelEncoder()

df['user'] = user_enc.fit_transform(df['user_id'])
df['item'] = item_enc.fit_transform(df['item_id'])

num_users = df['user'].nunique()
num_items = df['item'].nunique()

# Normalize ratings to 0-1 range (optional for regression)
df['rating'] = df['rating'] / 5.0

# Step 3: Split into training and test sets
X = df[['user', 'item']].values
y = df['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Neural Collaborative Filtering model
embedding_size = 50

user_input = tf.keras.Input(shape=(1,), name='user_input')
item_input = tf.keras.Input(shape=(1,), name='item_input')

user_embedding = tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_size)(user_input)
item_embedding = tf.keras.layers.Embedding(input_dim=num_items, output_dim=embedding_size)(item_input)

user_vec = tf.keras.layers.Flatten()(user_embedding)
item_vec = tf.keras.layers.Flatten()(item_embedding)

concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
dense = tf.keras.layers.Dense(128, activation='relu')(concat)
dense = tf.keras.layers.Dense(64, activation='relu')(dense)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dense)  # sigmoid because ratings are scaled to [0,1]

model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()

# Step 5: Train the model
history = model.fit(
    x=[X_train[:, 0], X_train[:, 1]],
    y=y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=2
)

# Step 6: Evaluate the model
loss, mae = model.evaluate([X_test[:, 0], X_test[:, 1]], y_test, verbose=2)
print(f"\nTest MAE: {mae:.4f}")

# Optional: Predict some ratings
sample_users = X_test[:5, 0]
sample_items = X_test[:5, 1]
predicted_ratings = model.predict([sample_users, sample_items])
print("\nSample Predictions:")
for i in range(5):
    print(f"User {sample_users[i]} - Item {sample_items[i]} → Predicted Rating: {predicted_ratings[i][0]*5:.2f}")

