import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import pickle

print("Ratings data load ho raha hai...")
ratings = pd.read_csv('Ratings.csv')
ratings.columns = ['user_id', 'isbn', 'rating']

print("Data clean ho raha hai...")
ratings = ratings[ratings['rating'] > 0]
ratings = ratings.sample(n=50000, random_state=42)
ratings = ratings.reset_index(drop=True)

print("Users aur Books encode ho rahe hain...")
user_ids = ratings['user_id'].unique().tolist()
book_ids = ratings['isbn'].unique().tolist()

user2idx = {v: i for i, v in enumerate(user_ids)}
book2idx = {v: i for i, v in enumerate(book_ids)}

ratings['user'] = ratings['user_id'].map(user2idx)
ratings['book'] = ratings['isbn'].map(book2idx)

num_users = len(user_ids)
num_books = len(book_ids)

print(f"Total Users: {num_users}")
print(f"Total Books: {num_books}")

X = ratings[['user', 'book']].values
y = ratings['rating'].values / 10.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nNeural Network ban raha hai...")
embedding_size = 32

user_input = keras.Input(shape=(1,), name='user')
book_input = keras.Input(shape=(1,), name='book')

user_embedding = layers.Embedding(
    num_users, embedding_size, name='user_embedding'
)(user_input)
book_embedding = layers.Embedding(
    num_books, embedding_size, name='book_embedding'
)(book_input)

user_vec = layers.Flatten()(user_embedding)
book_vec = layers.Flatten()(book_embedding)

combined = layers.Concatenate()([user_vec, book_vec])
dense1 = layers.Dense(64, activation='relu')(combined)
dense2 = layers.Dense(32, activation='relu')(dense1)
output = layers.Dense(1, activation='sigmoid')(dense2)

model = keras.Model([user_input, book_input], output)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

print("\nModel training ho raha hai...")
print("Thoda time lagega — 3-5 minute wait karo...")

history = model.fit(
    [X_train[:, 0], X_train[:, 1]],
    y_train,
    epochs=5,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

print("\nModel test ho raha hai...")
loss, mae = model.evaluate(
    [X_test[:, 0], X_test[:, 1]],
    y_test,
    verbose=0
)
print(f"Test Loss: {loss:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

print("\nModel save ho raha hai...")
model.save('dl_model.keras')
pickle.dump(user2idx, open('user2idx.pkl', 'wb'))
pickle.dump(book2idx, open('book2idx.pkl', 'wb'))

print("\nDeep Learning Model ready hai!")
print("Neural Network ne ratings se seekh liya!")