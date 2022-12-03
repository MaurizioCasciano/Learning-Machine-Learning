import tensorflow as tf
from keras.layers import Embedding, Bidirectional, LSTM, Dense

vocabulary_size = 10000
batch_size = 64

model = tf.keras.Sequential(
    Embedding(vocabulary_size, batch_size),
    Bidirectional(LSTM(batch_size, return_sequences=True)),
    Bidirectional(LSTM(batch_size / 2)),
    Dense(batch_size, activation="relu"),
    Dense(1, activation="sigmoid")
)
