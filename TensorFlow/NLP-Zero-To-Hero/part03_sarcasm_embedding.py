import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

if __name__ == '__main__':

    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    with open("../../data/Sarcasm/sarcasm.json", "r") as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    urls = []

    for item in datastore:
        sentences.append(item["headline"])
        labels.append(item["is_sarcastic"])
        urls.append(item["article_link"])

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]

    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    print(word_index)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    print(training_padded[0])
    print(training_padded.shape)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Need this block to get it to work with TensorFlow 2.x
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    num_epochs = 30
    history = model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels), verbose=2)

    sentence = ["granny starting to fear spiders in the garden might be real",
                "the weather today is bright and sunny"]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(model.predict(padded))
