from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

if __name__ == '__main__':
    sentences = [
        "I love my dog",
        "I love my cat",
        "You love my dog!",
        "Do you think my dog is amazing?"
    ]

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding="post", truncating="post", maxlen=5)

    print(word_index)
    print(sequences)
    print(padded)

    test_sentences = [
        "I really love my dog",
        "My dog loves my manatee"
    ]

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    print(test_sequences)
