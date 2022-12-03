from keras.preprocessing.text import Tokenizer

if __name__ == '__main__':
    sentences = [
        "I love my dog",
        "I love my cat",
        "You love my dog!"
    ]

    tokenizer = Tokenizer(num_words=10)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(word_index)
