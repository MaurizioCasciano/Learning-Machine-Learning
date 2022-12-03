from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from pandas import DataFrame

if __name__ == '__main__':
    sentences = [
        "Today is a sunny day",
        "Today is a rainy day",
        "Is it sunny today?",
        "I really enjoyed walking in the snow today"
    ]

    for sentence in sentences:
        print(sentence)

    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)

    word_index: dict[str, int] = tokenizer.word_index
    print("\nWord Index:")
    print(word_index)

    sequences: list[list[int]] = pad_sequences(tokenizer.texts_to_sequences(sentences),
                                               padding="post", maxlen=6, truncating="post")

    table: DataFrame = DataFrame(sequences)
    print("\nSequences:")
    print(table.to_string(header=False, index=False))

    test_sentences = [
        "Today is a snowy day",
        "Will it be rainy tomorrow?"
    ]

    test_sequences: list[list[int]] = tokenizer.texts_to_sequences(test_sentences)
    test_table: DataFrame = DataFrame(test_sequences)
    print("\nTest Sequences:")
    print(test_table.to_string(header=False, index=False))


