from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

# https://huggingface.co/arpanghoshal/EmoRoBERTa
if __name__ == '__main__':
    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

    emotion_labels = emotion("Thanks for using it.")
    print(emotion_labels)
