# TensorFlow RoBERTa Model for Text Emotions Recognition

I am trying to understand how to proceed with a task of emotions recognition from text,
and I have found some interesting resources on this topic.
However, I still have some questions and doubts.

Some premises:
1. The model(s) need(s) to be used with multiple languages (English, Italian, Spanish, ...).
2. The model(s) should return the percentage for each possible emotion (multi label).
3. This is a new field for me, therefore any suggestion is welcome.

The main references that I found for this topic are from:

### Google Research
1. https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html
2. https://github.com/google-research/google-research/tree/master/goemotions
3. https://arxiv.org/pdf/2005.00547.pdf
4. https://github.com/tensorflow/models/blob/master/research/seq_flow_lite/demo/colab/emotion_colab.ipynb

### Arpan Ghoshal
1. https://www.arpanghoshal.com/post/2ee6669b
2. https://huggingface.co/arpanghoshal/EmoRoBERTa

### Transfer Learning
1. https://www.tensorflow.org/tutorials/images/transfer_learning
2. https://keras.io/guides/transfer_learning/
3. [Transfer Learning and Fine-Tuning of a Pretrained Neural Network with Keras and TensorFlow](https://www.youtube.com/watch?v=oHl8U3NccAE&ab_channel=NicolaiNielsen-ComputerVision%26AI)
4. https://huggingface.co/course/chapter3/1?fw=tf
5. https://huggingface.co/course/chapter4/2?fw=tf

### Other
1. [Exploring Transformers in Emotion Recognition: a comparison of
BERT, DistillBERT, RoBERTa, XLNet and ELECTRA](https://arxiv.org/pdf/2104.02041v1.pdf)
2. [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf)
3. [RoBERTa's paper](https://arxiv.org/pdf/1907.11692.pdf)
4. [Facebook-Research - Finetuning RoBERTa on a custom classification task](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.custom_classification.md)
5. [Multilingual text classification with BERT](https://github.com/nlptown/nlp-notebooks/blob/master/Multilingual%20text%20classification%20with%20BERT.ipynb)
6. [bert-base-multilingual-uncased model](https://huggingface.co/bert-base-multilingual-uncased)
7. [bert-base-multilingual-cased model](https://huggingface.co/bert-base-multilingual-cased)
8. [Twitter English Emotions Dataset](https://huggingface.co/datasets/viewer/?dataset=emotion)
9. [Fine-Tuning BERT using TensorFlow](https://medium.com/mlearning-ai/fine-tuning-bert-using-tensorflow-21368d8414ba)

The [EmoRoBERTa model](https://huggingface.co/arpanghoshal/EmoRoBERTa) works very well, but it supports only English.

```python
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

# https://huggingface.co/arpanghoshal/EmoRoBERTa
if __name__ == '__main__':
    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

    emotion_labels = emotion("Thanks for using it.")
    print(emotion_labels)
```

Because I need to perform the emotions recognition task on multiple languages, 
I would like to understand the best approach to proceed for retraining the RoBERTa model 
on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset)
after translating it into other languages (Italian, Spanish, ...).

My doubts are the followings:
1. Should I start with a BERT multilangual model like [bert-base-multilingual-cased model](https://huggingface.co/bert-base-multilingual-cased)? Or would it be better to have a separate model for each language?
2. How to finetune and customize (into RoBERTa) the base model for the "emotion recognition" task?


