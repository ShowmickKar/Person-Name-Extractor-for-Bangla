# PERSON NAME EXTRACTOR FOR BANGLA

We're building a Named Entity Recognition (NER) model to extract names from Bangla text. The dataset we're using is the Wikiann dataset, which contains labeled data for various languages, including Bangla.

Here's a brief overview of the steps involved:

- **Dataset**: We use Wikiann's Bangla subset, which includes training, validation, and test sets with tokenized text and named entity tags.
- **Model**: Our approach involves fine-tuning a pre-trained Bangla BERT model to recognize named entities in the Bangla language.
- **Training and Evaluation**: We train the model on the Wikiann training set and evaluate its performance using precision, recall, and F1-score on the validation set.
- **Deployment**: After training, we save the fine-tuned model and test it on unseen Bangla text to ensure accurate extraction of names and other entities.

## Results

After training for three epochs, the model achieves remarkable performance on the validation set as seen below.
| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 | Accuracy |
|-------|---------------|-----------------|-----------|--------|----------|----------|
| 1 | 0.221500 | 0.145889 | 0.929527 | 0.941283 | 0.935368 | 0.965650 |
| 2 | 0.100500 | 0.117942 | 0.956912 | 0.962963 | 0.959928 | 0.977203 |
| 3 | 0.058200 | 0.101378 | 0.964928 | 0.969286 | 0.967102 | 0.980900 |

## DEMO

```
sentence = "আফজালুর রহমান বলেন, সবার হাতে হাতে প্রশ্ন দেখে তিনি ভেবেছিলেন এটি ভুয়া প্রশ্ন। উত্তম কুমার ভট্টাচার্য্য এ কথার সাথে দ্বিমত পোশষণ করেন।"

tokens_ner = token_classifier(sentence)

for ner in tokens_ner:
  if ner["entity_group"] == "PER":
    print(ner["word"])
```

```
Output:
আফজালুর রহমান
উত্তম কুমার ভট্টাচার্য্য
```
