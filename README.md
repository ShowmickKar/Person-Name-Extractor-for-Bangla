# Person-Name-Extractor-for-Bangla

This project implements a Bangla-Name-Extractor by finetuning DistilBERT on the Wikiann dataset for named entity recognition(NER). The dataset consists of three splits - train, test and validation, each containing tokenized text data along with correspondingNER tags for named entities.THe model is trained using the Trainer class from the Huggingface Transformer library. Training arguments such as evaluation strategy, learning rate and number of epochs are specified top optimized the model's performance.

## Results

After training for three epochs, the model achieves remarkable performance on the validation set as seen below.
| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 | Accuracy |
|-------|---------------|-----------------|-----------|--------|----------|----------|
| 1 | 0.221500 | 0.145889 | 0.929527 | 0.941283 | 0.935368 | 0.965650 |
| 2 | 0.100500 | 0.117942 | 0.956912 | 0.962963 | 0.959928 | 0.977203 |
| 3 | 0.058200 | 0.101378 | 0.964928 | 0.969286 | 0.967102 | 0.980900 |

## DEMO

```
sentence = "আফজালুর রহমান বলেন, সবার হাতে হাতে প্রশ্ন দেখে তিনি ভেবেছিলেন এটি ভুয়া প্রশ্ন।"
tokens_ner = token_classifier(sentence)

for ner in tokens_ner:
    if ner["entity_group"] == "PER":
        print(ner["word"])
```
```
Output: "আফজালুর রহমান"
```
