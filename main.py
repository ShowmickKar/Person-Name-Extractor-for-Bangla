import ast
from sklearn.model_selection import train_test_split
import numpy as np
from bnlp.embedding.fasttext import BengaliFasttext
import fasttext
fasttext.FastText.eprint = lambda x: None


# Read the file and store the dataset
dataset = []
with open('dataset.txt', 'r', encoding='utf-8') as file:
    for line in file:
        data = ast.literal_eval(line)
        text = data[0]
        labels = data[1]
        dataset.append((text, labels))

# Function to tokenize text and labels
def tokenize_dataset(dataset):
    tokenized_texts = []
    tokenized_labels = []
    for text, labels in dataset:
        # Tokenize text
        tokens = text.split()
        tokenized_texts.append(tokens)
        # Tokenize labels
        tokenized_labels.append(labels)
    return tokenized_texts, tokenized_labels

# Tokenize the dataset
tokenized_texts, tokenized_labels = tokenize_dataset(dataset)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(tokenized_texts, tokenized_labels, test_size=0.1, random_state=42)

# Load Bengali FastText word embeddings
bft = BengaliFasttext()

# Function to generate word embeddings for tokens
def generate_word_embeddings(tokens, embeddings_model):
    embeddings = []
    for token in tokens:
        vector = embeddings_model.get_word_vector(token)
        embeddings.append(vector)
    return embeddings

# Generate word embeddings for training and validation data
train_embeddings = [generate_word_embeddings(tokens, bft) for tokens in train_texts]
val_embeddings = [generate_word_embeddings(tokens, bft) for tokens in val_texts]
