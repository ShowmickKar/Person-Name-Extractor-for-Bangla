import ast
from sklearn.model_selection import train_test_split
import numpy as np
from bnlp.embedding.fasttext import BengaliFasttext
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder


dataset = []
with open('dataset.txt', 'r', encoding='utf-8') as file:
    for line in file:
        data = ast.literal_eval(line)
        text = data[0]
        labels = data[1]
        dataset.append((text, labels))

def tokenize_dataset(dataset):
    tokenized_texts = []
    tokenized_labels = []
    for text, labels in dataset:
        tokens = text.split()
        tokens = [token for token in tokens if token.strip()]
        tokenized_texts.append(tokens)
        tokenized_labels.append(labels)
    return tokenized_texts, tokenized_labels

tokenized_texts, tokenized_labels = tokenize_dataset(dataset)

print("Sample tokenized text:", tokenized_texts[:2])
print("Sample tokenized labels:", tokenized_labels[:2])

train_texts, val_texts, train_labels, val_labels = train_test_split(tokenized_texts, tokenized_labels, test_size=0.1, random_state=42)

bft = BengaliFasttext()

def generate_word_embeddings(tokens, embeddings_model):
    embeddings = []
    for token in tokens:
        vector = embeddings_model.get_word_vector(token)
        embeddings.append(vector)
    return embeddings

train_embeddings = [generate_word_embeddings(tokens, bft) for tokens in train_texts]
val_embeddings = [generate_word_embeddings(tokens, bft) for tokens in val_texts]

def extract_unique_entities(labels_list):
    unique_entities = set()
    for labels in labels_list:
        unique_entities.update(labels)
    return list(unique_entities)

unique_entities = extract_unique_entities(tokenized_labels)
print("Unique entities:", unique_entities)


def find_max_sequence_length(tokenized_texts):
    max_length = max(len(tokens) for tokens in tokenized_texts)
    return max_length

max_sequence_length = find_max_sequence_length(train_texts)
print("Maximum sequence length:", max_sequence_length)

def pad_sequences(embeddings, max_length):
    padded_sequences = []
    for tokens in embeddings:
        tokens = [token for token in tokens if token is not None]
        tokens = [torch.tensor(token, dtype=torch.float32) for token in tokens]
        padded_sequence = tokens + [torch.zeros_like(tokens[0])] * (max_length - len(tokens))
        padded_sequences.append(torch.stack(padded_sequence))  # Convert the list of tensors to a tensor and append to the result
    return padded_sequences

train_padded_sequences = pad_sequences(train_embeddings, max_sequence_length)
val_padded_sequences = pad_sequences(val_embeddings, max_sequence_length)


train_padded_sequences_tensor = torch.stack(train_padded_sequences, dim=0)
val_padded_sequences_tensor = torch.stack(val_padded_sequences, dim=0)

print(f"train_padded_sequence: {train_padded_sequences_tensor}")
print(f"SHAPE: {train_padded_sequences_tensor.shape}")

print("Shape of the first padded sequence (training set):", train_padded_sequences[0].shape)
print("Shape of the first padded sequence (validation set):", val_padded_sequences[0].shape)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional
        
    def forward(self, x):
        out, _ = self.bilstm(x)
        # Reshape out to (batch_size * seq_len, hidden_size*2)
        # out = out.reshape(-1, self.hidden_size * 2)
        out = self.fc(out)
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, class_weights_tensor):
    # print(f"Modes: {model}")
    # print(f"Train Loader: {train_loader}")
    # print(f"Val Loader: {val_loader}")
    # print(f"Criterion: {criterion}")
    # print(f"Optimizer: {optimizer}")
    # print(f"Class Weights Tensor: {class_weights_tensor}")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"OUTPUTS SHAPE: {outputs.shape}")
            # print(f"Outputs: {outputs}")
            # Calculate class weights for the current batch
            class_weights_batch = class_weights_tensor[labels.view(-1)]
            # print(f"CLASS WEIGHTS BATCH: {class_weights_batch}")
            # print(f"CLASS WEIGHTS BATCH SHAPE: {class_weights_batch.shape}")
            # Define the loss function with class weights for the current batch

            criterion_batch = nn.CrossEntropyLoss(weight=class_weights_batch)
            # print(f"CRITERION BATCH: {criterion_batch}")
            # print(f"Labels Shape: {labels.shape}")
            # print(f"LABELS: {labels}")
            labels_flattened = labels.view(-1)
            # loss = criterion_batch(outputs.permute(0, 2, 1), labels_flattened)  # Permute to match the shape
            # loss = criterion_batch(outputs, labels)
            loss = criterion(outputs.view(-1, num_classes), labels_flattened)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.permute(0, 2, 1), labels)  # Permute to match the shape
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')

    return train_losses, val_losses

from sklearn.metrics import classification_report

# Function to evaluate the model on the validation set and compute evaluation metrics

device = torch.device("cpu")

def evaluate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, num_classes), labels.view(-1))
            val_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs, 2)
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    val_loss /= len(val_loader.dataset)

    return val_loss, all_predictions, all_labels


# Convert labels to numeric values
label_mapping = {
    'L-LAW': 0,
    'O': 1,
    'I-PERSON': 2,
    'I-ORG': 3,
    'L-ORG': 4,
    'U-PERSON': 5,
    'U-GPE': 6,
    'B-LAW': 7,
    'U-ORG': 8,
    'B-PERSON': 9,
    'L-PERSON': 10,
    'I-LAW': 11,
    'U-DATE': 12,
    'I-GPE': 13,
    'B-GPE': 14,
    'L-GPE': 15,
    'B-ORG': 16
}

train_labels_numeric = []
for labels in train_labels:
    numeric_labels = [label_mapping[label] for label in labels]
    train_labels_numeric.append(numeric_labels)

val_labels_numeric = []
for labels in val_labels:
    numeric_labels = [label_mapping[label] for label in labels]
    val_labels_numeric.append(numeric_labels)

padding_value = 1
max_label_length = 214

# Pad the labels to have the same length
for i in range(len(train_labels_numeric)):
    train_labels_numeric[i] += [padding_value] * (max_label_length - len(train_labels_numeric[i]))

for i in range(len(val_labels_numeric)):
    val_labels_numeric[i] += [padding_value] * (max_label_length - len(val_labels_numeric[i]))

for i, l in enumerate(train_labels_numeric):
    if len(l) != 214:
        train_labels_numeric[i] = l[:214]

for i, l in enumerate(val_labels_numeric):
    if len(l) != 214:
        train_labels_numeric[i] = l[:214]

train_labels_numeric_tensor = torch.tensor(train_labels_numeric)
val_labels_numeric_tensor = torch.tensor(val_labels_numeric)

print(f"#######################################################")
padded_sequence_lengths = [len(seq) for seq in train_padded_sequences]
# Check the length of train_padded_sequences
print("Number of sequences in train_padded_sequences:", len(train_padded_sequences)) # 3290

# Check the shape of the first sequence in train_padded_sequences
print("Shape of the first sequence in train_padded_sequences:", train_padded_sequences[0].shape) # torch.Size([214, 100])

# Check the length of train_labels_numeric
print("Number of sequences in train_labels_numeric:", len(train_labels_numeric)) # 3190

# Check the shape of the first sequence in train_labels_numeric
print("Shape of the first sequence in train_labels_numeric:", train_labels_numeric_tensor[0].shape) # torch.Size([214, 100])



# Define LSTM Model
 
from sklearn.utils.class_weight import compute_class_weight
# Flatten the train_labels_numeric_tensor
train_labels_flattened = train_labels_numeric_tensor.flatten()

# Convert the flattened tensor to a numpy array
train_labels_numpy = train_labels_flattened.numpy()

# Compute class weights using compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_numpy), y=train_labels_numpy)

# Convert class weights to PyTorch tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


input_size = len(train_embeddings[0][0])
hidden_size = 128
num_layers = 2
num_classes = len(unique_entities) + 1  # Additional class for padding token
model = BiLSTMModel(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



train_data = TensorDataset(torch.stack(train_padded_sequences), torch.tensor(train_labels_numeric))
val_data = TensorDataset(torch.stack(val_padded_sequences), torch.tensor(val_labels_numeric))
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
num_epochs = 10
# Call the train_model function with the computed class weights tensor
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, class_weights_tensor)
val_loss, all_predictions, all_labels = evaluate_model(model, val_loader, criterion)

# Convert numeric predictions and labels to their original entities
numeric_to_label_mapping = {v: k for k, v in label_mapping.items()}

predicted_entities = [numeric_to_label_mapping[prediction] for prediction in all_predictions]
true_entities = [numeric_to_label_mapping[label] for label in all_labels]

# Compute classification report
classification_result = classification_report(true_entities, predicted_entities)
print("Classification Report:")
print(classification_result)
