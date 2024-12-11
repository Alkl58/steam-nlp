import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AlbertForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

# Load dataset
dataset = pd.read_csv('steam_reviews_constructiveness_1.5k.csv')
dataset = dataset[['review', 'constructive']]

#print(dataset)

# Split into train and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

# Tokenize function
def tokenize_data(data):
    return tokenizer(
        list(data['review']),
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

# Tokenize datasets
train_tokens = tokenize_data(train_data)
val_tokens = tokenize_data(val_data)

class SteamReviewDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = torch.tensor(labels.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.tokens.items()}, self.labels[idx]

# Create datasets
train_dataset = SteamReviewDataset(train_tokens, train_data['constructive'])
val_dataset = SteamReviewDataset(val_tokens, val_data['constructive'])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained ALBERT model
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)

# Optimizer and loss
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader)}")

# Save model
model.save_pretrained('./steam_review_model')
tokenizer.save_pretrained('./steam_review_model')

