import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AlbertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding

BASE_MODEL = 'albert-base-v2'
TRAIN_EPOCHS = 10

# Parse our training data
dataset = pd.read_csv('../steamreviews/evaluation_final.CSV', sep=";")
dataset = dataset[['Review Text', 'Helpful Yes/No? (Personal Evaluation)']]
dataset = dataset.rename(columns={'Review Text': 'text', 'Helpful Yes/No? (Personal Evaluation)': 'label'})
#print(dataset.head())


# Split into train (80%), validation (10%) and test sets (10%)
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)
val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42, shuffle=True)


# Load tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Tokenize function
def tokenize_data(data):
    # TO-DO: Test if changing padding/truncation changes the result
    encodings = tokenizer(
        list(data['text']),
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors="pt" # pytorch tensor format
    )
    labels = list(data['label'])
    return encodings, labels

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


# Tokenize Datasets
train_encodings, train_labels = tokenize_data(train_data)
val_encodings, val_labels = tokenize_data(val_data)
test_encodings, test_labels = tokenize_data(test_data)

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Load pre-trained ALBERT model
model = AlbertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)

training_arguments = TrainingArguments(output_dir='./steam_review_model',
                        num_train_epochs=TRAIN_EPOCHS,
                        fp16=True,            # Whether to use 16-bit (mixed) precision training (through NVIDIA apex) 
                        seed=42,              # Random seed for initialization - defaults to 42
                        learning_rate=2e-5,   # The initial learning rate - defaults to 5e-5
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        eval_strategy="epoch",
                        save_strategy = "epoch",
                        load_best_model_at_end=True)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average="macro")
  acc = accuracy_score(labels, preds)
  precision = precision_score(labels, preds, average="macro")
  recall = recall_score(labels, preds, average="macro")
  return {"precision": precision, "recall": recall, "acc": acc, "f1": f1}
     
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

trainer.train()
trainer.save_model()
trainer.evaluate(eval_dataset=test_dataset)
