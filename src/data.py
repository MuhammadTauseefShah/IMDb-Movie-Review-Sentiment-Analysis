import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from sklearn.model_selection import train_test_split
import os

class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def load_imdb_from_csv(csv_path="./data/IMDB Dataset.csv", test_size=0.2, random_state=42):
    """Load IMDb dataset from CSV file."""
    print(f"Loading dataset from {csv_path}...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {list(df.columns)}")
    
    df.columns = df.columns.str.strip()
    
    if 'sentiment' in df.columns:
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    elif 'label' in df.columns:
        if df['label'].dtype == 'object':
            df['label'] = df['label'].map({'positive': 1, 'negative': 0, 'pos': 1, 'neg': 0})
    else:
        raise ValueError("CSV must have 'sentiment' or 'label' column")
    
    texts = df['review'].tolist() if 'review' in df.columns else df['text'].tolist()
    labels = df['label'].tolist()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Positive ratio (train): {sum(train_labels)/len(train_labels):.2%}")
    
    return train_texts, train_labels, test_texts, test_labels


def create_data_loaders(config, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    csv_path = config.data_dir / "IMDB Dataset.csv"
    train_texts, train_labels, test_texts, test_labels = load_imdb_from_csv(csv_path)
    
    if config.train_sample_size:
        train_texts = train_texts[:config.train_sample_size]
        train_labels = train_labels[:config.train_sample_size]
    if config.val_sample_size:
        test_texts = test_texts[:config.val_sample_size]
        test_labels = test_labels[:config.val_sample_size]
    
    print(f"Using {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer, config.max_length)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader, tokenizer