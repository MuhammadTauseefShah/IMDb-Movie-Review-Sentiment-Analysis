import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import config
from src.data import load_imdb_from_csv, IMDbDataset
from src.model import DistilBERTSentimentClassifier, save_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import os

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    losses = []
    correct = 0
    total = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs.loss
        logits = outputs.logits
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        losses.append(loss.item())
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return np.mean(losses), correct / total


def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    losses = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss
            logits = outputs.logits
            
            losses.append(loss.item())
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actual_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_labels, predictions, average="binary"
    )
    
    return {
        "loss": np.mean(losses),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("Loading IMDb dataset...")
    csv_path = config.data_dir / "IMDB Dataset.csv"
    train_texts, train_labels, test_texts, test_labels = load_imdb_from_csv(csv_path)
    
    if config.train_sample_size:
        train_texts = train_texts[:config.train_sample_size]
        train_labels = train_labels[:config.train_sample_size]
    if config.val_sample_size:
        test_texts = test_texts[:config.val_sample_size]
        test_labels = test_labels[:config.val_sample_size]
    
    print(f"Training on {len(train_texts)} samples")
    print(f"Validating on {len(test_texts)} samples")
    
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = IMDbDataset(test_texts, test_labels, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    print("Initializing model...")
    model = DistilBERTSentimentClassifier(config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        
        val_metrics = eval_model(model, val_loader, device)
        print(f"Val loss: {val_metrics['loss']:.4f}, Val acc: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            save_model(model, config, tokenizer)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    print(f"\nTraining complete! Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()