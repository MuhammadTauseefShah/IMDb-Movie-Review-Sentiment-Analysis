import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from .config import Config

class DistilBERTSentimentClassifier(nn.Module):
    def __init__(self, config: Config, num_labels=2):
        super().__init__()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


def get_model(config: Config, num_labels=2):
    model = DistilBERTSentimentClassifier(config, num_labels)
    return model


def save_model(model, config: Config, tokenizer=None):
    save_path = config.model_dir / config.model_save_name
    model.model.save_pretrained(save_path)
    if tokenizer:
        tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


def load_trained_model(config: Config):
    model_path = config.model_dir / config.model_save_name
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found at {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model