import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import config
import numpy as np

def predict_sentiment(text, model_path=None, use_pretrained=True):
    """
    Predict sentiment of a text.
    
    Args:
        text: Input text
        model_path: Path to fine-tuned model (optional)
        use_pretrained: If True, use pre-trained model from Hugging Face
    """
    if use_pretrained and model_path is None:
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        result = classifier(text)[0]
        sentiment = result["label"].lower()
        confidence = result["score"]
        return sentiment, confidence
    
    if model_path is None:
        model_path = config.model_dir / config.model_save_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment, confidence


def predict_batch(texts, model_path=None):
    results = []
    for text in texts:
        sentiment, confidence = predict_sentiment(text, model_path)
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": sentiment,
            "confidence": f"{confidence:.2%}"
        })
    return results


if __name__ == "__main__":
    test_reviews = [
        "This movie was absolutely fantastic! The acting was great and the plot kept me engaged.",
        "Terrible film, waste of time. Poor acting.",
        "It was okay, nothing special but not bad."
    ]
    
    print("Testing with pre-trained model...")
    print("=" * 80)
    
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review)
        print(f"\nReview: {review[:60]}...")
        print(f"Sentiment: {sentiment.upper()} (confidence: {confidence:.2%})")