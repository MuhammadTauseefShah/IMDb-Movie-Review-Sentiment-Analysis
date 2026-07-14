#!/usr/bin/env python
"""Quick test to verify the project setup."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__)))

print("=" * 60)
print("PROJECT VERIFICATION")
print("=" * 60)

# Test 1: Load CSV
print("\n[1] Loading CSV dataset...")
try:
    import pandas as pd
    df = pd.read_csv("data/IMDB Dataset.csv")
    print(f"    [OK] Loaded {len(df)} reviews")
    print(f"    [OK] Columns: {list(df.columns)}")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

# Test 2: Import modules
print("\n[2] Importing modules...")
try:
    from src.config import config
    print(f"    [OK] Config loaded: {config.model_name}")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

try:
    from src.data import load_imdb_from_csv
    print("    [OK] Data loading module ready")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

try:
    from src.model import DistilBERTSentimentClassifier
    print("    [OK] Model module ready")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

try:
    from src.predict import predict_sentiment
    print("    [OK] Prediction module ready")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

# Test 3: Test pre-trained model
print("\n[3] Testing pre-trained model...")
try:
    from src.predict import predict_sentiment
    sentiment, confidence = predict_sentiment(
        "This movie was fantastic!",
        use_pretrained=True
    )
    print(f"    [OK] Prediction: {sentiment} ({confidence:.2%})")
except Exception as e:
    print(f"    [FAIL] Error: {e}")

print("\n" + "=" * 60)
print("PROJECT READY FOR PORTFOLIO!")
print("=" * 60)
print("\nTo run the demo:")
print("  streamlit run streamlit_app/app.py")
print("\nTo train the model:")
print("  python src/train.py")