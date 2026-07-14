import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config import config

st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_pretrained_model():
    """Load pre-trained sentiment model."""
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english"
        )
        return classifier, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_finetuned_model():
    """Load fine-tuned model."""
    model_path = config.model_dir / config.model_save_name
    if not model_path.exists():
        return None, f"Model not found at {model_path}. Please train first."
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return (tokenizer, model), None
    except Exception as e:
        return None, str(e)

def predict_pretrained(text, classifier):
    """Predict using pre-trained pipeline."""
    result = classifier(text)[0]
    prediction = 1 if result["label"].lower() == "positive" else 0
    confidence = result["score"]
    probabilities = [1 - confidence, confidence] if prediction == 1 else [confidence, 1 - confidence]
    return prediction, confidence, probabilities

def predict_finetuned(text, model_data):
    """Predict using fine-tuned model."""
    tokenizer, model = model_data
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
        probabilities_tensor = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities_tensor, dim=1).item()
        confidence = probabilities_tensor[0][prediction].item()
        probabilities = probabilities_tensor[0].tolist()
    
    return prediction, confidence, probabilities

st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below and the model will predict whether it's **positive** or **negative**.")

model_option = st.radio(
    "Select model:",
    ["Pre-trained (no training needed)", "Fine-tuned (requires training)"],
    index=0
)

if model_option == "Pre-trained (no training needed)":
    classifier, error = load_pretrained_model()
    
    if classifier is not None:
        st.success("✅ Pre-trained model loaded successfully!")
        
        text_input = st.text_area(
            "Movie Review",
            height=150,
            placeholder="Type or paste a movie review here..."
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Analyze Sentiment", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        pred, conf, probs = predict_pretrained(text_input, classifier)
                    
                    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
                    color = "green" if pred == 1 else "red"
                    
                    st.markdown(f"### Prediction: :{color}[{sentiment}]")
                    st.markdown(f"**Confidence:** {conf:.2%}")
                    st.progress(conf)
                    
                    st.markdown("### Probability Distribution")
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.metric("Positive", f"{probs[1]:.2%}")
                    with col_neg:
                        st.metric("Negative", f"{probs[0]:.2%}")
                else:
                    st.warning("Please enter a review to analyze.")
        
        with col2:
            st.markdown("### Example Reviews")
            examples = [
                "This movie was absolutely fantastic! The acting was great and the plot kept me engaged throughout.",
                "Terrible film, complete waste of time. The acting was poor and the story made no sense.",
                "It was an okay movie, nothing special but not terrible either. Decent acting."
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"Try example {i+1}", key=f"example_{i}"):
                    st.session_state.example_text = example
                    st.rerun()
            
            if "example_text" in st.session_state:
                st.text_area("Selected example:", st.session_state.example_text, height=100, disabled=True)
    else:
        st.error(f"Failed to load pre-trained model: {error}")
        st.info("Please check your internet connection and try again.")

else:
    model_data, error = load_finetuned_model()
    
    if model_data is not None:
        st.success("✅ Fine-tuned model loaded successfully!")
        
        text_input = st.text_area(
            "Movie Review",
            height=150,
            placeholder="Type or paste a movie review here..."
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Analyze Sentiment", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing..."):
                        pred, conf, probs = predict_finetuned(text_input, model_data)
                    
                    sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
                    color = "green" if pred == 1 else "red"
                    
                    st.markdown(f"### Prediction: :{color}[{sentiment}]")
                    st.markdown(f"**Confidence:** {conf:.2%}")
                    st.progress(conf)
                    
                    st.markdown("### Probability Distribution")
                    col_pos, col_neg = st.columns(2)
                    with col_pos:
                        st.metric("Positive", f"{probs[1]:.2%}")
                    with col_neg:
                        st.metric("Negative", f"{probs[0]:.2%}")
                else:
                    st.warning("Please enter a review to analyze.")
        
        with col2:
            st.markdown("### Example Reviews")
            examples = [
                "This movie was absolutely fantastic! The acting was great and the plot kept me engaged throughout.",
                "Terrible film, complete waste of time. The acting was poor and the story made no sense.",
                "It was an okay movie, nothing special but not terrible either. Decent acting."
            ]
            
            for i, example in enumerate(examples):
                if st.button(f"Try example {i+1}", key=f"example_{i}"):
                    st.session_state.example_text = example
                    st.rerun()
            
            if "example_text" in st.session_state:
                st.text_area("Selected example:", st.session_state.example_text, height=100, disabled=True)
    else:
        st.warning(f"Fine-tuned model not found: {error}")
        st.info("Run 'python src/train.py' to train your own model.")

st.markdown("---")
st.markdown("### Model Information")
st.markdown("- **Framework:** PyTorch + Hugging Face Transformers")
st.markdown("- **Dataset:** IMDb Movie Reviews (50,000 reviews)")
st.markdown("- **Architecture:** DistilBERT transformer")

st.markdown("---")
st.markdown("💡 **Tip:** The pre-trained model works immediately. No training required for portfolio demo!")