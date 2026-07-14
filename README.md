# Portfolio — Deep Learning Showcase

Welcome to my machine learning portfolio. This repository is a collection of end-to-end deep learning projects built to demonstrate practical NLP, modeling, and deployment skills. Each project is self-contained with reproducible code, documentation, and a live demo where applicable.

## Featured Project: IMDb Sentiment Analysis (DistilBERT)

A complete sentiment classification system that fine-tunes a **DistilBERT** transformer on 50,000 IMDb movie reviews to predict whether a review is **positive** or **negative**.

| | |
|---|---|
| **Task** | Binary text classification (sentiment) |
| **Model** | DistilBERT (transformer, ~66M params) |
| **Framework** | PyTorch + Hugging Face Transformers |
| **Dataset** | IMDb Movie Reviews (50K labeled reviews) |
| **Interface** | Interactive Streamlit web app |
| **Hardware** | CPU-compatible |

### Highlights
- End-to-end pipeline: data loading → tokenization → training → evaluation → deployment
- Pre-trained model support — works instantly with no training required
- Modular, production-style code (`src/` package, config-driven)
- Exploratory data analysis + training notebooks included
- Clean UI for real-time sentiment prediction with confidence scores

### Quick Start
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```
Then open http://localhost:8501 and select **"Pre-trained (no training needed)"**.

To fine-tune on your own data:
```bash
python src/train.py
```

### Project Structure
```
portfolio/
├── data/IMDB Dataset.csv        # Dataset (review, sentiment)
├── src/                         # Source code (config, data, model, train, predict)
├── notebooks/                   # EDA + training walkthrough
├── streamlit_app/app.py         # Interactive web demo
└── README.md                    # Project documentation
```

## Skills Demonstrated
- Transformer fine-tuning (BERT/DistilBERT)
- Text tokenization & sequence modeling
- PyTorch training loops, schedulers, evaluation metrics
- Data engineering with pandas
- Model packaging & serving via Streamlit
- Reproducible, documented ML project structure

## Repository Conventions
- `requirements.txt` pins dependencies
- `.gitignore` excludes caches, large model binaries, and local artifacts
- Notebooks are kept for exploration; `src/` holds reusable modules

---
Built as part of my ML/DL learning journey and portfolio. Feedback and collaboration welcome!
