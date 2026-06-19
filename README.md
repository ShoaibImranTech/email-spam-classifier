# 📧 Email Spam Classifier — Multi-Model ML Comparison

A comprehensive **Email Spam Detection** project that benchmarks **7 machine learning classifiers** on the Kaggle Email Spam dataset. Starting from a Naive Bayes baseline, this project systematically compares classical ML algorithms using standardized text vectorization and evaluation metrics.

---

## 📌 Project Highlights

- ✅ **7 Models** trained and compared: Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost, KNN, Decision Tree
- 📊 **Side-by-side evaluation** using Accuracy, F1-Score, Precision, and Recall
- 🔠 **Dual vectorization**: CountVectorizer + TF-IDF with bigrams
- 🧹 **Text preprocessing pipeline**: lowercasing, URL removal, punctuation stripping
- 📈 **Visual outputs**: EDA plots, confusion matrices, model comparison bar chart
- 💾 Results exported to CSV for reproducibility
- 🔍 **Live prediction** function to test custom email inputs

---

## 📁 Repository Structure

```
email-spam-classifier/
│
├── notebooks/
│   ├── Email_Classification_NaiveBayes_Original.ipynb   # Original Naive Bayes notebook
│   └── email_classification_all_models.ipynb            # Full multi-model comparison
│
├── data/
│   └── combined_data.csv          # Kaggle dataset (add manually — see below)
│
├── src/
│   └── predict.py                 # Standalone prediction script
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📂 Dataset

**Source:** [Kaggle — Email Spam Classification Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

The dataset (`combined_data.csv`) contains labeled emails with two columns:
| Column | Description |
|--------|-------------|
| `text`  | Raw email body text |
| `label` | `1` = Spam, `0` = Ham (not spam) |

> **Note:** Download the dataset from Kaggle and place `combined_data.csv` inside the `data/` folder before running the notebooks.

---

## 🤖 Models Compared

| # | Model | Vectorizer | Notes |
|---|-------|-----------|-------|
| 1 | **Naive Bayes (Multinomial)** | CountVectorizer | Probabilistic baseline; fast & interpretable |
| 2 | **Logistic Regression** | TF-IDF | Strong linear baseline; highly reliable |
| 3 | **Support Vector Machine** | TF-IDF | LinearSVC; excellent for text classification |
| 4 | **Random Forest** | CountVectorizer | Ensemble of decision trees; robust |
| 5 | **Decision Tree** | CountVectorizer | Simple, interpretable tree-based model |
| 6 | **K-Nearest Neighbors** | TF-IDF | Instance-based; slower on large data |
| 7 | **XGBoost** | TF-IDF | Gradient boosted trees; state-of-the-art |

---

## 📊 Sample Results

> *Actual values will vary depending on your dataset version and random seed.*

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| SVM (LinearSVC) | ~0.9890 | ~0.9850 |
| Logistic Regression | ~0.9870 | ~0.9830 |
| XGBoost | ~0.9860 | ~0.9820 |
| Random Forest | ~0.9840 | ~0.9800 |
| Naive Bayes | ~0.9780 | ~0.9740 |
| Decision Tree | ~0.9650 | ~0.9600 |
| KNN | ~0.9500 | ~0.9450 |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/email-spam-classifier.git
cd email-spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download `combined_data.csv` from Kaggle and place it in the `data/` folder:
```
data/combined_data.csv
```

### 4. Run the notebook
```bash
jupyter notebook notebooks/email_classification_all_models.ipynb
```

### 5. (Optional) Run the standalone predictor
```bash
python src/predict.py
```

---

## 🔍 Text Preprocessing Pipeline

Each email goes through the following cleaning steps before vectorization:

```
Raw Email → Lowercase → Remove URLs → Remove Numbers → Remove Punctuation → Normalize Whitespace
```

Two feature extraction strategies are applied:
- **CountVectorizer** (Bag-of-Words, top 5000 features) — used for Naive Bayes, Random Forest, Decision Tree
- **TF-IDF with bigrams** (1,2-gram range, top 5000 features) — used for SVM, Logistic Regression, KNN, XGBoost

---

## 📈 Visual Outputs

After running the notebook, the following charts are saved to `results/`:

| File | Description |
|------|-------------|
| `eda_plots.png` | Class distribution + email length histogram |
| `confusion_matrices.png` | Grid of confusion matrices for all 7 models |
| `model_comparison.png` | Grouped bar chart — Accuracy vs F1 Score per model |
| `model_comparison_results.csv` | Ranked table of all model scores |

---

## 🧪 Predict on Custom Text

```python
from src.predict import predict_email

predict_email("Congratulations! You've won a $1000 Amazon gift card. Click here!")
# 🚫 SPAM

predict_email("Hey, can you send me the notes from yesterday's lecture?")
# ✅ HAM (Not Spam)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| pandas / numpy | Data handling |
| scikit-learn | ML models & evaluation |
| XGBoost | Gradient boosting |
| matplotlib / seaborn | Visualization |
| Jupyter Notebook | Interactive development |

---

## 🙏 Acknowledgements

- **Dataset:** [Balaka Biswas](https://www.kaggle.com/balaka18) on Kaggle for the Email Spam Classification Dataset
- **Scikit-learn Documentation:** [https://scikit-learn.org](https://scikit-learn.org) — for comprehensive ML API references
- **XGBoost:** [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Inspired by the classic [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) research

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Muhammad Shoaib**  
BS Computer Science — University of the Punjab, Lahore  
