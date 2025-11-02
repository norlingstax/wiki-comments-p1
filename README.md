# Toxic Comment Classification (Part 1)

Detecting toxic comments from Wikipedia discussions using linear machine learning models. [**Part 2**](https://github.com/norlingstax/wiki-comments-p2) of this project applies modern ML tools (embeddings, transformers, LLMs) to the same [data](https://www.kaggle.com/datasets/hetvigandhi03/imported-data).

---

## Setup

### Requirements
- Python ≥ 3.9
- `pip`, `venv`

### Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
````

---

## How to Run

### Full pipeline

Runs preprocessing -> training -> evaluation -> interpretation

```bash
bash run.sh     # Linux/macOS
run.bat         # Windows
```

### Step-by-step

```bash
# Preprocess data
python -m src.preprocess

# Train models
python -m src.train --model logreg
python -m src.train --model lsvc
python -m src.train --model pac

# Evaluate models
python -m src.evaluate --model baseline
python -m src.evaluate --model logreg
python -m src.evaluate --model lsvc
python -m src.evaluate --model pac

# Interpret model predictions (FP/FN samples)
python -m src.interpret --model logreg
python -m src.interpret --model lsvc
python -m src.interpret --model pac
```

---

## Pipeline Overview

1. **Preprocessing**

   * Normalise text, add metadata columns
   * Lemmatisation + TF-IDF vectorisation (results in \~1M features)
   * Feature selection with `SelectKBest` -> 150k features
   * Cached datasets saved in `processed/`

2. **Baseline Model**

   * Dictionary-based model using an [external toxic lexicon](https://github.com/Orthrus-Lexicon/Toxic).


3. **Model Training**

   * Logistic Regression (hyperparameter tuning with `BayesSearch`)
   * Linear SVM (default hyperparameters)
   * Passive Aggressive Classifier (hyperparameter tuning with `BayesSearch`)
   * Models and configs stored in `models/<model_name>`

4. **Evaluation**

   * Metrics: precision, recall, F1, ROC-AUC, PR-AUC
   * Plots: ROC curves, PR curves, confusion matrices
   * Saved in `outputs/metrics/` and `outputs/figures/`

5. **Interpretation**

   * Sampled 3 false positives + 3 false negatives per model
   * Stored in `outputs/errors/`

---

## Repository Outputs

```
outputs/
└── <model_name>/
    ├── metrics/      # val/test results (CSV/JSON), ROC-AUC/PR-AUC scores
    ├── figures/      # ROC/PR curves, confusion matrix
    └── errors/       # FP/FN samples
data/processed/       # Cached preprocessed data, vectoriser, feature selector
models/
└── <model_name>/     # trained model + BayesSearchCV artifacts (if any)
```

---

## Notes

* Focus of this repo is learning to create a **clean, reproducible ML pipeline** rather than model optimisation.
* All stages can be reproduced with provided `.sh` / `.bat` runners.
