# Credit Card Fraud Detection

Detect fraudulent credit card transactions using machine learning on a heavily imbalanced dataset. Covers EDA, preprocessing, four class-balancing techniques, and four algorithms — with business-framed model selection.

---

## Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the root directory.

- 284,807 transactions · 492 fraudulent (0.172%)
- Features `V1`–`V28` are PCA-transformed · only `Amount` is scaled manually
- `Time` column is dropped after EDA confirms no predictive signal

---

## Pipeline

```
1. EDA               → class distribution, Time vs Class, Amount vs Class
2. Preprocessing     → drop Time, scale Amount, PowerTransform V-features
3. Train/Test Split  → stratified to preserve class ratio
4. Baseline Models   → trained on raw imbalanced data
5. Imbalance Handling→ Undersampling, Oversampling, SMOTE, AdaSyn
6. Balanced Models   → same four algorithms retrained on each balanced set
7. Evaluation        → ROC-AUC, Sensitivity, F1, optimal threshold from ROC curve
8. Model Selection   → cost-benefit framing for banking context
```

---

## Results

### Baseline (Imbalanced)

| Model               | ROC-AUC | Sensitivity |
|---------------------|---------|-------------|
| Logistic Regression | 0.97    | 0.77        |
| XGBoost             | **0.98**| 0.75        |
| Decision Tree       | 0.92    | 0.58        |
| Random Forest       | 0.96    | 0.62        |

### After Balancing (best per technique)

| Technique     | Best Model          | ROC-AUC | Sensitivity |
|---------------|---------------------|---------|-------------|
| Undersampling | XGBoost             | 0.98    | 0.92        |
| Oversampling  | Logistic Regression | 0.97    | 0.89        |
| SMOTE         | Logistic Regression | **0.97**| **0.90**    |
| AdaSyn        | Logistic Regression | 0.97    | 0.95        |

**Best model: Logistic Regression + SMOTE** — ROC-AUC 0.97, Sensitivity 0.90, threshold 0.53.

Chosen over XGBoost because the ROC-AUC gap is marginal after balancing, and Logistic Regression is interpretable, lightweight, and easier to justify to a fraud team.

---

## Requirements

```
pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
# place creditcard.csv in root directory
jupyter notebook fraud_detection.ipynb
```

---

## Project Structure

├── fraud_detection.ipynb
├── requirements.txt
└── README.md