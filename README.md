# 🏠 House Price Prediction

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Predict house sale prices using machine learning with a complete data science pipeline—from preprocessing to model training, evaluation, and feature interpretation.

**Dataset:** [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

## 📂 Project Structure

```
house_price_prediction.ipynb   # Main Jupyter Notebook
train.csv                     # Training dataset (includes SalePrice)
test.csv                      # Test dataset
data_description.txt          # Feature description
requirements.txt              # Python dependencies
final_pipeline.pkl            # Serialized trained pipeline
preprocessor.pkl              # Serialized preprocessing pipeline
sample_submission.csv         # Example submission
submission_final.csv          # Final model predictions
```

---

## 🧩 Methodology

1. **Data Loading & Splitting**

   - Load `train.csv` & `test.csv`.
   - Log-transform `SalePrice` to reduce skewness.
   - Split data into training and hold-out sets.

2. **Data Preprocessing**

   - `ColumnTransformer` pipeline:

     - **Numerical**: Impute missing values + scale
     - **Categorical**: Impute missing values + one-hot encode

3. **Model Training & Evaluation**

| Model             | Mean CV RMSE | Mean CV MAE | Mean CV R² |
| ----------------- | ------------ | ----------- | ---------- |
| Linear Regression | 0.1693       | 18,364.74   | 0.8192     |
| Random Forest     | 0.1439       | 18,452.47   | 0.8561     |
| XGBoost           | 0.1442       | 18,077.56   | 0.8600     |

4. **Final Model**

   - XGBoost selected and retrained on full training data.

5. **Hold-out Evaluation**

| Metric | Value     |
| ------ | --------- |
| RMSE   | 26,259.47 |
| MAE    | 17,175.15 |
| R²     | 0.8877    |

6. **Interpretability**

   - SHAP analysis to understand feature contributions.

---

## ⚙️ Installation

1. **Create virtual environment**

   - Windows:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   - macOS/Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Notebook

```bash
jupyter notebook
```

- Open `house_price_prediction.ipynb`
- Run all cells to preprocess data, train models, evaluate performance, and generate `submission_final.csv`.

---

## ⚡ Quick Start

```bash
# Activate environment and run the notebook in one command
source venv/bin/activate && jupyter notebook house_price_prediction.ipynb
```

---

## 📊 Features

- Complete preprocessing pipeline
- Model comparison: Linear Regression, Random Forest, XGBoost
- SHAP-based feature importance visualization
- Easy-to-use final pipeline (`final_pipeline.pkl`) for predictions

---

## 📖 References

- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/)
- [SHAP library](https://github.com/slundberg/shap)
