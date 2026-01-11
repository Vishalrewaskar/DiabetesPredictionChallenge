# Diabetes Prediction - Kaggle Competition Workflow

A quick reference for understanding the complete ML competition workflow from data to submission.

---

## 🎯 Project Overview

**Problem**: Predict diabetes diagnosis (Yes/No) based on health metrics  
**Model**: Neural Network (TensorFlow/Keras)  
**Result**: ~66% validation accuracy

---

## 📊 Dataset Quick Facts

**Features (25 total)**:
- 15 continuous (age, BMI, blood pressure, cholesterol, etc.)
- 3 ordinal (smoking, education, income - have natural order)
- 3 nominal (gender, ethnicity, employment - no order)
- 3 binary (family history, hypertension, cardiovascular)

**Target**: `diagnosed_diabetes` (0 or 1)

---

## 🔄 Complete Workflow

### **Phase 1: Data Exploration & Preprocessing** (`EDA.ipynb`)

**What**: Understand data and prepare features  
**Where**: Raw `train.csv` → Processed `train_encoded.csv`

#### Key Steps:

1. **Load & Inspect**
   - Check shape, missing values, duplicates
   - Identify feature types (numerical vs categorical)

2. **Encode Categorical Features**
   - **Ordinal Encoding** (features with order):
     - Smoking: Never=0, Former=1, Current=2
     - Education: No formal=0, Highschool=1, Graduate=2, Postgraduate=3
     - Income: Low=0, Lower-Middle=1, Middle=2, Upper-Middle=3, High=4
   
   - **One-Hot Encoding** (features without order):
     - Gender, Ethnicity, Employment Status
     - Use `drop_first=True` to avoid multicollinearity

3. **Save Processed Data**
   - Save encoded data for training phase

**Why These Encodings?**
- Ordinal preserves natural order (e.g., low → high income)
- One-hot prevents false assumptions (e.g., Male ≠ 1, Female ≠ 0)

---

### **Phase 2: Model Training** (`training.ipynb`)

**What**: Build, train, and save model  
**Where**: `train_encoded.csv` → Model files

#### Key Steps:

1. **Split Data**
   - 80% training (560K), 20% validation (140K)
   - Use `stratify=y` to maintain class balance

2. **Scale Features**
   - StandardScaler on continuous features ONLY
   - Fit on train, transform on validation
   - **Save the scaler** (critical for test preprocessing!)

3. **Build Model**
   - Input: 30 features (after encoding)
   - Hidden: 32 → 16 neurons (ReLU + Dropout)
   - Output: 1 neuron (Sigmoid for probability)
   - Loss: Binary Crossentropy
   - Optimizer: Adam

4. **Train with Early Stopping**
   - Max 50 epochs, stops if no improvement for 5 epochs
   - Monitors validation loss
   - Restores best weights automatically

5. **Evaluate**
   - Check validation accuracy (~66%)
   - Plot ROC curve for threshold tuning

6. **Save Artifacts**
   - Model: `diabetes_dl_model.keras`
   - Scaler: `model_accu_69.pkl`

**Why Save Scaler?**  
Test data MUST be scaled using the SAME statistics (mean/std) as training data.

---

### **Phase 3: Test Prediction & Submission** (`test.ipynb`)

**What**: Predict on test data and create submission  
**Where**: `test.csv` → `submission.csv`

#### Key Steps:

1. **Load Artifacts**
   - Load saved model and scaler

2. **Preprocess Test Data**
   - Apply EXACT same encoding as training:
     - Same ordinal mappings
     - Same one-hot encoding
   - **Align columns**: Ensure test has same features as training
   - Scale using saved scaler (`.transform()` NOT `.fit_transform()`)

3. **Generate Predictions**
   - Get probabilities from model
   - Convert to binary using threshold (default 0.5 or tune to 0.6)

4. **Create Submission**
   - Format: `id`, `diagnosed_diabetes`
   - Verify shape (300K rows, 2 columns)
   - Save as CSV without index

**Critical**: Test preprocessing must be IDENTICAL to training!

---

## 🎯 Model Architecture

```
Input (30 features)
    ↓
Dense (32 neurons, ReLU) → Dropout (20%)
    ↓
Dense (16 neurons, ReLU) → Dropout (20%)
    ↓
Dense (1 neuron, Sigmoid)
```

**Total Parameters**: 1,537  
**Regularization**: Dropout + Early Stopping

---

## 💡 Key Takeaways

### Critical Concepts

1. **Preprocessing Consistency**
   - Same encoding on train and test
   - Use `.transform()` not `.fit_transform()` on test
   - Save preprocessing objects

2. **Feature Engineering**
   - Ordinal encoding for ordered categories
   - One-hot encoding for unordered categories
   - Scale only continuous features

3. **Model Training**
   - Validation set for monitoring
   - Early stopping prevents overfitting
   - Save both model AND preprocessing

4. **Submission Format**
   - Match exact column names required
   - Verify row count and data types
   - Binary predictions (0 or 1)

---

## ⚠️ Common Mistakes

| Mistake | Impact | Solution |
|---------|--------|----------|
| Fitting scaler on test data | Data leakage | Use `.transform()` only |
| Different encodings for test | Shape mismatch | Apply exact same transformations |
| Missing column alignment | Model error | Use `.reindex()` with fill |
| Wrong submission format | Rejected submission | Verify columns and types |

---

## 🚀 Quick Reference for Next Competition

### Workflow Checklist

- [ ] Understand problem (classification/regression, metric)
- [ ] Explore data (missing values, types, distribution)
- [ ] Encode categoricals (ordinal vs one-hot)
- [ ] Split data (train/validation with stratification)
- [ ] Scale numericals (fit on train, transform on val/test)
- [ ] Build baseline model
- [ ] Add regularization (dropout, early stopping)
- [ ] Evaluate on validation set
- [ ] Save model + preprocessing objects
- [ ] Preprocess test identically
- [ ] Verify submission format
- [ ] Submit and iterate

### Files to Save

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_encoded.csv` | Preprocessed training data | Model training |
| `model.keras` | Trained model | Test predictions |
| `scaler.pkl` | Fitted scaler | Test preprocessing |
| `submission.csv` | Final predictions | Competition submission |

---

## 📁 Project Structure

```
project/
├── data/
│   ├── train.csv              # Original data
│   ├── test.csv               # Test data
│   ├── train_encoded.csv      # Processed training
│   └── submission.csv         # Final submission
├── models/
│   ├── diabetes_dl_model.keras
│   └── model_accu_69.pkl
└── notebooks/
    ├── EDA.ipynb              # Data prep
    ├── training.ipynb         # Model training
    └── test.ipynb             # Predictions
```

---

**What to Try Next**:
- Hyperparameter tuning (layers, neurons, dropout)
- Class weighting for imbalanced data
- Feature interactions (e.g., BMI × age)
- Ensemble methods

---
