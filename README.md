# Password Strength Tester Using Machine Learning

## 📌 Overview
This project predicts the strength of a password (Weak / Moderate / Strong) using **machine learning** with handcrafted password features.  
It includes:
- A **training script** (`model.py`) to build and save the ML model
- A **Flask web app** (`ui.py`) to test password strength interactively

---

## 📂 Project Structure
├── corrected_password_data.csv # Dataset (password + strength labels)
├── finla_model.pkl # Trained model file
├── model.py # Model training script
├── ui.py # Flask web app for predictions



---

## 📊 Dataset
The dataset contains:
- **password** — password string
- **strength** — integer label:
  - `0` → Weak  
  - `1` → Moderate  
  - `2` → Strong  

---

## 🛠 Feature Engineering
Features extracted from each password:
- `length` — total characters
- `digit_count` — number of digits
- `uppercase_count` — number of uppercase letters
- `lowercase_count` — number of lowercase letters
- `special_count` — number of special characters
- `has_digits` — binary flag (has digits or not)
- `has_upper` — binary flag (has uppercase or not)
- `has_lower` — binary flag (has lowercase or not)
- `has_special` — binary flag (has special characters or not)
- `repeated_chars` — count of characters that appear more than once
- `consecutive_digits` — number of consecutive digits
- `consecutive_letters` — number of consecutive letters

---

## 🤖 Model Training
- **Class balancing**: SMOTE  
- **Feature selection**: SelectKBest (top 5 features)  
- **Scaling**: StandardScaler  
- **Algorithms tested**:
  - Logistic Regression
  - Gaussian Naive Bayes
- **Cross-validation**: Stratified K-Fold (5 splits)
- **Final model**: Logistic Regression saved as `finla_model.pkl`

---

## ⚠ Known Issues
1. **Pipeline mismatch** — The Flask app (`ui.py`) loads the trained model but does not apply the same feature selection & scaling used during training.  
   ✅ *Fix*: Save a pipeline (selector + scaler + classifier) in `model.py` and load it in `ui.py`.
   
2. **Absolute path in `ui.py`** — The current code uses a hardcoded Windows path to load the model.  
   ✅ *Fix*: Use a relative path like:
   ```python
   model = pickle.load(open("finla_model.pkl", "rb"))
