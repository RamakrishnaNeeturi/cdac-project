import pandas as pd
import numpy as np
import pickle
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# Load dataset
file_path =("corrected_password_data (1).csv" ) # Replace with your actual path
df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
df.dropna(subset=['password'], inplace=True)
df = df[df['strength'].isin([0, 1, 2])]

# Feature extraction
def extract_features(password):
    password = str(password).strip()
    length = len(password)
    digits = sum(c.isdigit() for c in password)
    uppercase = sum(c.isupper() for c in password)
    lowercase = sum(c.islower() for c in password)
    special_chars = sum(c in string.punctuation for c in password)
    has_digits = int(digits > 0)
    has_upper = int(uppercase > 0)
    has_lower = int(lowercase > 0)
    has_special = int(special_chars > 0)
    repeated_chars = sum(password.count(c) > 1 for c in set(password))
    consecutive_digits = sum(1 for i in range(len(password)-1) if password[i].isdigit() and password[i+1].isdigit())
    consecutive_letters = sum(1 for i in range(len(password)-1) if password[i].isalpha() and password[i+1].isalpha())
    return [length, digits, uppercase, lowercase, special_chars, has_digits, has_upper, has_lower, has_special, repeated_chars, consecutive_digits, consecutive_letters]

feature_columns = [
    "length", "digit_count", "uppercase_count", "lowercase_count", "special_count",
    "has_digits", "has_upper", "has_lower", "has_special", "repeated_chars",
    "consecutive_digits", "consecutive_letters"
]
feature_matrix = np.array([extract_features(pwd) for pwd in df['password']])
features_df = pd.DataFrame(feature_matrix, columns=feature_columns)
features_df['strength'] = df['strength'].values

# 1. Address Class Imbalance *Before* Splitting (using SMOTE)
X = features_df.drop(columns=['strength']).values
y = features_df['strength'].values
smote = SMOTE(random_state=42)

# 2. Stratified K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies_lr = []
accuracies_nb = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Balance the training set only
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


    # Feature Selection (Inside CV)
    selector = SelectKBest(f_classif, k=5)
    X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
    X_test_selected = selector.transform(X_test)

    # Scaling (Inside CV)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=500, penalty='l2', C=0.1, random_state=42, solver='liblinear', tol=0.0005)
    log_reg.fit(X_train_scaled, y_train_resampled)
    log_reg_preds = log_reg.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test, log_reg_preds)
    accuracies_lr.append(acc_lr)

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train_resampled)
    nb_preds = nb_model.predict(X_test_scaled)
    acc_nb = accuracy_score(y_test, nb_preds)
    accuracies_nb.append(acc_nb)

print("Logistic Regression Cross-Validation Accuracies:", accuracies_lr)
print("Average Logistic Regression Accuracy:", np.mean(accuracies_lr))
print("Naive Bayes Cross-Validation Accuracies:", accuracies_nb)
print("Average Naive Bayes Accuracy:", np.mean(accuracies_nb))


# Example of retraining on the full dataset (after CV)
X_resampled_full, y_resampled_full = smote.fit_resample(X, y)

# Feature selection (full data)
selector_full = SelectKBest(f_classif, k=5)
X_selected_full = selector_full.fit_transform(X_resampled_full, y_resampled_full)

# Scaling (full data)
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_selected_full)

# Final Model Training (Logistic  Example)
final_log_reg = LogisticRegression(max_iter=500, penalty='l2', C=0.1, random_state=42, solver='liblinear', tol=0.0005)
final_log_reg.fit(X_scaled_full, y_resampled_full)
pickle.dump(final_log_reg, open("finla_model.pkl","wb"))

# ... (Now use final_log_reg for predictions on new data)