# ui.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import string
import math

app = Flask(__name__)

# -------------------------
# Dictionary / entropy utils
# -------------------------
COMMON_WORDS = {"password", "admin", "welcome", "qwerty", "letmein", "login"}

def calculate_entropy(password: str) -> float:
    """Shannon-like entropy approximation based on distinct character set size."""
    password = str(password) or ""
    if not password:
        return 0.0
    char_set_size = len(set(password))
    return len(password) * math.log2(char_set_size) if char_set_size > 0 else 0.0

def contains_common_word(password: str) -> bool:
    """Return True if password contains any word from COMMON_WORDS (case-insensitive)."""
    lower_pw = str(password).lower()
    return any(word in lower_pw for word in COMMON_WORDS)

# -------------------------
# Feature extraction
# -------------------------
def extract_features(password: str):
    """Returns the full feature vector used by the model pipeline.

    Features:
    - length, digit_count, uppercase_count, lowercase_count, special_count
    - has_digits, has_upper, has_lower, has_special
    - repeated_chars, consecutive_digits, consecutive_letters
    - entropy, has_common_word
    """
    password = str(password).strip()
    length = len(password)
    digit_count = sum(c.isdigit() for c in password)
    uppercase_count = sum(c.isupper() for c in password)
    lowercase_count = sum(c.islower() for c in password)
    special_count = sum(not c.isalnum() for c in password)

    has_digits = int(digit_count > 0)
    has_upper = int(uppercase_count > 0)
    has_lower = int(lowercase_count > 0)
    has_special = int(special_count > 0)

    repeated_chars = int(sum(password.count(c) > 1 for c in set(password)))
    consecutive_digits = int(sum(1 for i in range(len(password)-1) if password[i].isdigit() and password[i+1].isdigit()))
    consecutive_letters = int(sum(1 for i in range(len(password)-1) if password[i].isalpha() and password[i+1].isalpha()))

    entropy = calculate_entropy(password)
    has_common_word = int(contains_common_word(password))

    return [
        length, digit_count, uppercase_count, lowercase_count, special_count,
        has_digits, has_upper, has_lower, has_special, repeated_chars,
        consecutive_digits, consecutive_letters, entropy, has_common_word
    ]

# -------------------------
# Load trained pipeline
# -------------------------
MODEL_PATH = "finla_pipeline.pkl"  # relative path — ensure this file exists in project root

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print(f"[INFO] Loaded model pipeline from '{MODEL_PATH}'")
except FileNotFoundError:
    model = None
    print(f"[ERROR] Model file '{MODEL_PATH}' not found. Please ensure the trained pipeline exists.")

# -------------------------
# Flask routes
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    password = None
    details = None  # optional: we'll return feature info if model present
    if request.method == 'POST':
        password = request.form.get('password', '')
        features = np.array([extract_features(password)])

        # If model is a pipeline it will handle selection/scaling internally.
        if model is None:
            result = "Model not loaded (check server logs)."
        else:
            try:
                pred = model.predict(features)[0]
                strength_map = {0: 'Weak', 1: 'Moderate', 2: 'Strong'}
                result = strength_map.get(int(pred), 'Unknown')
                # Optionally include feature values (for debugging / UI display)
                details = {
                    "features": features.tolist()[0],
                    "prediction_raw": int(pred)
                }
            except Exception as e:
                result = f"Prediction error: {e}"

    return render_template('index.html', result=result, password=password, details=details)

if __name__ == '__main__':
    app.run(debug=True)
