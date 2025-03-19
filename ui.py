from flask import Flask, render_template, request
import joblib
import numpy as np
import string
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("C:\\Users\\ramak\\Desktop\cdac-project\\finla_model.pkl","rb"))

# Feature extraction function
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

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        password = request.form['password']
        features = np.array([extract_features(password)])
        strength = model.predict(features)[0]
        strength_map = {0: 'Weak', 1: 'Moderate', 2: 'Strong'}
        result = strength_map.get(strength, 'Unknown')
        return render_template('index.html', result=result, password=password)
    return render_template('index.html', result=None, password=None)

if __name__ == '__main__':
    app.run(debug=True)