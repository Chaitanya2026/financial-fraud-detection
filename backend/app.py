from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import openai

# === INITIAL SETUP ===
app = Flask(__name__)
CORS(app)

# === LOAD MODEL AND DATA ===
model = joblib.load("fraud_model_rf.pkl")
df = pd.read_csv("creditcard.csv")
expected_cols = df.drop("Class", axis=1).columns.tolist()

# === OPENAI API KEY ===
import os
openai.api_key = os.getenv("OPENAI_API_KEY")  

# === HOMEPAGE ===
@app.route('/')
def home():
    return jsonify({"message": "Fraud Detection API is running"})

# === GENAI ALERT MESSAGE GENERATOR ===
def generate_fraud_alert():
    prompt = "Generate a short, smart fraud alert message for a suspicious transaction:"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=30,
            temperature=0.7
        )
        msg = response.choices[0].text.strip()
        return f"🚨 {msg}" if msg else "🚨 Fraudulent activity detected!"
    except Exception as e:
        print("OpenAI API Error:", e)
        return "🚨 Fraudulent activity detected!"

# === PREDICTION ROUTE ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        df_input = df_input[expected_cols]

        # Predict correctly on new input
        prediction = model.predict(df_input)[0]

        if prediction == 1:
            prediction_label = generate_fraud_alert()
        else:
            prediction_label = "✅ Legit Transaction"

        return jsonify({
            "prediction": prediction_label,
            "actual": "Fraud" if prediction == 1 else "Legit"
        })

    except Exception as e:
        return jsonify({"error": str(e)})



# === RUN SERVER ===
if __name__ == '__main__':
    app.run(debug=True)
