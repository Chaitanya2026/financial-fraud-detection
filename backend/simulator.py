@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive actual input from frontend
        data = request.get_json()
        df_input = pd.DataFrame([data])
        df_input = df_input[expected_cols]  # match column order

        # Predict with model
        prediction = model.predict(df_input)[0]

        # Generate GenAI message
        if prediction == 1:
            prediction_label = generate_fraud_alert()
        else:
            prediction_label = "✅ Legit Transaction"

        return jsonify({
            "prediction": prediction_label,
            "actual": "Unknown"  # since we're not sending true labels from frontend
        })

    except Exception as e:
        return jsonify({"error": str(e)})
