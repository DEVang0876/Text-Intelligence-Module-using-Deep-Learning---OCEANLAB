from flask import Flask, jsonify, render_template_string, request

from src.predict import predict_text_details


app = Flask(__name__)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Text Intelligence Module</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #1f2933; }
    main { max-width: 920px; margin: 0 auto; padding: 40px 20px; }
    h1 { font-size: 32px; margin: 0 0 16px; }
    form { display: grid; gap: 16px; }
    textarea { min-height: 180px; resize: vertical; padding: 14px; border: 1px solid #ccd3dc; border-radius: 6px; font-size: 16px; line-height: 1.5; }
    button { width: fit-content; padding: 10px 18px; border: 0; border-radius: 6px; background: #126c55; color: white; font-size: 15px; cursor: pointer; }
    section { margin-top: 24px; padding: 18px; background: white; border: 1px solid #dde3ea; border-radius: 8px; }
    .error { border-color: #d64545; color: #9b1c1c; }
    .probability { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #eef1f4; }
  </style>
</head>
<body>
  <main>
    <h1>Text Intelligence Module</h1>
    <form method="post">
      <textarea name="text" placeholder="Paste a news article or headline..." required>{{ text }}</textarea>
      <button type="submit">Predict Category</button>
    </form>
    {% if error %}
      <section class="error">{{ error }}</section>
    {% endif %}
    {% if prediction %}
      <section>
        <h2>{{ prediction.category }}</h2>
        <p>Model: {{ prediction.model_type }}</p>
        {% if prediction.confidence is not none %}
          <p>Confidence: {{ "%.2f"|format(prediction.confidence * 100) }}%</p>
        {% endif %}
        {% for label, probability in prediction.probabilities.items() %}
          <div class="probability"><span>{{ label }}</span><strong>{{ "%.2f"|format(probability * 100) }}%</strong></div>
        {% endfor %}
      </section>
    {% endif %}
  </main>
</body>
</html>
"""


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    prediction = None
    error = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            try:
                prediction = predict_text_details(text)
            except Exception as exc:
                error = str(exc)
    return render_template_string(HTML_TEMPLATE, text=text, prediction=prediction, error=error)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Field `text` is required."}), 400
    try:
        return jsonify(predict_text_details(text))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
