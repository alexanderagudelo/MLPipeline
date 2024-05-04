from flask import Flask, request, render_template
from pycaret.regression import load_model, predict_model
import pandas as pd
import numpy as np


app = Flask(__name__)
model = load_model("model_03052024")
cols = ["age", "sex", "bmi", "children", "smoker", "region"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns=cols)
    prediction = predict_model(model, data=data_unseen, round=0)
    prediction = int(prediction.loc[0, "prediction_label"])
    return render_template("home.html", pred=f"Expected Bill will be {prediction}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
