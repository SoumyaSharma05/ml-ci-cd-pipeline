#app.py

import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify


#Config
MODEL_PATH = os.getenv("MODEL_PATH", "model/iris.pkl")

#App

app = Flask(__name__)



#load once at startup

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    #Fail fast with message
    raise RuntimeError(f"Couldn't load the model from {MODEL_PATH}:{e}")
    
    
@app.get("/health")
def health():
    return{"status":"ok"}, 200
    
    
@app.post("/predict")
def predict():
    """
    
    Accepts wither :
    {input":[[feature vector]]} #2d List
    
    or
    
    {"Input":[feature vector]  #1d list
    """
    
    try:
        payload = request.get_json(force=True)
        x = payload.get("input")
        if x is None:
            return jsonify(error="Missing input"), 400

        # Normalize 2d array
        if isinstance(x, list) and (len(x) > 0) and not isinstance(x[0], list):
            x = [x]

        X = np.array(x, dtype=float)
        preds = model.predict(X)
        preds = preds.tolist()
        return jsonify(prediction=preds), 200

    except Exception as e:
        return jsonify(error=str(e)), 500
    
    
    
if __name__ == "__main__":
    #Render wil run with gunicorn
    
    app.run(host = "0.0.0.0", port = int(os.environ.get("PORT", 8000)))
    
    
    