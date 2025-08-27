#aap.py
import os
import json
import joblib
import numpy as np
from flask import Flask, request jsonify

#---config---
MODEL_PATH = os.getenv("MODEL_PATH","model/iris_model.pkl") #adjust filename if needed

#--App---
app = Flask(__name__)

#Load once at start up 
try:
	model = joblib.load(MODEL_PATH)
except Exception as e:
	#Fail fastwith a helpful message
	raise Runtime error(f"Could not load model from {MODE_PATH}:{e}")

@app.get("/health")
def health():
	return {"status": "ok"},200

@app.post("/predict")
def predict():
	"""
	Accepts either:
	{"input":[[...Feature vector...],[...]]} #2D LIST
	or 
	{"input":[...Feature vector...]} 	#1D list
	"""
	try:
		payload = request.get_json(force=True)
		x = payload.get("input")
		if x is None:
			return jsonify(error="Missing'Input'"),400
	
		#Normalise to 2D array 
		if isinstance(x, list) and len(x)>0) and not isinstance(x[0], list):
			x = [x]

		X = np.array(x, dtype=float)
		preds=model.predict(X)
		#if your model returns numpy types, converts to python 
		preds = preds.tolist()
		return jsonify (prediction=preds),200
	
	except Exception as e:
		return jsonify(error=str(e)),500

if __name__ == "__main__":
	# loacal dev only; Render will run with gunicorn(see start command below)
	app.run(host="0.0.0.0",port int(os.environ.get("PORT",8000)))