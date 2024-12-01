from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

mlflow.set_tracking_uri('http://localhost:5000') 

MODEL_URI = "models:/IrisClassifier/Production"  
model = mlflow.pyfunc.load_model(MODEL_URI)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        input_data = pd.DataFrame(data["data"], columns=data["columns"])
        
        predictions = model.predict(input_data)
        
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
