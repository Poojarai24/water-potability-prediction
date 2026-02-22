#predicting the water potability by the model which is in production in mlflow registry
import mlflow
import pandas as pd

mlflow.set_tracking_uri(
    "https://dagshub.com/Poojarai24/water-potability-prediction.mlflow"
)

model_name = "Best_Model"

try:
    # 🔥 Load directly from registry Production stage
    model_uri = f"models:/{model_name}/Production"
    print("Loading model from:", model_uri)

    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")

    # Input data
    data = pd.DataFrame({
        'ph': [3.71608],
        'Hardness': [204.89045],
        'Solids': [20791.318981],
        'Chloramines': [7.300212],
        'Sulfate': [368.516441],
        'Conductivity': [564.308654],
        'Organic_carbon': [10.379783],
        'Trihalomethanes': [86.99097],
        'Turbidity': [2.963135]
    })

    prediction = loaded_model.predict(data)
    print("Prediction:", prediction)

except Exception as e:
    print("Error:", e)