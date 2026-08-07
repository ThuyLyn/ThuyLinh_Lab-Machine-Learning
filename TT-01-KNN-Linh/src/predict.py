import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_pipeline.joblib")

# Load model đã train
model = joblib.load(MODEL_PATH)

# Dữ liệu bệnh nhân mới
patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 120,
    "BMI": 31.2,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 45
}])

# Dự đoán
prediction = model.predict(patient)

if prediction[0] == 1:
    print("Kết quả: Có nguy cơ mắc tiểu đường")
else:
    print("Kết quả: Không có nguy cơ mắc tiểu đường")