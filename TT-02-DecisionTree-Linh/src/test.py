import os
import joblib
import pandas as pd

# Đường dẫn tới model

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tree.joblib"
)

# Load model

model = joblib.load(MODEL_PATH)

# Dữ liệu nhân viên cần dự đoán

data = pd.DataFrame([
    {
        "Age": 30,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1100,
        "Department": "Sales",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Marketing",
        "EnvironmentSatisfaction": 4,
        "Gender": "Male",
        "HourlyRate": 80,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 1,
        "OverTime": "Yes",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }
])

# Dự đoán

prediction = model.predict(data)

# Hiển thị kết quả

print("Kết quả dự đoán:", prediction[0])

if prediction[0] == "Yes":
    print("→ Nhân viên có khả năng nghỉ việc.")
else:
    print("→ Nhân viên không có khả năng nghỉ việc.")