import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

data = pd.read_excel('/content/drive/MyDrive/Traffic_Accidents.xlsx')

X = data[['Traffic_Volume', 'Road_Type', 'Weather_Conditions', 'Speed_of_Vehicle', 'Traffic_Density']]
y = data['Accident_Occurred']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}

cv = StratifiedKFold(n_splits=3)

svm_model = GridSearchCV(SVC(), param_grid, cv=cv)

svm_model.fit(X_train_scaled, y_train)

def predict_accident_risk(real_time_data):

    real_time_data_encoded = pd.get_dummies(real_time_data)

    real_time_data_encoded = real_time_data_encoded.reindex(columns=X_train.columns, fill_value=0)

    real_time_data_scaled = scaler.transform(real_time_data_encoded)

    prediction = svm_model.predict(real_time_data_scaled)

    reason = []
    if real_time_data['Speed_of_Vehicle'].values[0] > 80:
        reason.append("high speed")
    if real_time_data['Weather_Conditions_Clear'].values[0] == 0:
        reason.append("bad weather conditions")
    if real_time_data['Traffic_Density_High'].values[0] == 1:
        reason.append("high traffic density")

    if prediction[0] == 1:
        reason_message = " and ".join(reason) if reason else "multiple factors"
        return f"High risk of accident due to {reason_message}."
    else:
        return "Low risk of accident."
real_time_data1 = pd.DataFrame({
    'Traffic_Volume': [1200],
    'Road_Type_Highway': [0], 'Road_Type_Urban': [1], 'Road_Type_Rural': [0],
    'Weather_Conditions_Clear': [0], 'Weather_Conditions_Rain': [0], 'Weather_Conditions_Fog': [1],
    'Speed_of_Vehicle': [70],
    'Traffic_Density_Low': [0], 'Traffic_Density_Medium': [1], 'Traffic_Density_High': [0]
})
real_time_data2 = pd.DataFrame({
    'Traffic_Volume': [800],
    'Road_Type_Highway': [0], 'Road_Type_Urban': [0], 'Road_Type_Rural': [1],
    'Weather_Conditions_Clear': [1], 'Weather_Conditions_Rain': [0], 'Weather_Conditions_Fog': [0],
    'Speed_of_Vehicle': [90],
    'Traffic_Density_Low': [1], 'Traffic_Density_Medium': [0], 'Traffic_Density_High': [0]
})

real_time_data3 = pd.DataFrame({
    'Traffic_Volume': [500],
    'Road_Type_Highway': [0], 'Road_Type_Urban': [1], 'Road_Type_Rural': [0],
    'Weather_Conditions_Clear': [1], 'Weather_Conditions_Rain': [0], 'Weather_Conditions_Fog': [0],
    'Speed_of_Vehicle': [40],
    'Traffic_Density_Low': [1], 'Traffic_Density_Medium': [0], 'Traffic_Density_High': [0]
})

print(predict_accident_risk(real_time_data1))
print(predict_accident_risk(real_time_data2))
print(predict_accident_risk(real_time_data3))

