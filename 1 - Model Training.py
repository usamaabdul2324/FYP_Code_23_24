import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'heart.csv'
heart_data = pd.read_csv(file_path)

# Data Preprocessing
heart_data['Sex'] = heart_data['Sex'].map({'M': 'Male', 'F': 'Female'})
heart_data['ChestPainType'] = heart_data['ChestPainType'].map({'ATA': 'ATA', 'NAP': 'NAP', 'ASY': 'ASY', 'TA': 'TA'})
heart_data['FastingBS'] = heart_data['FastingBS'].map({0: '0', 1: '1'})
heart_data['RestingECG'] = heart_data['RestingECG'].map({'Normal': 'Normal', 'ST': 'ST', 'LVH': 'LVH'})
heart_data['ExerciseAngina'] = heart_data['ExerciseAngina'].map({'N': 'No', 'Y': 'Yes'})
heart_data['ST_Slope'] = heart_data['ST_Slope'].map({'Up': 'Up', 'Flat': 'Flat', 'Down': 'Down'})
heart_data_encoded = pd.get_dummies(heart_data, drop_first=True)

# Splitting the data
X = heart_data_encoded.drop('HeartFailure', axis=1)
y = heart_data_encoded['HeartFailure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=806)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
initial_accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))