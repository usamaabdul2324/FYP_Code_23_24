import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# Load the dataset
heart_data = pd.read_csv('heart.csv')

# Encode categorical variables
label_encoders = {}
for column in heart_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    heart_data[column] = le.fit_transform(heart_data[column])
    label_encoders[column] = le

# Splitting the dataset into the Training set and Test set
X = heart_data.drop("HeartFailure", axis=1)
y = heart_data["HeartFailure"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training with Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=0)
log_reg.fit(X_train, y_train)

# Predictions and Model Evaluation
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Calculating Permutation Feature Importance
perm_importance = permutation_importance(log_reg, X_test, y_test, n_repeats=30, random_state=0)

# Organizing the results
perm_importance_df = pd.DataFrame(data={
    'Features': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=True)

# Normalizing the permutation feature importance
min_importance = perm_importance_df['Importance'].min()
perm_importance_df['Normalized Importance'] = perm_importance_df['Importance'] - min_importance

# Plotting the Normalized Permutation Feature Importance
plt.figure(figsize=(10,6))
plt.barh(perm_importance_df['Features'], perm_importance_df['Normalized Importance'], color='skyblue')
plt.xlabel('Normalized Importance')
plt.title('Normalized Permutation Feature Importance')
plt.show()

# Displaying the DataFrame
perm_importance_df[['Features', 'Normalized Importance']]
