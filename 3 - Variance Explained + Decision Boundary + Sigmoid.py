import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
classification_rep = classification_report(y_test, y_pred)

# Initialize PCA model
pca = PCA(n_components=X_train_scaled.shape[1])

# Fit the PCA model to the scaled training data
pca.fit(X_train_scaled)

# Calculate the percentage of variance explained by each component
explained_variance_ratio = pca.explained_variance_ratio_ * 100

# Cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Plot the Scree plot with a smooth line
plt.figure(figsize=(7, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, color='blue', label='Individual explained variance')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, color='red', label='Cumulative explained variance', marker='o')
plt.xlabel('Number of principal components')
plt.ylabel('Percentage explained variance')
plt.title('Variance Explained Plot')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Based on the scree plot, we select the first two principal components for visualization
pca_2d = PCA(n_components=2)

# Fit the PCA model to the scaled training data and transform the data
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)

# Train a new logistic regression model on the 2D PCA-transformed data
log_reg_2d = LogisticRegression()
log_reg_2d.fit(X_train_pca_2d, y_train)

# Create a mesh grid for the decision boundary plot
x_min, x_max = X_train_pca_2d[:, 0].min() - 1, X_train_pca_2d[:, 0].max() + 1
y_min, y_max = X_train_pca_2d[:, 1].min() - 1, X_train_pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict each point on the grid
Z = log_reg_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=y_train, s=20, edgecolor='k')

# Title and labels
plt.title('Decision Boundary with PCA Transformed Data & New Data Point')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Let's assume there's a new data point that we want to plot
# For demonstration purposes, we will choose a point at (3, 2)
new_point = pca_2d.transform(scaler.transform([[3, 2] + [0]*(X_train.shape[1] - 2)]))
plt.scatter(new_point[:, 0], new_point[:, 1], c='purple', s=100, marker='.', label='Normal')
plt.scatter(new_point[:, 0], new_point[:, 1], c='yellow', s=100, marker='.', label='HeartFailure')
plt.scatter(new_point[:, 0], new_point[:, 1], c='red', s=100, marker='*', label='New Data Point')
plt.legend()

plt.show()

def plot_sigmoid_and_decision_boundary(X, y, model, scaler, pca_transformer):
    # Plot the sigmoid function for one feature
    plt.figure(figsize=(15, 7))

    # Left plot - Sigmoid function for one feature
    plt.subplot(1, 2, 1)
    # Generate a range of values for the feature
    x_feature = np.linspace(-4, 4, 500)
    # Compute the sigmoid function for these values using the first coefficient and intercept
    y_sigmoid = 1 / (1 + np.exp(-(model.coef_[0][0] * x_feature + model.intercept_[0])))
    # Plot the sigmoid curve
    plt.plot(x_feature, y_sigmoid, label='Sigmoid Function')
    # Plot the decision threshold (0.5 probability)
    plt.axhline(y=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    # Plot the decision boundary (where z=0)
    plt.axvline(x=-model.intercept_[0]/model.coef_[0][0], color='red', linestyle='--', label='Decision Boundary (z=0)')
    plt.xlabel('z')
    plt.ylabel('Probability')
    plt.title('Sigmoid Function for One Feature')
    plt.legend()

    # Right plot - Decision boundary in PCA transformed space
    plt.subplot(1, 2, 2)
    # Transform the original features to the PCA space
    X_pca = pca_transformer.transform(scaler.transform(X))
    # Plot the data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='brg', edgecolor='k', alpha=0.7)
    # Create a mesh grid in the transformed PCA space
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    # Predict the class probabilities on the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='red', linestyles='--')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Decision Boundary in PCA Transformed Space')

    plt.tight_layout()
    plt.show()

# Using the previously trained logistic regression model (log_reg_2d) and PCA object (pca_2d)
plot_sigmoid_and_decision_boundary(X, y, log_reg_2d, scaler, pca_2d)