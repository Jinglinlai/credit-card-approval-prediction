# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None)

# Replace '?' with NaN
cc_apps = cc_apps.replace("?", np.nan)

# Fill missing values
for col in cc_apps.columns:
    if cc_apps[col].dtype == "object":
        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].mode()[0])
    else:
        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].mean())

# Encode categorical variables
cc_apps_encoded = pd.get_dummies(cc_apps, drop_first=True)

# Split features and target
X = cc_apps_encoded.iloc[:, :-1].values
y = cc_apps_encoded.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

y_train_pred = logreg.predict(X_train_scaled)
print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

# GridSearchCV
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "max_iter": [100, 200, 300]
}

grid_model = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=5
)

grid_model.fit(X_train_scaled, y_train)

# Best parameters
print("Best Params:", grid_model.best_params_)
print("Best CV Score:", grid_model.best_score_)

# Best model
best_model = grid_model.best_estimator_

# Final evaluation on test set
y_pred = best_model.predict(X_test_scaled)

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Final Test Accuracy:")
print(best_model.score(X_test_scaled, y_test))