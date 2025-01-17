# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
data_path = 'datapathforyourtrainingdataset'  # Replace with the correct path to your training dataset
dataset = pd.read_pickle(data_path)

# Preprocessing the features
# Convert 'deviceName' to categorical and encode as integers
dataset['deviceName'] = dataset['deviceName'].astype('category')
dataset['device_cat'] = dataset['deviceName'].cat.codes

# Select Pinh19 features and target variable
features = ['mean_byte_out', 'std_byte_out', 'total_byte_out']
target = 'device_cat'

# Create a new DataFrame for training and testing
data = dataset[features + [target]]
data = data.set_index('start_time')  # Ensure 'start_time' is the index

# Split the data into features (X) and target (y)
X = data.drop(columns=[target])
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Hyperparameter tuning phase using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc_ovr',
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f"Best Parameters from Grid Search: {best_params}")

# Training phase with the best parameters
print("Training the Random Forest model...")
rf_model = RandomForestClassifier(**best_params, random_state=42)
rf_model.fit(X_train, y_train)

# Testing phase
print("Testing the model...")
y_pred_proba = rf_model.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
print(f"Weighted ROC-AUC Score on Test Set: {roc_auc:.4f}")

# Load the testing dataset for a different use case
data_test_path = 'datapathforyourtestingcasedataset'  # Replace with the correct path to your testing case dataset
testset = pd.read_pickle(data_test_path)

# Preprocess the testing dataset
X_usecase = testset.drop(columns=[target])
y_usecase = testset[target]

# Evaluate the model on the new testing dataset
y_pred_proba_usecase = rf_model.predict_proba(X_usecase)
roc_auc_usecase = roc_auc_score(y_usecase, y_pred_proba_usecase, multi_class='ovr', average='weighted')
print(f"Weighted ROC-AUC Score on Testing Use Case: {roc_auc_usecase:.4f}")

# Classification report for further insights
print("Classification Report on Test Set:")
print(classification_report(y_test, rf_model.predict(X_test)))
