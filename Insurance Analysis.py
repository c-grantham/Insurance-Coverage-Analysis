# insurance_pipeline.py
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, mean_squared_error, r2_score)
import joblib

# load data
insurance = pd.read_csv('insurance_dataset.csv')

x_coverage = insurance.iloc[:, 0:10].values
y_coverage = insurance.iloc[:, 10].values

x_cost = insurance.iloc[:, 0:11].values
y_cost = insurance.iloc[:, 11].values

# train/test splits
x_cov_train, x_cov_test, y_cov_train, y_cov_test = train_test_split(
    x_coverage, y_coverage, test_size=0.2, random_state=0, stratify=y_coverage if len(np.unique(y_coverage))>1 else None
)

x_cost_train, x_cost_test, y_cost_train, y_cost_test = train_test_split(
    x_cost, y_cost, test_size=0.2, random_state=1
)

# Preprocessing for coverage model
cov_ct = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=True), [1, 4, 5, 6, 7]),
    ('ordinal', OrdinalEncoder(), [8, 9]),
    ('scale', StandardScaler(), [0, 2, 3])
], remainder='drop', sparse_threshold=0.0)

x_cov_train_trans = cov_ct.fit_transform(x_cov_train)
x_cov_test_trans = cov_ct.transform(x_cov_test)

# encode y for classification (ensure shape is (n,))
ord_enc = OrdinalEncoder()
y_cov_train_enc = ord_enc.fit_transform(y_cov_train.reshape(-1, 1)).ravel()
y_cov_test_enc = ord_enc.transform(y_cov_test.reshape(-1, 1)).ravel()

# Coverage model: logistic regression with GridSearch (use compatible param grid)
log_clf = LogisticRegression(random_state=0, max_iter=1000)

param_grid = [
    # liblinear supports l1/l2 but not elasticnet; keep simple hyperparamization
    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10], 'max_iter': [200]},
    # saga supports elasticnet
    {'solver': ['saga'], 'penalty': ['l1', 'l2', 'elasticnet'], 'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.0, 0.5, 1.0], 'max_iter': [300]}
]

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

search = GridSearchCV(log_clf, param_grid, cv=5, scoring=scoring, refit='accuracy', n_jobs=-1)
search.fit(x_cov_train_trans, y_cov_train_enc)

best_cov_model = search.best_estimator_

# predictions + evaluation for coverage model
ycov_pred = best_cov_model.predict(x_cov_test_trans)
print("=== Coverage classification results ===")
print("Best params:", search.best_params_)
print("Accuracy:", accuracy_score(y_cov_test_enc, ycov_pred))
print("Precision (macro):", precision_score(y_cov_test_enc, ycov_pred, average='macro', zero_division=0))
print("Recall (macro):", recall_score(y_cov_test_enc, ycov_pred, average='macro', zero_division=0))
print("F1 (macro):", f1_score(y_cov_test_enc, ycov_pred, average='macro', zero_division=0))
print("\nClassification report:\n", classification_report(y_cov_test_enc, ycov_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_cov_test_enc, ycov_pred))

# save coverage model and transformer
os.makedirs('models', exist_ok=True)
joblib.dump(best_cov_model, 'models/coverage_model.joblib')
joblib.dump(cov_ct, 'models/coverage_preprocessor.joblib')
joblib.dump(ord_enc, 'models/coverage_label_encoder.joblib')

# ---------------------------
# Cost prediction model
# Option A: include the predicted coverage as an additional feature
# Build preprocessing for cost features (ensure consistent handling)
cost_ct = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False), [1, 4, 5, 6, 7]),
    ('ordinal', OrdinalEncoder(), [8, 9]),
    ('scale', StandardScaler(), [0, 2, 3, 10])  # including col index 10 if numeric feature exists
], remainder='drop', sparse_threshold=0)

# transform cost training features
# if x_cost has fewer/more columns than expected, adjust indices above
x_cost_train_trans = cost_ct.fit_transform(x_cost_train)
x_cost_test_trans = cost_ct.transform(x_cost_test)

# Optionally append predicted coverage label as a feature
# Predict coverage for the cost splits (use cov preprocessor + model)
# We need to transform the original cost X using the coverage preprocessor's expected format
# For simplicity, predict coverage using cov_ct model if columns align
try:
    # use cov_ct to transform matching columns if shapes align
    cov_features_for_cost_train = cov_ct.transform(x_cost_train[:, :10])
    cov_features_for_cost_test = cov_ct.transform(x_cost_test[:, :10])
    cov_pred_train = best_cov_model.predict(cov_features_for_cost_train)
    cov_pred_test = best_cov_model.predict(cov_features_for_cost_test)
    # append as single numeric column to cost features
    x_cost_train_final = np.hstack([x_cost_train_trans, cov_pred_train.reshape(-1, 1)])
    x_cost_test_final = np.hstack([x_cost_test_trans, cov_pred_test.reshape(-1, 1)])
except Exception:
    # fallback: no appended coverage
    x_cost_train_final = x_cost_train_trans
    x_cost_test_final = x_cost_test_trans

# cost model: random forest regressor with simple gridsearch
rf = RandomForestRegressor(random_state=0)
rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf_search = GridSearchCV(rf, rf_param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1)
rf_search.fit(x_cost_train_final, y_cost_train)

best_cost_model = rf_search.best_estimator_

# predictions and evaluation for cost model
ycost_pred = best_cost_model.predict(x_cost_test_final)
rmse = mean_squared_error(y_cost_test, ycost_pred, squared=False)
r2 = r2_score(y_cost_test, ycost_pred)

print("\n=== Cost regression results ===")
print("Best params:", rf_search.best_params_)
print("RMSE:", rmse)
print("R2:", r2)

# save cost model and preprocessor
joblib.dump(best_cost_model, 'models/cost_model.joblib')
joblib.dump(cost_ct, 'models/cost_preprocessor.joblib')

print("\nModels and preprocessors saved to ./models/")
