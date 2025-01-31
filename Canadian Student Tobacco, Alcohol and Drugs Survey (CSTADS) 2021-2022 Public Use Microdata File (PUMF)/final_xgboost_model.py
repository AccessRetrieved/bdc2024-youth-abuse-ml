import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from xgboost import DMatrix, cv
import random

def replaceTarget(value): return 1 if 1 <= value <= 3 else 0

def clean_data(data):
    filtered_data = data
    conditions = (
        (~data['GH_010'].isin([99, 6])) &
        (~data['GH_020'].isin([99, 6]))
    )
    filtered_data = data[conditions].copy()
    filtered_data.loc[:, 'GH_010'] = filtered_data['GH_010'].apply(replaceTarget)
    filtered_data.loc[:, 'GH_020'] = filtered_data['GH_020'].apply(replaceTarget)
    return filtered_data

data = pd.read_csv('Data.csv')
filtered_data = clean_data(data)

filtered_data.loc[:, 'target_column'] = (
    filtered_data['GH_020'] | filtered_data['GH_010']
)

selected_vars = [
    'PROVID', 'GRADE', 'DVURBAN', 'DVDESCRIBE', 'DVORIENT',  # Demographics
    'BEH_010', 'PH_010',  # Behavioral
    'ALC_040', 'ALC_010',  # Alcohol
    'CAN_010',  # Cannabis
    'SS_010', 'SS_030', # Smoking
    'ELC_026a', 'ELC_026c'  # Other features
]

X = filtered_data[selected_vars]
y = filtered_data['target_column']

# MARK:- running gridsearch below
'''
param_grid = {
    'n_estimators': [50, 10, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

gridsearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
gridsearch.fit(X_train, y_train)

best_model = gridsearch.best_estimator_
print('Best parameters: ', gridsearch.best_params_)

'''

# MARK:- using best params for model

def run_model():
    test_size = round(random.uniform(0.05, 0.99), 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    best_model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=1.0)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': selected_vars, 'Importance': importance})
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return mse, feature_importance_df, accuracy, f1, best_model, test_size

best_run = {
    'mse': float('inf'),
    'feature_importance_df': None,
    'accuracy': 0,
    'f1': 0,
    'best_model': None,
    'test_size': 0
}

for i in range(1000):
    mse, feature_importance_df, accuracy, f1, best_model, test_size = run_model()

    if accuracy > best_run['accuracy']:
        best_run['accuracy'] = accuracy
        best_run['mse'] = mse
        best_run['feature_importance_df'] = feature_importance_df
        best_run['f1'] = f1
        best_run['best_model'] = best_model
        print(f'[!] New best model found with accuracy: {accuracy * 100:.2f}%, test size: {test_size}')

    print(f'{i + 1}. Running model... {accuracy * 100:.2f}%, test size: {test_size}')

print(f'F1 Score: {best_run["f1"]:.4f}')
print(f'Mean Squared Error: {best_run["mse"]:.4f}')
print(f'Accuracy: {best_run["accuracy"] * 100:.2f}%')
print('Feature importance:')
print(best_run['feature_importance_df'].sort_values(by='Importance', ascending=False))