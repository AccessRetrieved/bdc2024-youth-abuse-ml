from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# data = pd.read_csv('Data.csv')
# data = data.fillna(data.mode().iloc[0])
# data_encoded = data.replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)

def replaceTarget(value):
    return 1 if 1 <= value <= 3 else 0

def clean_data(data):
    filtered_data = data.loc[
        (~data['GH_010'].isin([99, 6])) &
        (~data['GH_020'].isin([99, 6]))
    ].copy()
    
    filtered_data.loc[:, 'GH_010'] = filtered_data['GH_010'].apply(replaceTarget)
    filtered_data.loc[:, 'GH_020'] = filtered_data['GH_020'].apply(replaceTarget)
    filtered_data.loc[:, 'target_column'] = (
        filtered_data['GH_020'] & filtered_data['GH_010']
    )
    return filtered_data


data = pd.read_csv('Data.csv')
data = data.dropna()
filtered_data = clean_data(data)

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
selected_features_indices = selector.get_support(indices=True)
selected_features = [selected_vars[i] for i in selected_features_indices]
print("Selected Features:", selected_features)

X_test_selected = selector.transform(X_test)

smote_tomek = SMOTETomek(smote=SMOTE(k_neighbors=1), random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_selected, y_train)

# param_grid = {
#     'n_estimators': [50, 100, 200, 300, 400, 500],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# # grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=100, cv=10, scoring='accuracy', n_jobs=-1, random_state=42)
# grid.fit(X_train_resampled, y_train_resampled)

best_rf_model = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=4, min_samples_split=5, bootstrap=True)
best_rf_model.fit(X_train_resampled, y_train_resampled)
# best_rf_model = grid.best_estimator_
# print("Best Model Parameters:", grid.best_params_)

y_pred = best_rf_model.predict(X_test_selected)
y_pred_prob = best_rf_model.predict_proba(X_test_selected)[:, 1] * 100

print(f'\n\nAccuracy: {accuracy_score(y_test, y_pred)}', end='\n\n')
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))

feature_importance = pd.Series(best_rf_model.feature_importances_, index=selected_features).sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)

joblib.dump(best_rf_model, 'random_forest_model.pkl')

output_df = pd.DataFrame(X_test_selected, columns=selected_features)
output_df['Predicted Probability (%)'] = y_pred_prob
print(output_df[['Predicted Probability (%)']].head())