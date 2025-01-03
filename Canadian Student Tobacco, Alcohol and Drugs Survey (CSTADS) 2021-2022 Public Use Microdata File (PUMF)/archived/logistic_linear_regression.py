from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

data = pd.read_csv('Data.csv')
data = data.fillna(data.mode().iloc[0])
data_encoded = data.replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)

# selected_vars = [
#     # smoking
#     'SS_010', # have they ever tried smoking
#     'SS_020', # how old when first tried smoking
#     'SS_030', # have they ever tried smoking a whole cigarette
#     'SS_040', # have they smoked 100 or more whole cigarettes in their life
#     'WP_040a', # # of whole cigarette smoked on Sunday (over the last 7 days)
#     'WP_040b', # # of whole cigarette smoked on Monday (over the last 7 days)
#     'WP_040c', # # of whole cigarette smoked on Tuesday (over the last 7 days)
#     'WP_040d', # # of whole cigarette smoked on Wednesday (over the last 7 days) 
#     'WP_040e', # # of whole cigarette smoked on Thursday (over the last 7 days)
#     'WP_040f', # # of whole cigarette smoked on Friday (over the last 7 days)
#     'WP_040g', # # of whole cigarette smoked on Saturday (over the last 7 days)

#     # alcohol
#     'ALC_010', # have they ever tried alcohol (more than a sip)
#     'ALC_020', # have they ever tried alcohol (more than a sip) in the last year
#     'ALC_030', # age at first drink of alcohol (more than a sip)
#     'ALC_040', # how often did they have a drink of alchol (more than a sip) in the past 30 days
#     'ALC_050', # how often did they have 5 or more drinks of alcohol on one occasion (in the last year)
#     'ALC_060', # age when they first had 5 or more drinks of alcohol in one occasion
#     'ALC_071', # how often did they have 5 or more drinks of alcohol on one occasion (in the last 30 days)

#     # cannabis
#     'CAN_010',
#     'CAN_020',
#     'CAN_030',
#     'CAN_040',
#     'CAN_050',
#     'CAN_060',
#     'CAN_070',
#     'CAN_080',
#     'CAN_091',
#     'CAN_092',
#     'CAN_100',
#     'CAN_110',
#     'CAN_121',
#     'CAN_130',
#     'CAN_140',

#     # other
#     'BEH_010', # behaviour indicators
#     'GRADE', # grade level
#     'DVURBAN', # where they live; urban/rural
#     'ELC_026a', # e cigarettes
#     'PH_010', # physical activity levels,
#     'DVDESCRIBE', # race,
#     'DVORIENT' # sexual orientation
# ]

# testing below
selected_vars = [
    'GRADE', 'DVURBAN', 'DVDESCRIBE', 'DVORIENT', # Demographics
    'BEH_010', 'PH_010',                         # Behavioral
    'ALC_010', 'ALC_040', 'ALC_050',             # Alcohol
    'CAN_030',                                   # Cannabis
    'SS_010',                                    # Smoking
    'ELC_026a',                                  # E-cigarettes
    'UND_010'
]

# data_encoded['target_column'] = (
#     data_encoded['GH_010'] & # general physical health
#     data_encoded['GH_020'] # general mental health
# ).astype(int)


# testing below
data_encoded['target_column'] = (
    data_encoded['ALC_020'] | data_encoded['CAN_010'] | data_encoded['SS_040']
).astype(int)


X = data_encoded[selected_vars]
y = data_encoded['target_column'] # substance use

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)

selected_features_indices = selector.get_support(indices=True)
selected_features = [selected_vars[i] for i in selected_features_indices]
print("Selected Features:", selected_features)

X_test_selected = selector.transform(X_test)

# smote = SMOTE(random_state=42)
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_selected, y_train)

param_grid = {
    'C': [0.1, 1, 10, 100],
    # 'solver': ['liblinear', 'lbfgs']
    'solver': ['liblinear']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced', C=100, solver='liblinear'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_resampled, y_train_resampled)

best_model = grid.best_estimator_
print("Best Model:", best_model)

# best_model.fit(X_train_resampled, y_train_resampled)

y_pred = best_model.predict(X_test_selected)
y_pred_prob = best_model.predict_proba(X_test_selected)[:, 1] * 100

# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_resampled, y_train_resampled)

# y_pred = model.predict(X_test_selected)
# y_pred_prob = model.predict_proba(X_test_selected)[:, 1] * 100

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification report:\n', classification_report(y_test, y_pred))

output_df = pd.DataFrame(X_test_selected, columns=selected_features)
output_df['Predicted Probability (%)'] = y_pred_prob
print(output_df[['Predicted Probability (%)']].head())

joblib.dump(best_model, 'logistic_model.pkl')