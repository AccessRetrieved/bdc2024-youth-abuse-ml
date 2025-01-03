import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('USE_student_addiction_dataset_training.csv')

data_imputed = data.fillna(data.mode().iloc[0])
data_encoded = data_imputed.replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)

# split data into features and target
X = data_encoded.drop(columns=['Addiction_Class'])
y = data_encoded['Addiction_Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.to_csv('processed_datasets/X_train.csv', index=False)
X_test.to_csv('processed_datasets/X_test.csv', index=False)
y_train.to_csv('processed_datasets/y_train.csv', index=False)
y_test.to_csv('processed_datasets/y_test.csv', index=False)

print(f'Features shape: {X_train.shape}')
print(f'Target distribution in training set: \n{y_train.value_counts()}')
