import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

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
    filtered_data['GH_020'] & filtered_data['GH_010']
)

selected_vars = [
    'PROVID', 'GRADE', 'DVURBAN', 'DVDESCRIBE', 'DVORIENT',  # Demographics
    'BEH_010', 'PH_010',  # Behavioral
    'ALC_040', 'ALC_010',  # Alcohol
    'CAN_010',  # Cannabis
    'SS_010', 'SS_030', # Smoking
]

X = filtered_data[selected_vars]
y = filtered_data['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(64, input_dim=len(selected_vars), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.25, callbacks=[early_stopping, reduce_lr])

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Plot Training History
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

