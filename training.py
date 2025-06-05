import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import joblib

# Load dataset
data = pd.read_csv('data.csv')

# Encode target column (Yes/No to 1/0)
label_encoder = LabelEncoder()# Initialize label encoder
data['Chance of Admission'] = label_encoder.fit_transform(data['Chance of Admission'])# Encode target values

# Split features and target
X = data.drop('Chance of Admission', axis=1)# Input features (drop target column)
y = data['Chance of Admission'] # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler() #Initialize standard scaler
X_train = scaler.fit_transform(X_train)# Fit on training data and transform
X_test = scaler.transform(X_test)# Transform test data using same scaler

# Build deep learning model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)), # First hidden layer with 16 units and ReLU
    Dense(8, activation='relu'),# Second hidden layer with 8 units and ReLU
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")# Print accuracy in percentage

# Save model and preprocessing tools
model.save('model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')