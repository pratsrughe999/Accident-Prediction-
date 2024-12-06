import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('Cleaned_Accident.csv')

# Display columns and data types for debugging
print(data.columns)
print(data.dtypes)

# Preprocessing categorical data (features)
categorical_columns = ['LIGHT CONDITION', 'ROAD CONDITIONS']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Convert all to string first
    label_encoders[col] = le  # Save encoders for later use if needed

# Preprocess the target column (y)
target_encoder = LabelEncoder()
data['INJURY'] = target_encoder.fit_transform(data['INJURY'].astype(str))  # Convert to string, then encode

# Define features and target
X = data[['LIGHT CONDITION', 'ROAD CONDITIONS', 'PASSENGER CAR ACCIDENT',
          'BICYCLE ACCIDENT', 'ACCCIDENT WITH GOODS ROAD VEHICLE']]

# Force conversion of numeric features (ensure no strings remain)
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

y = data['INJURY']

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model and encoders to a file
joblib.dump(model, 'accident_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoder, 'target_encoder.pkl')
print("Model and encoders saved.")
