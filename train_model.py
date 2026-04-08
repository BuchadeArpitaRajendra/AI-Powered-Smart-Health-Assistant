# train_model.py - COMPLETE FIXED VERSION

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')  # To keep output clean

print("=" * 50)
print("🤖 AI HEALTH ASSISTANT - MODEL TRAINING")
print("=" * 50)

# Load the dataset
print("\n📂 Loading dataset...")
data = pd.read_csv('data/symptoms.csv')
print(f"✅ Loaded {len(data)} patient records")

# Show what the data looks like
print("\n📊 Sample of our data:")
print(data.head(3))

# Prepare features (X) - what we know about the patient
feature_columns = ['fever', 'cough', 'fatigue', 'headache', 'age']
X = data[feature_columns]

# Prepare target (y) - what we want to predict
y = data['disease']

print(f"\n📋 Input features: {', '.join(feature_columns)}")
print(f"🎯 Output: Disease name")

# Convert disease names to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\n📋 Disease codes:")
for i, name in enumerate(label_encoder.classes_):
    print(f"   {i} = {name}")

# Split data (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42
)

print(f"\n📊 Training data: {len(X_train)} patients")
print(f"📊 Testing data: {len(X_test)} patients")

# Create and train model
print("\n🤖 Training Decision Tree model...")
model = DecisionTreeClassifier(
    max_depth=3,           # Simple tree
    random_state=42
)
model.fit(X_train, y_train)

# Calculate accuracy
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"\n📈 Accuracy:")
print(f"   Training: {train_score * 100:.1f}%")
print(f"   Testing: {test_score * 100:.1f}%")

# Save the model and encoder
print("\n💾 Saving model files...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/disease_model.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(feature_columns, 'models/feature_columns.pkl')  # Save feature names!
print("✅ Saved: models/disease_model.pkl")
print("✅ Saved: models/label_encoder.pkl")
print("✅ Saved: models/feature_columns.pkl")

# Test with proper format
print("\n" + "=" * 50)
print("🧪 TESTING THE MODEL")
print("=" * 50)

# Create test data AS A DATAFRAME (with column names)
test_data = pd.DataFrame([[1, 1, 0, 0, 30]], 
                         columns=['fever', 'cough', 'fatigue', 'headache', 'age'])

print("\n🔬 Test Case 1: Patient with Fever and Cough")
print(f"   Symptoms: Fever=Yes, Cough=Yes, Fatigue=No, Headache=No, Age=30")

prediction_num = model.predict(test_data)[0]
predicted_disease = label_encoder.inverse_transform([prediction_num])[0]
print(f"   🤖 AI Prediction: {predicted_disease}")

# Test case 2
test_data2 = pd.DataFrame([[0, 0, 0, 0, 25]], 
                          columns=['fever', 'cough', 'fatigue', 'headache', 'age'])

print("\n🔬 Test Case 2: Patient with No Symptoms")
print(f"   Symptoms: None, Age=25")

prediction_num2 = model.predict(test_data2)[0]
predicted_disease2 = label_encoder.inverse_transform([prediction_num2])[0]
print(f"   🤖 AI Prediction: {predicted_disease2}")

# Test case 3
test_data3 = pd.DataFrame([[1, 1, 1, 1, 50]], 
                          columns=['fever', 'cough', 'fatigue', 'headache', 'age'])

print("\n🔬 Test Case 3: Patient with All Symptoms")
print(f"   Symptoms: All Yes, Age=50")

prediction_num3 = model.predict(test_data3)[0]
predicted_disease3 = label_encoder.inverse_transform([prediction_num3])[0]
print(f"   🤖 AI Prediction: {predicted_disease3}")

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE! Model is ready for the web app.")
print("=" * 50)