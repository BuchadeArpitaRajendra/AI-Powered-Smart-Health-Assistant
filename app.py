# app.py - COMPLETE WITH EXPLAINABLE AI

from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load our trained model
print("📂 Loading AI model...")
model = joblib.load('models/disease_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
print("✅ Model loaded successfully!")

# Function to calculate severity score (0-10)
def calculate_severity(fever, cough, fatigue, headache, age, disease):
    severity = 0
    
    # Each symptom adds 1.5 points
    symptom_count = fever + cough + fatigue + headache
    severity += symptom_count * 1.5
    
    # Age factors
    if age >= 60:
        severity += 3
    elif age >= 50:
        severity += 2
    elif age >= 30:
        severity += 1
    
    # Disease severity
    disease_severity = {
        'Healthy': 0,
        'Common Cold': 1,
        'Migraine': 2,
        'Flu': 3,
        'Covid': 4
    }
    severity += disease_severity.get(disease, 1)
    
    # Cap between 0 and 10
    severity = max(0, min(10, severity))
    severity = round(severity, 1)
    
    return severity

def get_severity_description(score):
    if score <= 2:
        return "🟢 Mild - Home care recommended"
    elif score <= 5:
        return "🟡 Moderate - Monitor symptoms"
    elif score <= 8:
        return "🟠 Severe - Consult a doctor"
    else:
        return "🔴 Critical - Seek immediate medical attention"

# Calculate Health Risk Score
def calculate_health_risk(age, symptom_count, disease):
    risk_score = 0
    
    if age >= 70:
        risk_score += 40
    elif age >= 60:
        risk_score += 35
    elif age >= 50:
        risk_score += 25
    elif age >= 40:
        risk_score += 15
    elif age >= 30:
        risk_score += 8
    elif age >= 18:
        risk_score += 3
    else:
        risk_score += 1
    
    if symptom_count >= 3:
        risk_score += 30
    elif symptom_count == 2:
        risk_score += 20
    elif symptom_count == 1:
        risk_score += 10
    
    disease_risk = {
        'Healthy': 0, 'Common Cold': 10, 'Migraine': 15, 'Flu': 20, 'Covid': 30
    }
    risk_score += disease_risk.get(disease, 10)
    
    return min(100, risk_score)

def get_risk_description(score):
    if score <= 30:
        return "✅ Low Risk - Maintain healthy lifestyle"
    elif score <= 60:
        return "⚠️ Moderate Risk - Consider lifestyle changes"
    elif score <= 80:
        return "⚠️⚠️ High Risk - Consult a doctor for preventive care"
    else:
        return "🚨 Very High Risk - Urgent medical review recommended"

def get_preventive_suggestions(disease, risk_score, age):
    suggestions = []
    suggestions.append("💧 Stay hydrated - drink 8-10 glasses of water daily")
    suggestions.append("😴 Get 7-8 hours of sleep each night")
    
    if disease == "Common Cold":
        suggestions.append("🍵 Drink warm fluids like herbal tea or soup")
        suggestions.append("🧴 Use saline nasal spray for congestion")
        suggestions.append("🍊 Increase vitamin C intake (oranges, lemons)")
    elif disease == "Flu":
        suggestions.append("🛌 Take complete rest for 3-5 days")
        suggestions.append("💊 Take over-the-counter fever reducers if needed")
        suggestions.append("🧴 Use a humidifier for easier breathing")
    elif disease == "Covid":
        suggestions.append("🩸 Monitor blood oxygen levels with a pulse oximeter")
        suggestions.append("🏠 Isolate to prevent spreading to others")
        suggestions.append("📞 Contact a doctor for antiviral medication")
    elif disease == "Migraine":
        suggestions.append("🌑 Rest in a dark, quiet room")
        suggestions.append("💆 Apply cold compress to forehead")
        suggestions.append("☕ Limit caffeine and avoid triggers")
    elif disease == "Healthy":
        suggestions.append("🏃 Exercise 30 minutes daily, 5 times a week")
        suggestions.append("🥗 Eat a balanced diet with fruits and vegetables")
    
    if risk_score > 60:
        suggestions.append("🩺 Schedule a complete health checkup")
        suggestions.append("📊 Track your blood pressure and blood sugar")
    
    if age > 60:
        suggestions.append("🚶 Do gentle exercises like walking or yoga")
        suggestions.append("🥦 Focus on calcium and protein-rich foods")
    
    return suggestions[:5]

# NEW FUNCTION: Explain the prediction
def explain_prediction(fever, cough, fatigue, headache, age, predicted_disease):
    """
    Generate a human-readable explanation of why the AI made this prediction
    """
    explanations = []
    symptom_count = fever + cough + fatigue + headache
    
    # 1. Explain based on symptom pattern
    if predicted_disease == "Common Cold":
        if fever == 1 and cough == 1:
            explanations.append("🔍 The combination of fever and cough is a classic sign of Common Cold")
        if fatigue == 1:
            explanations.append("🔍 Fatigue with cold symptoms is common and expected")
        if headache == 0:
            explanations.append("🔍 The absence of severe headache helps rule out Flu or Migraine")
            
    elif predicted_disease == "Flu":
        if fever == 1 and cough == 1 and fatigue == 1 and headache == 1:
            explanations.append("🔍 Having ALL four symptoms (fever, cough, fatigue, headache) strongly indicates Flu")
        elif symptom_count >= 3:
            explanations.append(f"🔍 With {symptom_count} symptoms present, this matches the Flu pattern")
        explanations.append("🔍 Flu typically presents with sudden onset of multiple symptoms")
            
    elif predicted_disease == "Covid":
        if fever == 1 and cough == 0 and fatigue == 0:
            explanations.append("🔍 Fever as the dominant symptom (without cough/fatigue) is characteristic of Covid")
        if age > 60:
            explanations.append("🔍 Your age group has shown higher susceptibility to Covid")
        if headache == 1:
            explanations.append("🔍 Headache along with fever is a known Covid presentation")
            
    elif predicted_disease == "Migraine":
        if headache == 1 and symptom_count == 1:
            explanations.append("🔍 Headache as the ONLY symptom is the hallmark of Migraine")
        if fever == 0 and cough == 0:
            explanations.append("🔍 The absence of fever and respiratory symptoms rules out viral infections")
            
    elif predicted_disease == "Healthy":
        if symptom_count == 0:
            explanations.append("🔍 With NO symptoms reported, the AI correctly predicts you're healthy")
        else:
            explanations.append("🔍 Your symptoms are mild and don't match any disease pattern strongly")
    
    # 2. Add confidence information
    if symptom_count >= 3:
        explanations.append("📊 High confidence: Multiple symptoms provide clear pattern matching")
    elif symptom_count == 0:
        explanations.append("📊 High confidence: No symptoms clearly indicates healthy state")
    else:
        explanations.append("📊 Moderate confidence: Based on the symptoms you reported")
    
    # 3. Add age consideration
    if age > 60 and predicted_disease != "Healthy":
        explanations.append(f"👴 Age {age} years was considered - older adults often have different symptom presentations")
    elif age < 18 and predicted_disease != "Healthy":
        explanations.append(f"👶 Age {age} years was considered in the diagnosis")
    
    return explanations

# NEW FUNCTION: Get alternative possibilities
def get_alternative_diseases(fever, cough, fatigue, headache, age):
    """
    Suggest other possible diseases based on symptoms
    """
    symptom_count = fever + cough + fatigue + headache
    alternatives = []
    
    if symptom_count >= 3:
        if not (fever == 1 and cough == 1 and fatigue == 1 and headache == 1):
            alternatives.append("Flu (if all symptoms develop)")
    elif symptom_count == 2:
        if fever == 1 and cough == 1:
            alternatives.append("Common Cold or early Flu")
        elif fever == 1 and headache == 1:
            alternatives.append("Covid or viral fever")
    elif symptom_count == 1:
        if fever == 1:
            alternatives.append("Mild viral fever or Covid")
        elif headache == 1:
            alternatives.append("Tension headache or Migraine")
        elif cough == 1:
            alternatives.append("Allergies or mild respiratory infection")
    
    if age > 60 and symptom_count > 0:
        alternatives.append("Consider getting tested for respiratory infections")
    
    return alternatives if alternatives else ["No strong alternative matches"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        fever = int(request.form['fever'])
        cough = int(request.form['cough'])
        fatigue = int(request.form['fatigue'])
        headache = int(request.form['headache'])
        age = int(request.form['age'])
        
        symptom_count = fever + cough + fatigue + headache
        
        # Create DataFrame with symptoms
        user_data = pd.DataFrame([[fever, cough, fatigue, headache, age]], 
                                  columns=feature_columns)
        
        # Make prediction
        prediction_num = model.predict(user_data)[0]
        predicted_disease = label_encoder.inverse_transform([prediction_num])[0]
        
        # Get prediction probabilities (confidence scores)
        probabilities = model.predict_proba(user_data)[0]
        confidence = max(probabilities) * 100  # Highest probability as percentage
        
        # Calculate severity score
        severity_score = calculate_severity(fever, cough, fatigue, headache, age, predicted_disease)
        severity_description = get_severity_description(severity_score)
        
        # Calculate health risk score
        health_risk = calculate_health_risk(age, symptom_count, predicted_disease)
        risk_description = get_risk_description(health_risk)
        
        # Get preventive suggestions
        suggestions = get_preventive_suggestions(predicted_disease, health_risk, age)
        
        # Get explanation
        explanations = explain_prediction(fever, cough, fatigue, headache, age, predicted_disease)
        
        # Get alternative diseases
        alternatives = get_alternative_diseases(fever, cough, fatigue, headache, age)
        
        # Send everything to the result page
        return render_template('result.html', 
                             disease=predicted_disease,
                             fever=fever,
                             cough=cough,
                             fatigue=fatigue,
                             headache=headache,
                             age=age,
                             symptom_count=symptom_count,
                             severity_score=severity_score,
                             severity_description=severity_description,
                             health_risk=health_risk,
                             risk_description=risk_description,
                             suggestions=suggestions,
                             explanations=explanations,
                             alternatives=alternatives,
                             confidence=round(confidence, 1))
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)