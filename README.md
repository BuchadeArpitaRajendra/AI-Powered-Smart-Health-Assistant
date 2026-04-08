# 🏥 AI-Powered Smart Health Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent web application that predicts diseases based on symptoms using Machine Learning, provides severity scoring, health risk assessment, and personalized preventive suggestions with explainable AI.

## 🌟 Live Demo

## ✨ Features

### 1. 🤖 Disease Prediction
- Uses Decision Tree Machine Learning algorithm
- Predicts 5 conditions: Common Cold, Flu, Covid, Migraine, Healthy
- 85-90% accuracy on test data

### 2. ⚠️ Severity Score (0-10)
- Real-time severity calculation based on:
  - Number of symptoms present
  - Patient age
  - Disease type
- Color-coded severity levels (Green → Yellow → Orange → Red)

### 3. 📊 Health Risk Score (0-100)
- Long-term health risk assessment
- Factors considered:
  - Age-based risk (18-78+ years)
  - Symptom count impact
  - Disease chronicity
- Actionable risk levels with recommendations

### 4. 💡 Preventive Suggestions
- Personalized recommendations based on:
  - Predicted disease
  - Risk score
  - Age group
- Includes lifestyle modifications and medical advice

### 5. 🔍 Explainable AI
- Clear explanations for each prediction
- Why the AI made specific diagnosis
- Alternative possibilities based on symptoms
- Confidence score for each prediction

### 6. 🎨 Professional UI
- Modern gradient design
- Fully responsive (works on mobile, tablet, desktop)
- Smooth animations and transitions
- Color-coded health indicators
## 🎯 How It Works

### Disease Logic Matrix:

| Symptom Pattern | Disease | Severity |
|----------------|---------|----------|
| Fever + Cough | Common Cold | Mild-Moderate |
| All 4 symptoms | Flu | Severe |
| Fever only (esp. elderly) | Covid | Severe-Critical |
| Headache only | Migraine | Mild-Moderate |
| No symptoms | Healthy | None |

## 🛠 Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework
- **Scikit-learn** - Machine learning (Decision Tree Classifier)
- **Pandas** - Data manipulation
- **Joblib** - Model persistence

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **Jinja2** - Template engine

### Database
- **CSV** - Lightweight dataset storage (220+ patient records)

## 📥 Installation Guide

### Prerequisites
- Python 3.8 or higher installed
- Git (optional, for cloning)
- Basic understanding of command line

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-health-assistant.git
cd ai-health-assistant
pip install flask pandas scikit-learn joblib
python train_model.py
python app.py
Navigate to: http://127.0.0.1:5000
