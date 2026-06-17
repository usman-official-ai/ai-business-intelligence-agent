# 🤖 AI-Powered Business Intelligence Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> An intelligent AI-powered system that classifies companies into 21 industry categories, scores lead quality, and generates actionable sales recommendations.



## 📊 Overview

This project is a complete **AI-powered business intelligence agent** that helps sales, marketing, and business development teams:

- 🔍 **Identify** which industry a company belongs to
- ⭐ **Score** lead quality based on key features
- 💡 **Generate** actionable sales recommendations
- 📊 **Process** single or batch predictions
- 📈 **Visualize** analytics and trends

---

## ✨ Features

### 🏢 Industry Classification
- **21 Industry Categories**: AI, SaaS, Healthcare, Fintech, CleanTech, EdTech, RetailTech, Logistics, HRTech, Cybersecurity, PropTech, Automotive, FoodTech, Energy, Telecommunications, Government, Aerospace, Biotech, NanoTech, Tourism, Manufacturing
- **3 ML Models**: Logistic Regression, Random Forest, Multinomial Naive Bayes
- **Automatic Model Selection**: Best performing model is automatically selected

### ⭐ Lead Scoring
| Feature | Points |
|---------|--------|
| Website Exists | 25 points |
| Contact Form | 20 points |
| Services Count | Up to 30 points |
| Country Score | Up to 24 points |
| Industry Bonus | Up to 10 points |

**Score Categories:**
- 🟢 High Quality: 80-100
- 🟡 Medium Quality: 60-79
- 🟠 Low Quality: 40-59
- 🔴 Poor Quality: 0-39

### 💡 Sales Recommendations
- **Priority based**: Urgent, Follow-up, Nurture, Low Priority
- **Industry specific**: Tailored recommendations for each industry
- **Actionable**: Clear next steps for sales teams

### 📊 Dashboard Features
| Feature | Description |
|---------|-------------|
| 🔍 Single Prediction | Analyze one company at a time |
| 📊 Batch Prediction | Upload CSV and process multiple companies |
| 📈 Analytics | Visualize industry distribution, score trends |
| 📋 History | Track all predictions with filters |

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - REST API framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Machine Learning
- **scikit-learn** - ML models and preprocessing
- **NLTK** - Text preprocessing
- **joblib** - Model serialization

### Frontend
- **Streamlit** - Interactive dashboard
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

---

## 📁 Project Structure
