# 🤖 AI-Powered Business Intelligence Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)  
  
<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/e8624359-1cb0-4736-aba5-9669815f6377" />  
  
  
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

  ai-business-intelligence-agent/  
├── src/ # Source code  
│ ├── init.py  
│ ├── data_processor.py # Data cleaning and preprocessing  
│ ├── feature_extractor.py # TF-IDF feature extraction  
│ ├── model_trainer.py # Model training and evaluation  
│ └── predictor.py # Prediction and lead scoring  
│  
├── api/ # FastAPI application  
│ ├── init.py  
│ ├── main.py # API endpoints  
│ └── schemas.py # Pydantic schemas  
│  
├── dashboard/ # Streamlit dashboard  
│ └── app.py # Main dashboard application  
│  
├── models/ # Trained models  
│ ├── best_model.joblib  
│ ├── feature_extractor.joblib  
│ └── classes.json  
│  
├── data/ # Data files  
├── tests/ # Unit tests  
├── logs/ # Log files  
├── requirements.txt # Python dependencies  
├── train_model.py # Model training script  
├── streamlit_app.py # Streamlit entry point  
└── README.md # Project documentation  

  
---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip (Python package manager)

### Local Installation

#### Step 1: Clone Repository

```bash
git clone https://github.com/usman-official-ai/ai-business-intelligence-agent.git
cd ai-business-intelligence-agent

**Step 2: Create Virtual Environment**  

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate

**Step 3: Install Dependencies**

pip install -r requirements.txt

**Step 4: Train Models**

python train_model.py

**Step 5: Run the Application**

Terminal 1 - Run API:

bash
uvicorn api.main:app --reload --port 8000

Terminal 2 - Run Dashboard:

bash
streamlit run dashboard/app.py

'
📊 Model Performance
Model	Accuracy	F1 Score	CV Mean
Random Forest	89.00%	88.76%	88.00%
Logistic Regression	85.00%	84.32%	84.00%
Multinomial NB	82.00%	81.54%	81.00%
Best Model: Random Forest (89% accuracy)

🎯 Use Cases
1. Sales Teams
Prioritize leads

Save 80% research time

Close more deals

2. Marketing Teams
Target right industries

Create better campaigns

Higher ROI

3. Business Development
Find new opportunities

Identify high-value clients

Expand market reach

4. Entrepreneurs
Analyze competition

Find potential partners

Validate business ideas

5. Investors
Evaluate startups

Identify high-growth sectors

Better investment decisions

🧪 Testing
bash
# Run all tests
pytest tests/

# Test with coverage
pytest --cov=src tests/

author
   usman-official-ai

⭐ Star this repo if you like it!
      
