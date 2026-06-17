"""
AI Business Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np
import time

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="AI Business Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DARK THEME CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background: #0a0e1a;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #e2e8f0;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(56, 189, 248, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        backdrop-filter: blur(20px);
        padding: 1.2rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.08);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.2);
        box-shadow: 0 8px 32px rgba(56, 189, 248, 0.08);
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    
    .score-high {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.2), rgba(52, 211, 153, 0.05));
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        display: inline-block;
        font-weight: 600;
        color: #34d399;
        font-size: 1.1rem;
    }
    .score-medium {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(251, 191, 36, 0.05));
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        display: inline-block;
        font-weight: 600;
        color: #fbbf24;
        font-size: 1.1rem;
    }
    .score-low {
        background: linear-gradient(135deg, rgba(251, 146, 60, 0.2), rgba(251, 146, 60, 0.05));
        border: 1px solid rgba(251, 146, 60, 0.3);
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        display: inline-block;
        font-weight: 600;
        color: #fb923c;
        font-size: 1.1rem;
    }
    
    .recommendation-item {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        backdrop-filter: blur(10px);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin: 0.4rem 0;
        border-left: 3px solid #38bdf8;
        color: #e2e8f0;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }
    .recommendation-item:hover {
        background: rgba(30, 41, 59, 0.8);
        border-left-color: #818cf8;
        transform: translateX(4px);
    }
    
    .css-1d391kg, .css-12oz5g7 {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(10, 14, 26, 0.98));
        border-right: 1px solid rgba(56, 189, 248, 0.06);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(56, 189, 248, 0.15);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.25);
    }
    
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.1) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(56, 189, 248, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: rgba(56, 189, 248, 0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.1));
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.15);
    }
    
    .stAlert {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(56, 189, 248, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px);
    }
    
    .footer {
        text-align: center;
        color: #475569;
        padding: 2rem 0 1rem 0;
        font-size: 0.85rem;
        border-top: 1px solid rgba(56, 189, 248, 0.06);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# ============================================
# HEADER - JUST APP NAME
# ============================================
st.markdown('<div class="main-header">⚡ AI Business Intelligence Agent</div>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url
    )
    st.session_state.api_url = api_url
    
    if st.button("🔌 Check Health", use_container_width=True):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API is healthy")
            else:
                st.error(f"❌ Status: {response.status_code}")
        except:
            st.error("❌ Connection failed")
    
    st.markdown("---")
    st.markdown("### 📊 Features")
    st.info(
        "• 🏢 21 Industry Classification\n"
        "• ⭐ Lead Quality Scoring\n"
        "• 💡 Sales Recommendations\n"
        "• 📊 Batch Processing"
    )
    st.markdown("---")
    st.caption(f"v2.0 | {datetime.now().strftime('%Y-%m-%d')}")

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Single Prediction",
    "📊 Batch Prediction",
    "📈 Analytics",
    "📋 History"
])

# ============================================
# TAB 1: SINGLE PREDICTION
# ============================================
with tab1:
    st.markdown('<div class="sub-header">🔍 Company Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_name = st.text_input(
            "Company Name *",
            placeholder="e.g., NeuralTech Solutions Inc."
        )
        description = st.text_area(
            "Company Description *",
            placeholder="Describe what the company does...",
            height=120
        )
    
    with col2:
        st.markdown("### 📋 Features")
        website_exists = st.selectbox(
            "Website Exists",
            [1, 0],
            format_func=lambda x: "✅ Yes" if x else "❌ No"
        )
        contact_form = st.selectbox(
            "Contact Form",
            [1, 0],
            format_func=lambda x: "✅ Yes" if x else "❌ No"
        )
        services_count = st.slider("Services Count", 1, 20, 5)
        country_score = st.select_slider(
            "Country Score",
            options=[1, 2, 3],
            value=2,
            format_func=lambda x: ["Low", "Medium", "High"][x-1]
        )
    
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if not company_name or not description:
            st.error("❌ Please fill Company Name and Description")
        else:
            with st.spinner("🔍 Analyzing..."):
                try:
                    payload = {
                        "company_name": company_name,
                        "description": description,
                        "features": {
                            "website_exists": website_exists,
                            "contact_form": contact_form,
                            "services_count": services_count,
                            "country_score": country_score
                        }
                    }
                    
                    response = requests.post(
                        f"{api_url}/predict",
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.session_state.history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'company': company_name,
                            'industry': result['predicted_industry'],
                            'score': result['lead_score']['score'],
                            'category': result['lead_score']['category']
                        })
                        
                        st.markdown("---")
                        st.markdown("### 📊 Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">🏢 Industry</div>
                                <div class="metric-value">{result['predicted_industry']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">🎯 Confidence</div>
                                <div class="metric-value">{result['confidence']:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            score = result['lead_score']['score']
                            cls = "score-high" if score >= 80 else "score-medium" if score >= 60 else "score-low"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">⭐ Lead Score</div>
                                <div class="metric-value"><span class="{cls}">{score:.1f}/100</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            category = result['lead_score']['category']
                            emoji = "🟢" if category == "High Quality" else "🟡" if category == "Medium Quality" else "🔴"
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">📊 Category</div>
                                <div class="metric-value">{emoji} {category}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("### 🎯 Top Predictions")
                        top_preds = pd.DataFrame(result['top_predictions'])
                        fig = px.bar(
                            top_preds,
                            x='industry',
                            y='probability',
                            color='probability',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            height=300,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#e2e8f0'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("### 💡 Recommendations")
                        for rec in result['recommendations']:
                            st.markdown(f'<div class="recommendation-item">📌 {rec}</div>', unsafe_allow_html=True)
                        
                        priority = result['lead_score']['priority']
                        icon = "🔴" if priority == "Urgent" else "🟡" if priority == "Follow-up" else "🔵"
                        st.info(f"**Priority:** {icon} {priority}")
                        
                    else:
                        st.error(f"❌ Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# ============================================
# TAB 2: BATCH PREDICTION
# ============================================
with tab2:
    st.markdown('<div class="sub-header">📊 Batch Prediction</div>', unsafe_allow_html=True)
    
    st.info("Upload CSV with columns: company_name, description, website_exists, contact_form, services_count, country_score")
    
    with st.expander("📄 Download Sample Template"):
        sample = pd.DataFrame({
            'company_name': ['Tech Corp', 'Health Inc', 'FinTech Ltd'],
            'description': [
                'AI company specializing in machine learning',
                'Healthcare provider with digital solutions',
                'Financial technology company'
            ],
            'website_exists': [1, 1, 0],
            'contact_form': [1, 0, 0],
            'services_count': [10, 5, 8],
            'country_score': [3, 2, 1]
        })
        st.dataframe(sample)
        csv = sample.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "sample.csv", "text/csv")
    
    uploaded = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded:
        df = pd.read_csv(uploaded)
        required = ['company_name', 'description', 'website_exists', 'contact_form', 'services_count', 'country_score']
        
        if all(col in df.columns for col in required):
            st.success(f"✅ Loaded {len(df)} companies")
            st.dataframe(df.head())
            
            if st.button("🚀 Run Batch", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} companies..."):
                    try:
                        batch = []
                        for _, row in df.iterrows():
                            batch.append({
                                "company_name": row['company_name'],
                                "description": row['description'],
                                "features": {
                                    "website_exists": int(row['website_exists']),
                                    "contact_form": int(row['contact_form']),
                                    "services_count": int(row['services_count']),
                                    "country_score": int(row['country_score'])
                                }
                            })
                        
                        response = requests.post(f"{api_url}/predict/batch", json=batch, timeout=60)
                        
                        if response.status_code == 200:
                            results = response.json()
                            df_results = pd.DataFrame(results['results'])
                            
                            for _, row in df_results.iterrows():
                                st.session_state.history.append({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'company': row['company_name'],
                                    'industry': row['predicted_industry'],
                                    'score': row['lead_score']['score'],
                                    'category': row['lead_score']['category']
                                })
                            
                            st.success(f"✅ Processed {len(df_results)} companies")
                            st.dataframe(df_results)
                            
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv"
                            )
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total", len(df_results))
                            with col2:
                                avg = df_results['lead_score'].apply(lambda x: x['score']).mean()
                                st.metric("Avg Score", f"{avg:.1f}")
                            with col3:
                                high = df_results['lead_score'].apply(lambda x: x['category']).value_counts().get('High Quality', 0)
                                st.metric("High Quality", high)
                        else:
                            st.error(f"❌ Error: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            st.error(f"❌ Missing columns")

# ============================================
# TAB 3: ANALYTICS
# ============================================
with tab3:
    st.markdown('<div class="sub-header">📈 Analytics</div>', unsafe_allow_html=True)
    
    if len(st.session_state.history) == 0:
        st.info("📊 No data yet. Start making predictions!")
        
        if st.button("📊 Generate Sample Data"):
            industries = ['AI', 'SaaS', 'Healthcare', 'Fintech', 'CleanTech', 'EdTech']
            for i in range(50):
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'company': f"Company {i+1}",
                    'industry': np.random.choice(industries),
                    'score': np.random.randint(40, 95),
                    'category': np.random.choice(['High Quality', 'Medium Quality', 'Low Quality'])
                })
            st.success("✅ Sample data generated!")
            st.rerun()
    else:
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total", len(df))
        with col2:
            st.metric("Avg Score", f"{df['score'].mean():.1f}")
        with col3:
            high = len(df[df['category'] == 'High Quality'])
            st.metric("High Quality", high)
        with col4:
            st.metric("Industries", df['industry'].nunique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, names='industry', title='Industry Distribution')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='score', title='Score Distribution')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 4: HISTORY
# ============================================
with tab4:
    st.markdown('<div class="sub-header">📋 History</div>', unsafe_allow_html=True)
    
    if len(st.session_state.history) == 0:
        st.info("📭 No predictions yet")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2 = st.columns(2)
        with col1:
            industry_filter = st.multiselect(
                "Filter Industry",
                options=sorted(df['industry'].unique()),
                default=[]
            )
        with col2:
            category_filter = st.multiselect(
                "Filter Category",
                options=['High Quality', 'Medium Quality', 'Low Quality'],
                default=[]
            )
        
        filtered = df.copy()
        if industry_filter:
            filtered = filtered[filtered['industry'].isin(industry_filter)]
        if category_filter:
            filtered = filtered[filtered['category'].isin(category_filter)]
        
        st.dataframe(filtered.sort_values('timestamp', ascending=False), use_container_width=True, height=400)
        
        if not filtered.empty:
            csv = filtered.to_csv(index=False)
            st.download_button("📥 Download History", csv, f"history_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.success("History cleared!")
            st.rerun()

# ============================================
# FOOTER
# ============================================
st.markdown("""
    <div class="footer">
        Built with ❤️ using FastAPI, Streamlit & Machine Learning
    </div>
""", unsafe_allow_html=True)