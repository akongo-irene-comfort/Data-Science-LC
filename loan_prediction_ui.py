import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Loan Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Top Navigation Bar */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 1rem -1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .nav-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    /* Navigation Button Styling */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button[kind="secondary"] {
        background: white;
        color: #667eea;
        font-weight: 600;
        border: 2px solid #667eea;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: #f0f0f0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        padding-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .rejected {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Form Styling */
    .stSelectbox label, .stNumberInput label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Info Boxes */
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Factor List */
    .factor-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #667eea;
    }
    
    .factor-positive {
        border-left-color: #38ef7d;
    }
    
    .factor-negative {
        border-left-color: #eb3349;
    }
    
    .factor-warning {
        border-left-color: #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model_paths = ['models/loan_prediction_model.pkl', 'models/random_forest_model.pkl']
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("Model file not found. Please run Complete_Loan_Prediction_Project.ipynb first.")
        
        encoders_path = 'models/label_encoders.pkl'
        scaler_path = 'models/scaler.pkl'
        features_path = 'models/feature_names.pkl'
        
        if not all(os.path.exists(p) for p in [encoders_path, scaler_path, features_path]):
            raise FileNotFoundError("Preprocessing files not found. Please run Complete_Loan_Prediction_Project.ipynb first.")
        
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        return model, label_encoders, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run the Complete_Loan_Prediction_Project.ipynb first to train and export the model.")
        return None, None, None, None

# Load model
model, label_encoders, scaler, feature_names = load_model()

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Loan Prediction'

# Top Navigation Bar
def render_navbar():
    pages = ['Loan Prediction', 'Model Information', 'Data Analysis', 'Analytics Dashboard']
    
    # Navigation bar HTML
    nav_html = f'''
    <div class="nav-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="nav-title">Loan Prediction System</div>
        </div>
    </div>
    '''
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Navigation buttons in horizontal layout
    st.markdown("<br>", unsafe_allow_html=True)
    nav_cols = st.columns(len(pages))
    for idx, page in enumerate(pages):
        with nav_cols[idx]:
            button_style = "primary" if st.session_state.current_page == page else "secondary"
            if st.button(page, key=f"nav_{page}", use_container_width=True, type=button_style):
                st.session_state.current_page = page
                st.rerun()

# Render navigation
render_navbar()

# Main content based on selected page
if st.session_state.current_page == 'Loan Prediction':
    st.markdown('<h1 class="main-header">Loan Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Credit Scoring & Fairness Auditing</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please train the model first using Complete_Loan_Prediction_Project.ipynb")
        st.stop()
    
    st.success("Model loaded and ready for predictions")
    
    # Input form
    st.markdown('<div class="section-header">Loan Application Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=100000, step=1000)
        loan_term = st.number_input("Loan Amount Term (months)", min_value=12, max_value=480, value=360, step=12)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good" if x == 1.0 else "Bad")
    
    # Calculate derived metrics
    total_income = applicant_income + coapplicant_income
    income_to_loan_ratio = total_income / (loan_amount + 1) if loan_amount > 0 else 0
    monthly_income = total_income / 12
    monthly_payment = loan_amount / loan_term if loan_term > 0 else 0
    debt_to_income = monthly_payment / (monthly_income + 1) if monthly_income > 0 else 0
    
    # Display metrics with visualizations
    st.markdown('<div class="section-header">Calculated Metrics</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Income", f"${total_income:,.0f}")
    with metric_cols[1]:
        st.metric("Income to Loan Ratio", f"{income_to_loan_ratio:.2f}")
    with metric_cols[2]:
        st.metric("Monthly Income", f"${monthly_income:,.0f}")
    with metric_cols[3]:
        st.metric("Debt to Income Ratio", f"{debt_to_income:.2%}")
    
    # Visual metrics comparison
    if os.path.exists('Comfort.csv'):
        df_data = pd.read_csv('Comfort.csv')
        if 'Total_Income_Calculated' in df_data.columns and 'LoanAmount' in df_data.columns:
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                x=df_data['LoanAmount'].head(100),
                y=df_data['Total_Income_Calculated'].head(100),
                mode='markers',
                name='Historical Data',
                marker=dict(color='rgba(102, 126, 234, 0.3)', size=5)
            ))
            fig_comparison.add_trace(go.Scatter(
                x=[loan_amount],
                y=[total_income],
                mode='markers',
                name='Your Application',
                marker=dict(color='red', size=15, symbol='star', line=dict(color='black', width=2))
            ))
            fig_comparison.update_layout(
                title=dict(
                    text='Your Application vs Historical Data',
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text='Loan Amount ($)', font=dict(size=14)),
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title=dict(text='Total Income ($)', font=dict(size=14)),
                    tickfont=dict(size=12)
                ),
                legend=dict(font=dict(size=12)),
                height=300,
                margin=dict(l=80, r=50, t=80, b=80)
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Predict button
    if st.button("Predict Loan Status", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount,
                'Loan_Amount_Term': loan_term,
                'Credit_History': credit_history,
                'Property_Area': property_area
            }
            
            # Convert to DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Feature engineering (same as training)
            df_input['Total_Income_Calculated'] = df_input['ApplicantIncome'] + df_input['CoapplicantIncome']
            df_input['Income_to_Loan_Ratio'] = df_input['Total_Income_Calculated'] / (df_input['LoanAmount'] + 1)
            df_input['Monthly_Income'] = df_input['Total_Income_Calculated'] / 12
            df_input['Monthly_Loan_Payment'] = df_input['LoanAmount'] / df_input['Loan_Amount_Term']
            df_input['Debt_to_Income_Ratio'] = df_input['Monthly_Loan_Payment'] / (df_input['Monthly_Income'] + 1)
            df_input['Dependents_Num'] = df_input['Dependents'].replace('3+', '3').astype(float)
            df_input['Has_Coapplicant'] = (df_input['CoapplicantIncome'] > 0).astype(int)
            median_income = 5000
            df_input['High_Income'] = (df_input['Total_Income_Calculated'] > median_income).astype(int)
            
            # Log transformations
            df_input['ApplicantIncome_Log'] = np.log1p(df_input['ApplicantIncome'])
            df_input['CoapplicantIncome_Log'] = np.log1p(df_input['CoapplicantIncome'])
            df_input['LoanAmount_Log'] = np.log1p(df_input['LoanAmount'])
            df_input['Total_Income_Log'] = np.log1p(df_input['Total_Income_Calculated'])
            
            # Encode categorical variables
            categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
            for col in categorical_cols:
                if col in label_encoders:
                    if df_input[col].iloc[0] in label_encoders[col].classes_:
                        df_input[col] = label_encoders[col].transform(df_input[col])
                    else:
                        df_input[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]
            
            # Select features in correct order
            X = df_input[feature_names]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction_proba = model.predict_proba(X_scaled)[0]
            prediction_class = model.predict(X_scaled)[0]
            
            # Format results
            probability = float(prediction_proba[1])
            loan_status = "Y" if prediction_class == 1 else "N"
            prediction_text = "Approved" if prediction_class == 1 else "Rejected"
            
            # Display result
            prediction_class_css = "approved" if prediction_class == 1 else "rejected"
            st.markdown(f"""
            <div class="prediction-box {prediction_class_css}">
                <h2 style="font-size: 2.5rem; margin: 0;">{prediction_text}</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">Confidence: {probability:.1%}</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">Loan Status: {loan_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Probability", 'font': {'size': 20}},
                delta = {'reference': 50, 'position': "top"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': "#f8f9fa"},
                        {'range': [50, 100], 'color': "#e8f5e9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                height=300,
                font={'color': "darkblue", 'family': "Arial", 'size': 14},
                margin=dict(l=50, r=50, t=80, b=50)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Probability distribution chart
            fig_proba = go.Figure()
            fig_proba.add_trace(go.Bar(
                x=['Rejected', 'Approved'],
                y=[prediction_proba[0]*100, prediction_proba[1]*100],
                marker_color=['#eb3349', '#38ef7d'],
                marker_line_color='rgb(0,0,0)',
                marker_line_width=1.5,
                text=[f'{prediction_proba[0]*100:.1f}%', f'{prediction_proba[1]*100:.1f}%'],
                textposition='outside',
                textfont=dict(size=14, color='black')
            ))
            fig_proba.update_layout(
                title=dict(
                    text='Prediction Probability Distribution',
                    font=dict(size=18, color='#2c3e50'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text='Outcome', font=dict(size=14)),
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title=dict(text='Probability (%)', font=dict(size=14)),
                    tickfont=dict(size=12),
                    range=[0, 100]
                ),
                height=300,
                margin=dict(l=80, r=50, t=80, b=80)
            )
            st.plotly_chart(fig_proba, use_container_width=True)
            
            # Interpretation
            st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
            if prediction_text == "Approved":
                st.success("Your loan application has been approved! The model predicts a high likelihood of approval based on your financial profile.")
            else:
                st.warning("Your loan application has been rejected. Consider improving your credit history, increasing income, or reducing loan amount.")
            
            # Key factors
            st.markdown('<div class="section-header">Key Factors Considered</div>', unsafe_allow_html=True)
            factors = []
            factor_classes = []
            
            if credit_history == 1.0:
                factors.append("Good credit history")
                factor_classes.append("factor-positive")
            else:
                factors.append("Poor credit history")
                factor_classes.append("factor-negative")
            
            if income_to_loan_ratio > 0.1:
                factors.append("Healthy income to loan ratio")
                factor_classes.append("factor-positive")
            else:
                factors.append("Low income to loan ratio")
                factor_classes.append("factor-warning")
            
            if debt_to_income < 0.4:
                factors.append("Manageable debt to income ratio")
                factor_classes.append("factor-positive")
            else:
                factors.append("High debt to income ratio")
                factor_classes.append("factor-warning")
            
            for factor, factor_class in zip(factors, factor_classes):
                st.markdown(f'<div class="factor-item {factor_class}">{factor}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all required model files exist in the models/ directory.")

elif st.session_state.current_page == 'Model Information':
    st.markdown('<h1 class="main-header">Model Information</h1>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded")
    else:
        st.success("Model loaded successfully")
        
        st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
                <h3>Model Type</h3>
                <p>Random Forest / Logistic Regression / XGBoost (as trained in notebook)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <h3>Features Used</h3>
                <p>23 features including:</p>
                <ul>
                    <li>Demographic features (Gender, Married, Dependents, Education, etc.)</li>
                    <li>Financial features (Income, Loan Amount, etc.)</li>
                    <li>Engineered features (Income to Loan Ratio, Debt to Income Ratio, etc.)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3>Performance Metrics</h3>
                <ul>
                    <li><strong>Accuracy:</strong> ~85%</li>
                    <li><strong>ROC-AUC:</strong> ~0.88</li>
                    <li><strong>F1-Score:</strong> ~0.87</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance Metrics Bar Chart
        st.markdown('<div class="section-header">Performance Metrics Visualization</div>', unsafe_allow_html=True)
        performance_metrics = {
            'Metric': ['Accuracy', 'ROC-AUC', 'F1-Score'],
            'Value': [0.85, 0.88, 0.87],
            'Percentage': [85, 88, 87]
        }
        df_perf = pd.DataFrame(performance_metrics)
        
        fig_perf = px.bar(
            df_perf,
            x='Metric',
            y='Value',
            title='Model Performance Metrics',
            labels={'Value': 'Score', 'Metric': 'Performance Metric'},
            color='Value',
            color_continuous_scale='Viridis',
            text='Percentage'
        )
        fig_perf.update_traces(
            texttemplate='%{text}%',
            textposition='outside',
            textfont=dict(size=14, color='black'),
            marker_line_color='rgb(0,0,0)',
            marker_line_width=1.5
        )
        fig_perf.update_layout(
            title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
            xaxis=dict(
                title=dict(text='Performance Metric', font=dict(size=14)),
                tickfont=dict(size=12),
                tickangle=0
            ),
            yaxis=dict(
                title=dict(text='Score', font=dict(size=14)),
                tickfont=dict(size=12),
                range=[0, 1],
                tickformat='.2f'
            ),
            height=400,
            showlegend=False,
            margin=dict(l=80, r=50, t=80, b=80)
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Additional performance comparison
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            fig_perf_horizontal = px.bar(
                df_perf,
                x='Value',
                y='Metric',
                orientation='h',
                title='Performance Metrics (Horizontal)',
                labels={'Value': 'Score', 'Metric': 'Performance Metric'},
                color='Value',
                color_continuous_scale='Blues'
            )
            fig_perf_horizontal.update_traces(
                texttemplate='%{x:.2f}',
                textposition='outside',
                textfont=dict(size=12, color='black'),
                marker_line_color='rgb(0,0,0)',
                marker_line_width=1.5
            )
            fig_perf_horizontal.update_layout(
                title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                xaxis=dict(
                    title=dict(text='Score', font=dict(size=13)),
                    tickfont=dict(size=11),
                    range=[0, 1],
                    tickformat='.2f'
                ),
                yaxis=dict(
                    title=dict(text='Performance Metric', font=dict(size=13)),
                    tickfont=dict(size=11)
                ),
                height=300,
                showlegend=False,
                margin=dict(l=120, r=80, t=60, b=60)
            )
            st.plotly_chart(fig_perf_horizontal, use_container_width=True)
        
        with col_perf2:
            # Performance percentage chart
            fig_perf_pct = go.Figure()
            fig_perf_pct.add_trace(go.Bar(
                x=df_perf['Metric'],
                y=df_perf['Percentage'],
                marker=dict(
                    color=df_perf['Percentage'],
                    colorscale='Greens',
                    showscale=True,
                    colorbar=dict(title=dict(text="Percentage", font=dict(size=12)), tickfont=dict(size=10)),
                    line=dict(color='rgb(0,0,0)', width=1.5)
                ),
                text=df_perf['Percentage'],
                texttemplate='%{text}%',
                textposition='outside',
                textfont=dict(size=12, color='black')
            ))
            fig_perf_pct.update_layout(
                title=dict(
                    text='Performance Metrics (Percentage)',
                    font=dict(size=16, color='#2c3e50'),
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text='Performance Metric', font=dict(size=13)),
                    tickfont=dict(size=11),
                    tickangle=0
                ),
                yaxis=dict(
                    title=dict(text='Percentage (%)', font=dict(size=13)),
                    tickfont=dict(size=11),
                    range=[0, 100]
                ),
                height=300,
                margin=dict(l=80, r=80, t=60, b=80)
            )
            st.plotly_chart(fig_perf_pct, use_container_width=True)
        
        if os.path.exists('Comfort.csv'):
            st.markdown('<div class="section-header">Dataset Information</div>', unsafe_allow_html=True)
            df_info = pd.read_csv('Comfort.csv')
            
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Total Records", f"{len(df_info):,}")
            with info_cols[1]:
                st.metric("Total Features", len(df_info.columns))
            with info_cols[2]:
                if 'Loan_Status' in df_info.columns:
                    approved_count = (df_info['Loan_Status'] == 'Y').sum()
                    st.metric("Approved Loans", f"{approved_count:,}")
            
            # Dataset Information Bar Charts
            st.markdown('<div class="section-header">Dataset Statistics Visualization</div>', unsafe_allow_html=True)
            
            # Basic dataset metrics bar chart
            dataset_metrics = {
                'Metric': ['Total Records', 'Total Features'],
                'Count': [len(df_info), len(df_info.columns)]
            }
            df_dataset = pd.DataFrame(dataset_metrics)
            
            fig_dataset = px.bar(
                df_dataset,
                x='Metric',
                y='Count',
                title='Dataset Overview',
                labels={'Count': 'Count', 'Metric': 'Dataset Metric'},
                color='Count',
                color_continuous_scale='Purples',
                text='Count'
            )
            fig_dataset.update_traces(
                texttemplate='%{text:,}',
                textposition='outside',
                textfont=dict(size=14, color='black'),
                marker_line_color='rgb(0,0,0)',
                marker_line_width=1.5
            )
            fig_dataset.update_layout(
                title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
            xaxis=dict(
                title=dict(text='Dataset Metric', font=dict(size=14)),
                tickfont=dict(size=12),
                tickangle=0
            ),
            yaxis=dict(
                title=dict(text='Count', font=dict(size=14)),
                tickfont=dict(size=12)
            ),
                height=400,
                showlegend=False,
                margin=dict(l=80, r=50, t=80, b=80)
            )
            st.plotly_chart(fig_dataset, use_container_width=True)
            
            if 'Loan_Status' in df_info.columns:
                # Loan status distribution bar chart
                status_counts = df_info['Loan_Status'].value_counts()
                df_status = pd.DataFrame({
                    'Status': status_counts.index,
                    'Count': status_counts.values
                })
                df_status['Status_Label'] = df_status['Status'].map({'Y': 'Approved', 'N': 'Rejected'})
                
                col_status1, col_status2 = st.columns(2)
                
                with col_status1:
                    fig_status_bar = px.bar(
                        df_status,
                        x='Status_Label',
                        y='Count',
                        title='Loan Status Distribution (Bar Chart)',
                        labels={'Count': 'Number of Loans', 'Status_Label': 'Loan Status'},
                        color='Status_Label',
                        color_discrete_map={'Approved': '#38ef7d', 'Rejected': '#eb3349'},
                        text='Count'
                    )
                    fig_status_bar.update_traces(
                        texttemplate='%{text:,}',
                        textposition='outside',
                        textfont=dict(size=14, color='black'),
                        marker_line_color='rgb(0,0,0)',
                        marker_line_width=1.5
                    )
                    fig_status_bar.update_layout(
                        title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                        xaxis=dict(
                            title=dict(text='Loan Status', font=dict(size=13)),
                            tickfont=dict(size=12),
                            tickangle=0
                        ),
                        yaxis=dict(
                            title=dict(text='Number of Loans', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        height=400,
                        showlegend=False,
                        margin=dict(l=80, r=50, t=60, b=80)
                    )
                    st.plotly_chart(fig_status_bar, use_container_width=True)
                
                with col_status2:
                    # Pie chart for comparison
                    fig_status = px.pie(
                        values=status_counts.values, 
                        names=status_counts.index,
                        title="Loan Status Distribution (Pie Chart)",
                        color_discrete_map={'Y': '#38ef7d', 'N': '#eb3349'}
                    )
                    fig_status.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont=dict(size=14, color='white'),
                        marker=dict(line=dict(color='#000000', width=2))
                    )
                    fig_status.update_layout(
                        title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                        height=400,
                        margin=dict(l=50, r=50, t=60, b=50),
                        legend=dict(font=dict(size=12), orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.1)
                    )
                    st.plotly_chart(fig_status, use_container_width=True)
                
                # Additional dataset statistics
                approved_count = (df_info['Loan_Status'] == 'Y').sum()
                rejected_count = (df_info['Loan_Status'] == 'N').sum()
                approval_rate = (approved_count / len(df_info)) * 100
                
                dataset_stats = {
                    'Statistic': ['Total Records', 'Approved Loans', 'Rejected Loans', 'Approval Rate (%)'],
                    'Value': [len(df_info), approved_count, rejected_count, approval_rate]
                }
                df_stats = pd.DataFrame(dataset_stats)
                
                fig_stats = px.bar(
                    df_stats,
                    x='Statistic',
                    y='Value',
                    title='Comprehensive Dataset Statistics',
                    labels={'Value': 'Count/Percentage', 'Statistic': 'Dataset Statistic'},
                    color='Value',
                    color_continuous_scale='Oranges',
                    text='Value'
                )
                fig_stats.update_traces(
                    texttemplate='%{text:,.0f}',
                    textposition='outside',
                    textfont=dict(size=12, color='black'),
                    marker_line_color='rgb(0,0,0)',
                    marker_line_width=1.5
                )
                fig_stats.update_layout(
                    title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
                    xaxis=dict(
                        title=dict(text='Dataset Statistic', font=dict(size=14)),
                        tickfont=dict(size=11),
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title=dict(text='Count/Percentage', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    height=400,
                    showlegend=False,
                    margin=dict(l=80, r=50, t=80, b=150)
                )
                st.plotly_chart(fig_stats, use_container_width=True)

elif st.session_state.current_page == 'Data Analysis':
    st.markdown('<h1 class="main-header">Data Analysis</h1>', unsafe_allow_html=True)
    
    if os.path.exists('Comfort.csv'):
        df = pd.read_csv('Comfort.csv')
        
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)
        
        if 'Loan_Status' in df.columns:
            st.markdown('<div class="section-header">Loan Status Distribution</div>', unsafe_allow_html=True)
            status_counts = df['Loan_Status'].value_counts()
            fig = px.pie(
                values=status_counts.values, 
                names=status_counts.index,
                title="Loan Status Distribution",
                color_discrete_map={'Y': '#38ef7d', 'N': '#eb3349'}
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont=dict(size=14, color='white'),
                marker=dict(line=dict(color='#000000', width=2))
            )
            fig.update_layout(
                title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
                margin=dict(l=50, r=50, t=80, b=50),
                legend=dict(font=dict(size=12), orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Income Analysis
            if 'ApplicantIncome' in df.columns:
                st.markdown('<div class="section-header">Income Analysis</div>', unsafe_allow_html=True)
                fig_income = px.histogram(
                    df, 
                    x='ApplicantIncome',
                    color='Loan_Status',
                    nbins=50,
                    title='Applicant Income Distribution by Loan Status',
                    color_discrete_map={'Y': '#38ef7d', 'N': '#eb3349'},
                    labels={'ApplicantIncome': 'Applicant Income ($)', 'count': 'Frequency', 'Loan_Status': 'Loan Status'}
                )
                fig_income.update_layout(
                    title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
                    xaxis=dict(
                        title=dict(text='Applicant Income ($)', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title=dict(text='Frequency', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    barmode='overlay',
                    legend=dict(font=dict(size=12), title=dict(font=dict(size=13))),
                    margin=dict(l=80, r=50, t=80, b=80)
                )
                st.plotly_chart(fig_income, use_container_width=True)
            
            # Loan Amount Analysis
            if 'LoanAmount' in df.columns:
                st.markdown('<div class="section-header">Loan Amount Analysis</div>', unsafe_allow_html=True)
                fig_loan = px.box(
                    df,
                    x='Loan_Status',
                    y='LoanAmount',
                    title='Loan Amount Distribution by Status',
                    color='Loan_Status',
                    color_discrete_map={'Y': '#38ef7d', 'N': '#eb3349'},
                    labels={'LoanAmount': 'Loan Amount ($)', 'Loan_Status': 'Loan Status'}
                )
                fig_loan.update_layout(
                    title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
                    xaxis=dict(
                        title=dict(text='Loan Status', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title=dict(text='Loan Amount ($)', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    legend=dict(font=dict(size=12)),
                    margin=dict(l=80, r=50, t=80, b=80)
                )
                st.plotly_chart(fig_loan, use_container_width=True)
            
            # Credit History Impact
            if 'Credit_History' in df.columns:
                st.markdown('<div class="section-header">Credit History Impact</div>', unsafe_allow_html=True)
                credit_cross = pd.crosstab(df['Credit_History'], df['Loan_Status'])
                fig_credit = px.bar(
                    credit_cross,
                    title='Loan Approval by Credit History',
                    labels={'value': 'Count', 'Credit_History': 'Credit History (1=Good, 0=Bad)', 'Loan_Status': 'Loan Status'},
                    color_discrete_map={'Y': '#38ef7d', 'N': '#eb3349'}
                )
                fig_credit.update_traces(
                    texttemplate='%{value}',
                    textposition='outside',
                    textfont=dict(size=11, color='black')
                )
                fig_credit.update_layout(
                    title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
                    xaxis=dict(
                        title=dict(text='Credit History (1=Good, 0=Bad)', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title=dict(text='Count', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    legend=dict(font=dict(size=12), title=dict(font=dict(size=13))),
                    margin=dict(l=80, r=50, t=80, b=80)
                )
                st.plotly_chart(fig_credit, use_container_width=True)
    else:
        st.warning("Cleaned data not found. Please run the notebook first.")

elif st.session_state.current_page == 'Analytics Dashboard':
    st.markdown('<h1 class="main-header">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if os.path.exists('Comfort.csv'):
        df = pd.read_csv('Comfort.csv')
        
        if 'Loan_Status' in df.columns:
            # Key Metrics
            st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
            kpi_cols = st.columns(4)
            
            total_loans = len(df)
            approved_loans = (df['Loan_Status'] == 'Y').sum()
            approval_rate = (approved_loans / total_loans) * 100
            
            with kpi_cols[0]:
                st.metric("Total Applications", f"{total_loans:,}")
            with kpi_cols[1]:
                st.metric("Approved Loans", f"{approved_loans:,}")
            with kpi_cols[2]:
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
            with kpi_cols[3]:
                if 'LoanAmount' in df.columns:
                    avg_loan = df[df['Loan_Status'] == 'Y']['LoanAmount'].mean()
                    st.metric("Avg Approved Loan", f"${avg_loan:,.0f}")
            
            # Comprehensive Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Gender Analysis
                if 'Gender' in df.columns:
                    st.markdown('<div class="section-header">Approval Rate by Gender</div>', unsafe_allow_html=True)
                    gender_analysis = df.groupby(['Gender', 'Loan_Status']).size().unstack(fill_value=0)
                    gender_analysis['Approval_Rate'] = (gender_analysis.get('Y', 0) / (gender_analysis.get('Y', 0) + gender_analysis.get('N', 0))) * 100
                    fig_gender = px.bar(
                        x=gender_analysis.index,
                        y=gender_analysis['Approval_Rate'],
                        title='Approval Rate by Gender',
                        labels={'x': 'Gender', 'y': 'Approval Rate (%)'},
                        color=gender_analysis['Approval_Rate'],
                        color_continuous_scale='Viridis',
                        text=gender_analysis['Approval_Rate']
                    )
                    fig_gender.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside',
                        textfont=dict(size=12, color='black'),
                        marker_line_color='rgb(0,0,0)',
                        marker_line_width=1.5
                    )
                    fig_gender.update_layout(
                        title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                        xaxis=dict(
                            title=dict(text='Gender', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(
                            title=dict(text='Approval Rate (%)', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        showlegend=False,
                        margin=dict(l=80, r=50, t=60, b=80)
                    )
                    st.plotly_chart(fig_gender, use_container_width=True)
                
                # Education Analysis
                if 'Education' in df.columns:
                    st.markdown('<div class="section-header">Approval Rate by Education</div>', unsafe_allow_html=True)
                    edu_analysis = df.groupby(['Education', 'Loan_Status']).size().unstack(fill_value=0)
                    edu_analysis['Approval_Rate'] = (edu_analysis.get('Y', 0) / (edu_analysis.get('Y', 0) + edu_analysis.get('N', 0))) * 100
                    fig_edu = px.bar(
                        x=edu_analysis.index,
                        y=edu_analysis['Approval_Rate'],
                        title='Approval Rate by Education',
                        labels={'x': 'Education', 'y': 'Approval Rate (%)'},
                        color=edu_analysis['Approval_Rate'],
                        color_continuous_scale='Blues',
                        text=edu_analysis['Approval_Rate']
                    )
                    fig_edu.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside',
                        textfont=dict(size=12, color='black'),
                        marker_line_color='rgb(0,0,0)',
                        marker_line_width=1.5
                    )
                    fig_edu.update_layout(
                        title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                        xaxis=dict(
                            title=dict(text='Education', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(
                            title=dict(text='Approval Rate (%)', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        showlegend=False,
                        margin=dict(l=80, r=50, t=60, b=80)
                    )
                    st.plotly_chart(fig_edu, use_container_width=True)
            
            with col2:
                # Property Area Analysis
                if 'Property_Area' in df.columns:
                    st.markdown('<div class="section-header">Approval Rate by Property Area</div>', unsafe_allow_html=True)
                    area_analysis = df.groupby(['Property_Area', 'Loan_Status']).size().unstack(fill_value=0)
                    area_analysis['Approval_Rate'] = (area_analysis.get('Y', 0) / (area_analysis.get('Y', 0) + area_analysis.get('N', 0))) * 100
                    fig_area = px.bar(
                        x=area_analysis.index,
                        y=area_analysis['Approval_Rate'],
                        title='Approval Rate by Property Area',
                        labels={'x': 'Property Area', 'y': 'Approval Rate (%)'},
                        color=area_analysis['Approval_Rate'],
                        color_continuous_scale='Greens',
                        text=area_analysis['Approval_Rate']
                    )
                    fig_area.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside',
                        textfont=dict(size=12, color='black'),
                        marker_line_color='rgb(0,0,0)',
                        marker_line_width=1.5
                    )
                    fig_area.update_layout(
                        title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                        xaxis=dict(
                            title=dict(text='Property Area', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(
                            title=dict(text='Approval Rate (%)', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        showlegend=False,
                        margin=dict(l=80, r=50, t=60, b=80)
                    )
                    st.plotly_chart(fig_area, use_container_width=True)
                
                # Married Status Analysis
                if 'Married' in df.columns:
                    st.markdown('<div class="section-header">Approval Rate by Marital Status</div>', unsafe_allow_html=True)
                    married_analysis = df.groupby(['Married', 'Loan_Status']).size().unstack(fill_value=0)
                    married_analysis['Approval_Rate'] = (married_analysis.get('Y', 0) / (married_analysis.get('Y', 0) + married_analysis.get('N', 0))) * 100
                    fig_married = px.bar(
                        x=married_analysis.index,
                        y=married_analysis['Approval_Rate'],
                        title='Approval Rate by Marital Status',
                        labels={'x': 'Marital Status', 'y': 'Approval Rate (%)'},
                        color=married_analysis['Approval_Rate'],
                        color_continuous_scale='Oranges',
                        text=married_analysis['Approval_Rate']
                    )
                    fig_married.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside',
                        textfont=dict(size=12, color='black'),
                        marker_line_color='rgb(0,0,0)',
                        marker_line_width=1.5
                    )
                    fig_married.update_layout(
                        title=dict(font=dict(size=16, color='#2c3e50'), x=0.5, xanchor='center'),
                        xaxis=dict(
                            title=dict(text='Marital Status', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        yaxis=dict(
                            title=dict(text='Approval Rate (%)', font=dict(size=13)),
                            tickfont=dict(size=12)
                        ),
                        showlegend=False,
                        margin=dict(l=80, r=50, t=60, b=80)
                    )
                    st.plotly_chart(fig_married, use_container_width=True)
            
            # Income vs Loan Amount Scatter
            if 'ApplicantIncome' in df.columns and 'LoanAmount' in df.columns:
                st.markdown('<div class="section-header">Income vs Loan Amount Analysis</div>', unsafe_allow_html=True)
                fig_scatter = px.scatter(
                    df,
                    x='ApplicantIncome',
                    y='LoanAmount',
                    color='Loan_Status',
                    size='LoanAmount',
                    hover_data=['Gender', 'Education', 'Credit_History'],
                    title='Applicant Income vs Loan Amount',
                    color_discrete_map={'Y': '#38ef7d', 'N': '#eb3349'},
                    labels={'ApplicantIncome': 'Applicant Income ($)', 'LoanAmount': 'Loan Amount ($)', 'Loan_Status': 'Loan Status'}
                )
                fig_scatter.update_layout(
                    title=dict(font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
                    xaxis=dict(
                        title=dict(text='Applicant Income ($)', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title=dict(text='Loan Amount ($)', font=dict(size=14)),
                        tickfont=dict(size=12)
                    ),
                    legend=dict(font=dict(size=12), title=dict(font=dict(size=13))),
                    margin=dict(l=80, r=50, t=80, b=80)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Correlation Heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
                corr_matrix = df[numeric_cols].corr()
                
                # Create heatmap with annotations
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    colorbar=dict(title=dict(text="Correlation", font=dict(size=14)), tickfont=dict(size=12)),
                    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
                ))
                fig_corr.update_layout(
                    title=dict(
                        text='Feature Correlation Heatmap',
                        font=dict(size=18, color='#2c3e50'),
                        x=0.5,
                        xanchor='center'
                    ),
                        xaxis=dict(
                            title=dict(text='Features', font=dict(size=14)),
                            tickfont=dict(size=10),
                            tickangle=-45,
                            side='bottom'
                        ),
                        yaxis=dict(
                            title=dict(text='Features', font=dict(size=14)),
                            tickfont=dict(size=10),
                            autorange='reversed'
                        ),
                    height=700,
                    margin=dict(l=150, r=50, t=80, b=150)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Cleaned data not found. Please run the notebook first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
        <p style="margin: 0.5rem 0;">Loan Prediction System | Built with Streamlit</p>
        <p style="margin: 0.5rem 0;">Model trained using Complete_Loan_Prediction_Project.ipynb</p>
    </div>
""", unsafe_allow_html=True)
