import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score 
import pickle
import base64
from openai import OpenAI

# Store API key in session state
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

def get_llm_insights(data, kpi_scores, achievement_scores, api_key):
    """Get insights from LLM about the performance metrics"""
    client = OpenAI(api_key=api_key)
    
    # Prepare performance summary
    summary = f"""
    Performance Metrics Summary:
    - Total Opportunities: {len(data)}
    - Average KPI Score: {kpi_scores.mean():.2f}
    - Average Achievement Score: {achievement_scores.mean():.2f}%
    - Won Deals: {data['ISWON'].sum()}
    - Revenue Generated: ${data['LOANTOTALREVENUE__C'].sum():,.2f}
    - Closed Deals: {data['ISCLOSED'].sum()}
    
    Top Performing Metrics:
    - Max KPI Score: {kpi_scores.max():.2f}
    - Max Achievement Score: {achievement_scores.max():.2f}%
    
    Areas of Concern:
    - Min KPI Score: {kpi_scores.min():.2f}
    - Min Achievement Score: {achievement_scores.min():.2f}%
    """
    
    prompt = f"""
    You are an expert Loan Officer Performance Analyst. Based on the following performance metrics, provide:
    1. Key insights about the performance
    2. Specific recommendations for improvement
    3. Areas of strength
    4. Areas needing attention
    5. Strategic action items
    
    {summary}
    
    Provide your analysis in a clear, actionable format. Be specific and data-driven in your recommendations.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def get_deal_analysis(row_data, api_key):
    """Get LLM analysis for a specific deal"""
    client = OpenAI(api_key=api_key)
    
    deal_summary = f"""
    Deal Metrics:
    - Revenue: ${row_data['LOANTOTALREVENUE__C']:,.2f}
    - Won: {'Yes' if row_data['ISWON'] else 'No'}
    - Closed: {'Yes' if row_data['ISCLOSED'] else 'No'}
    - Has Overdue Tasks: {'Yes' if row_data['HASOVERDUETASK'] else 'No'}
    - Has Open Activities: {'Yes' if row_data['HASOPENACTIVITY'] else 'No'}
    - Probability: {row_data['PROBABILITY']}%
    - KPI Score: {row_data['Predicted_KPI_Score']:.2f}
    - Achievement Score: {row_data['Predicted_Achievement_Score']:.2f}%
    """
    
    prompt = f"""
    As a Loan Officer Performance Expert, analyze this specific deal:
    {deal_summary}
    
    Provide:
    1. Deal strength assessment
    2. Risk factors
    3. Specific recommendations
    4. Next best actions
    
    Keep the analysis concise and actionable.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# [Previous functions remain the same: create_features_and_targets, train_model, predict_scores, create_download_link]
def create_features_and_targets(opportunities_df):
    """
    Create features and target variables from opportunities data for ML model
    """
    # Create copy to avoid modifying original
    df = opportunities_df.copy()
    
    # Create features DataFrame
    features_df = pd.DataFrame()
    
    # Binary features
    features_df['is_won'] = df['ISWON'].astype(int)
    features_df['is_closed'] = df['ISCLOSED'].astype(int)
    features_df['has_overdue'] = df['HASOVERDUETASK'].astype(int)
    features_df['has_activity'] = df['HASOPENACTIVITY'].astype(int)
    
    # Numeric features
    features_df['revenue'] = df['LOANTOTALREVENUE__C'].fillna(0)
    features_df['probability'] = df['PROBABILITY'].fillna(0)
    
    # Calculate KPI score
    kpi_scores = (
        df['ISWON'].astype(int) * 9 +  # Win score
        df['ISCLOSED'].astype(int) * 7 +  # Closure score
        (df['LOANTOTALREVENUE__C'].fillna(0) / df['LOANTOTALREVENUE__C'].max()) * 10 +  # Revenue score
        (df['HASOPENACTIVITY'].astype(int) - df['HASOVERDUETASK'].astype(int)) * 6  # Activity score
    )
    
    # Calculate achievement score (0-100%)
    max_possible_score = 32  # Sum of all weights
    achievement_scores = (kpi_scores / max_possible_score) * 100
    
    return features_df, kpi_scores, achievement_scores

def train_model(opportunities_df):
    """
    Train the KPI prediction model
    """
    print("Preparing features and targets...")
    X, kpi_scores, achievement_scores = create_features_and_targets(opportunities_df)
    
    print("Splitting data...")
    # Split data for KPI scores
    X_train, X_test, y_train_kpi, y_test_kpi = train_test_split(
        X, kpi_scores, test_size=0.2, random_state=42
    )
    
    print("Scaling features...")
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training KPI Score Model...")
    # Train KPI Score Model
    kpi_model = RandomForestRegressor(n_estimators=100, random_state=42)
    kpi_model.fit(X_train_scaled, y_train_kpi)
    
    print("Training Achievement Score Model...")
    # Train Achievement Score Model
    achievement_model = RandomForestRegressor(n_estimators=100, random_state=42)
    achievement_model.fit(X_train_scaled, achievement_scores[:len(y_train_kpi)])
    
    # Evaluate models
    kpi_pred = kpi_model.predict(X_test_scaled)
    achievement_pred = achievement_model.predict(X_test_scaled)
    
    print("\nModel Evaluation:")
    print("KPI Score R²:", r2_score(y_test_kpi, kpi_pred))
    print("Achievement Score RMSE:", np.sqrt(mean_squared_error(
        achievement_scores[len(y_train_kpi):], achievement_pred
    )))
    
    return kpi_model, achievement_model, scaler

def predict_scores(model_kpi, model_achievement, scaler, new_data):
    """
    Predict scores for new data
    """
    # Prepare features
    X, _, _ = create_features_and_targets(new_data)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    kpi_scores = model_kpi.predict(X_scaled)
    achievement_scores = model_achievement.predict(X_scaled)
    
    return kpi_scores, achievement_scores

def create_download_link(df):
    """
    Create a download link for DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'

def main():
    st.set_page_config(page_title="KPI Score Predictor", layout="wide")
    
    st.title("Loan Officer KPI Score Prediction Dashboard")
    
    # Initialize session state for LLM analysis
    if 'selected_deal' not in st.session_state:
        st.session_state.selected_deal = None
    
    # API Key input in sidebar
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        st.session_state.openai_api_key = api_key
    
    # File upload
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose your opportunities CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and process data
        data = pd.read_csv(uploaded_file)
        
        # Train model and get predictions
        with st.spinner("Training model and generating predictions..."):
            kpi_model, achievement_model, scaler = train_model(data)
            kpi_scores, achievement_scores = predict_scores(kpi_model, achievement_model, scaler, data)
            
            results_df = data.copy()
            results_df['Predicted_KPI_Score'] = kpi_scores
            results_df['Predicted_Achievement_Score'] = achievement_scores
        
        # Main dashboard sections
        tab1, tab2, tab3 = st.tabs(["Overview", "Deal Analysis", "AI Insights"])
        
        with tab1:
            st.header("Performance Dashboard")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Opportunities", len(data))
            col2.metric("Won Opportunities", data['ISWON'].sum())
            col3.metric("Closed Opportunities", data['ISCLOSED'].sum())
            
            # Add visualizations
            col1, col2 = st.columns(2)
            with col1:
                fig_kpi = px.histogram(
                    results_df, 
                    x='Predicted_KPI_Score',
                    title="KPI Score Distribution"
                )
                st.plotly_chart(fig_kpi)
            
            with col2:
                fig_achievement = px.histogram(
                    results_df, 
                    x='Predicted_Achievement_Score',
                    title="Achievement Score Distribution"
                )
                st.plotly_chart(fig_achievement)
        
        with tab2:
            st.header("Individual Deal Analysis")
            if st.session_state.openai_api_key:
                selected_deal = st.selectbox(
                    "Select a deal to analyze:",
                    options=results_df.index,
                    format_func=lambda x: f"Deal {x} - Revenue: ${results_df.loc[x, 'LOANTOTALREVENUE__C']:,.2f}"
                )
                
                if selected_deal is not None:
                    with st.spinner("Analyzing deal..."):
                        deal_analysis = get_deal_analysis(results_df.loc[selected_deal], st.session_state.openai_api_key)
                        st.markdown(deal_analysis)
            else:
                st.warning("Please enter your OpenAI API key in the sidebar to enable AI analysis.")
        
        with tab3:
            st.header("AI Performance Insights")
            if st.session_state.openai_api_key:
                if st.button("Generate Performance Analysis"):
                    with st.spinner("Analyzing performance data..."):
                        insights = get_llm_insights(data, kpi_scores, achievement_scores, st.session_state.openai_api_key)
                        st.markdown(insights)
                
                # Custom analysis
                st.subheader("Ask Custom Questions")
                user_question = st.text_input("What would you like to know about the performance data?")
                if user_question:
                    with st.spinner("Analyzing..."):
                        client = OpenAI(api_key=st.session_state.openai_api_key)
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "user", "content": f"""
                                Given the following performance metrics, answer this question: {user_question}
                                
                                Metrics:
                                - Average KPI Score: {kpi_scores.mean():.2f}
                                - Average Achievement Score: {achievement_scores.mean():.2f}%
                                - Total Opportunities: {len(data)}
                                - Won Deals: {data['ISWON'].sum()}
                                - Total Revenue: ${data['LOANTOTALREVENUE__C'].sum():,.2f}
                                """}
                            ],
                            temperature=0.7
                        )
                        st.markdown(response.choices[0].message.content)
            else:
                st.warning("Please enter your OpenAI API key in the sidebar to enable AI analysis.")
        
        # Download section
        st.header("Download Results")
        st.markdown(create_download_link(results_df), unsafe_allow_html=True)

if __name__ == "__main__":
    main()