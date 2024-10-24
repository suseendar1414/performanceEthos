import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import base64

def create_features_and_targets(df):
    """
    Create features and target variables from opportunities dataframe
    """
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
        df['ISWON'].astype(int) * 9 +
        df['ISCLOSED'].astype(int) * 7 +
        (df['LOANTOTALREVENUE__C'].fillna(0) / df['LOANTOTALREVENUE__C'].max()) * 10 +
        (df['HASOPENACTIVITY'].astype(int) - df['HASOVERDUETASK'].astype(int)) * 6
    )
    
    # Calculate achievement score
    max_possible_score = 32
    achievement_scores = (kpi_scores / max_possible_score) * 100
    
    return features_df, kpi_scores, achievement_scores

def train_model(data):
    """Train the KPI prediction model"""
    X, kpi_scores, achievement_scores = create_features_and_targets(data)
    
    X_train, X_test, y_train_kpi, y_test_kpi = train_test_split(
        X, kpi_scores, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    kpi_model = RandomForestRegressor(n_estimators=100, random_state=42)
    kpi_model.fit(X_train_scaled, y_train_kpi)
    
    achievement_model = RandomForestRegressor(n_estimators=100, random_state=42)
    achievement_model.fit(X_train_scaled, achievement_scores[:len(y_train_kpi)])
    
    return kpi_model, achievement_model, scaler

def predict_scores(model_kpi, model_achievement, scaler, data):
    """Predict scores for new data"""
    X, _, _ = create_features_and_targets(data)
    X_scaled = scaler.transform(X)
    
    kpi_scores = model_kpi.predict(X_scaled)
    achievement_scores = model_achievement.predict(X_scaled)
    
    return kpi_scores, achievement_scores

def create_download_link(df):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV File</a>'

def main():
    st.set_page_config(page_title="KPI Score Predictor", layout="wide")
    
    st.title("Loan Officer KPI Score Prediction Dashboard")
    
    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose your opportunities CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load and display raw data
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
        
        # Display data summary
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Opportunities", len(data))
        col2.metric("Won Opportunities", data['ISWON'].sum())
        col3.metric("Closed Opportunities", data['ISCLOSED'].sum())
        
        # Train model and make predictions
        with st.spinner("Training model and generating predictions..."):
            kpi_model, achievement_model, scaler = train_model(data)
            kpi_scores, achievement_scores = predict_scores(kpi_model, achievement_model, scaler, data)
            
            # Add predictions to dataframe
            results_df = data.copy()
            results_df['Predicted_KPI_Score'] = kpi_scores
            results_df['Predicted_Achievement_Score'] = achievement_scores
        
        # Display results
        st.header("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("KPI Score Distribution")
            fig_kpi = px.histogram(
                results_df, 
                x='Predicted_KPI_Score',
                title="Distribution of KPI Scores",
                nbins=20
            )
            st.plotly_chart(fig_kpi)
            
        with col2:
            st.subheader("Achievement Score Distribution")
            fig_achievement = px.histogram(
                results_df, 
                x='Predicted_Achievement_Score',
                title="Distribution of Achievement Scores",
                nbins=20
            )
            st.plotly_chart(fig_achievement)
        
        # Summary statistics
        st.header("Performance Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("KPI Score Statistics")
            st.write(pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Min', 'Max'],
                'Value': [
                    f"{kpi_scores.mean():.2f}",
                    f"{np.median(kpi_scores):.2f}",
                    f"{kpi_scores.min():.2f}",
                    f"{kpi_scores.max():.2f}"
                ]
            }))
        
        with col2:
            st.subheader("Achievement Score Statistics")
            st.write(pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Min', 'Max'],
                'Value': [
                    f"{achievement_scores.mean():.2f}%",
                    f"{np.median(achievement_scores):.2f}%",
                    f"{achievement_scores.min():.2f}%",
                    f"{achievement_scores.max():.2f}%"
                ]
            }))
        
        # Performance categories
        st.header("Performance Categories")
        achievement_categories = pd.cut(
            achievement_scores,
            bins=[0, 25, 50, 75, 100],
            labels=['Needs Improvement', 'Average', 'Good', 'Excellent']
        )
        category_counts = achievement_categories.value_counts()
        
        fig_categories = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Distribution of Performance Categories"
        )
        st.plotly_chart(fig_categories)
        
        # Download predictions
        st.header("Download Predictions")
        st.markdown(create_download_link(results_df), unsafe_allow_html=True)
        
        # Detailed results table
        st.header("Detailed Results")
        st.dataframe(results_df)

if __name__ == "__main__":
    main()