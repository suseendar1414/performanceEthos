import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

def calculate_detailed_kpi_scores(data):
    """
    Calculate individual scores for each KPI metric with proper datetime handling
    """
    kpi_scores = {}
    
    # Create a copy of the data to avoid modifying the original
    df = data.copy()
    
    # 1. Total Number of Loans Closed (Weight: 9)
    closed_loans = df['ISCLOSED'].sum()
    kpi_scores['Total_Loans_Closed'] = {
        'raw_value': closed_loans,
        'achievement_score': (df['ISCLOSED'].mean() * 100),
        'kpi_score': (df['ISCLOSED'].mean() * 9),
        'weight': 9,
        'description': 'Total number of loans that have been closed'
    }
    
    # 2. Total Dollar Value (Weight: 10)
    total_value = df['LOANTOTALREVENUE__C'].sum()
    max_possible = df['LOANTOTALREVENUE__C'].max() * len(df)
    kpi_scores['Total_Dollar_Value'] = {
        'raw_value': total_value,
        'achievement_score': (total_value / max_possible * 100) if max_possible > 0 else 0,
        'kpi_score': (total_value / max_possible * 10) if max_possible > 0 else 0,
        'weight': 10,
        'description': 'Total monetary value of all loans'
    }
    
    # 3. Loan Types (Weight: 6)
    if 'RECORDTYPEID' in df.columns:
        loan_types = df['RECORDTYPEID'].nunique()
        kpi_scores['Loan_Types'] = {
            'raw_value': loan_types,
            'achievement_score': (loan_types / 5 * 100),
            'kpi_score': (loan_types / 5 * 6),
            'weight': 6,
            'description': 'Diversity of loan types handled'
        }
    
    # 4. Average Loan Size (Weight: 8)
    avg_loan = df['LOANTOTALREVENUE__C'].mean()
    max_loan = df['LOANTOTALREVENUE__C'].max()
    kpi_scores['Average_Loan_Size'] = {
        'raw_value': avg_loan,
        'achievement_score': (avg_loan / max_loan * 100) if max_loan > 0 else 0,
        'kpi_score': (avg_loan / max_loan * 8) if max_loan > 0 else 0,
        'weight': 8,
        'description': 'Average size of loans processed'
    }
    
    # 5. Loan Approval Rate (Weight: 7)
    approval_rate = df['ISWON'].mean()
    kpi_scores['Loan_Approval_Rate'] = {
        'raw_value': approval_rate * 100,
        'achievement_score': approval_rate * 100,
        'kpi_score': approval_rate * 7,
        'weight': 7,
        'description': 'Percentage of loans approved'
    }
    
    # 6. Time to Close (Weight: 6)
    if 'CREATEDDATE' in df.columns and 'CLOSEDATE' in df.columns:
        try:
            # Convert to datetime and remove timezone information
            df['CREATEDDATE'] = pd.to_datetime(df['CREATEDDATE']).dt.tz_localize(None)
            df['CLOSEDATE'] = pd.to_datetime(df['CLOSEDATE']).dt.tz_localize(None)
            
            # Calculate only for closed loans
            closed_loans = df[df['ISCLOSED'] == True]
            if len(closed_loans) > 0:
                avg_days = (closed_loans['CLOSEDATE'] - closed_loans['CREATEDDATE']).dt.days.mean()
                time_score = max(0, (90 - avg_days) / 90) if not pd.isna(avg_days) else 0
            else:
                avg_days = 0
                time_score = 0
                
            kpi_scores['Time_to_Close'] = {
                'raw_value': avg_days,
                'achievement_score': time_score * 100,
                'kpi_score': time_score * 6,
                'weight': 6,
                'description': 'Average time taken to close loans'
            }
        except Exception as e:
            st.warning(f"Error calculating Time to Close: {str(e)}")
            avg_days = 0
            kpi_scores['Time_to_Close'] = {
                'raw_value': 0,
                'achievement_score': 0,
                'kpi_score': 0,
                'weight': 6,
                'description': 'Average time taken to close loans'
            }
    
    # 7. Conversion Rate (Weight: 6)
    conversion_rate = len(df[df['ISCLOSED']]) / len(df) if len(df) > 0 else 0
    kpi_scores['Conversion_Rate'] = {
        'raw_value': conversion_rate * 100,
        'achievement_score': conversion_rate * 100,
        'kpi_score': conversion_rate * 6,
        'weight': 6,
        'description': 'Rate of opportunity conversion'
    }
    
    # 8. Pipeline Management (Weight: 6)
    pipeline_score = df['HASOPENACTIVITY'].mean() - df['HASOVERDUETASK'].mean()
    kpi_scores['Pipeline_Management'] = {
        'raw_value': pipeline_score * 100,
        'achievement_score': ((pipeline_score + 1) / 2) * 100,
        'kpi_score': ((pipeline_score + 1) / 2) * 6,
        'weight': 6,
        'description': 'Effectiveness of pipeline management'
    }
    
    # Calculate totals
    total_weight = sum(kpi['weight'] for kpi in kpi_scores.values())
    total_score = sum(kpi['kpi_score'] for kpi in kpi_scores.values())
    overall_achievement = (total_score / total_weight * 100) if total_weight > 0 else 0
    
    summary = {
        'total_score': total_score,
        'total_possible': total_weight,
        'overall_achievement': overall_achievement,
        'number_of_kpis_measured': len(kpi_scores)
    }
    
    return {
        'detailed_scores': kpi_scores,
        'summary': summary
    }

def create_radar_chart(kpi_scores):
    """Create radar chart for KPI scores"""
    categories = list(kpi_scores['detailed_scores'].keys())
    values = [kpi_scores['detailed_scores'][cat]['achievement_score'] for cat in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Achievement Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False
    )
    
    return fig

def create_download_link(df):
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="kpi_scores.csv">Download Detailed Report</a>'

def main():
    st.set_page_config(page_title="Loan Officer KPI Dashboard", layout="wide")
    
    st.title("Loan Officer KPI Performance Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your opportunities CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Calculate KPI scores
        scores = calculate_detailed_kpi_scores(data)
        
        # Display summary metrics
        st.header("Performance Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Achievement", f"{scores['summary']['overall_achievement']:.1f}%")
        with col2:
            st.metric("Total KPI Score", f"{scores['summary']['total_score']:.1f}/{scores['summary']['total_possible']}")
        with col3:
            st.metric("KPIs Measured", scores['summary']['number_of_kpis_measured'])
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["KPI Scores", "Visualizations", "Detailed Analysis"])
        
        with tab1:
            st.header("Individual KPI Scores")
            
            # Create a DataFrame for display
            kpi_df = pd.DataFrame.from_dict(scores['detailed_scores'], orient='index')
            
            # Display each KPI with its scores
            for kpi, values in scores['detailed_scores'].items():
                with st.expander(f"{kpi} - Score: {values['kpi_score']:.2f}/{values['weight']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Achievement", f"{values['achievement_score']:.1f}%")
                        st.metric("Raw Value", f"{values['raw_value']:.2f}")
                    with col2:
                        st.write("Description:", values['description'])
                        st.progress(values['achievement_score'] / 100)
        
        with tab2:
            st.header("Performance Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                st.subheader("KPI Achievement Radar")
                radar_fig = create_radar_chart(scores)
                st.plotly_chart(radar_fig)
            
            with col2:
                # Bar chart of KPI scores
                st.subheader("KPI Scores vs Weights")
                kpi_scores_df = pd.DataFrame({
                    'KPI': list(scores['detailed_scores'].keys()),
                    'Score': [v['kpi_score'] for v in scores['detailed_scores'].values()],
                    'Weight': [v['weight'] for v in scores['detailed_scores'].values()]
                })
                fig = px.bar(kpi_scores_df, x='KPI', y=['Score', 'Weight'], barmode='group')
                st.plotly_chart(fig)
        
        with tab3:
            st.header("Detailed Analysis")
            
            # Time series analysis if dates are available
            if 'CREATEDDATE' in data.columns:
                st.subheader("Performance Over Time")
                data['CREATEDDATE'] = pd.to_datetime(data['CREATEDDATE'])
                monthly_data = data.set_index('CREATEDDATE').resample('M').agg({
                    'ISCLOSED': 'sum',
                    'LOANTOTALREVENUE__C': 'sum',
                    'ISWON': 'mean'
                })
                fig = px.line(monthly_data, title="Monthly Performance Trends")
                st.plotly_chart(fig)
            
            # Download link for detailed report
            st.markdown("### Download Detailed Report")
            detailed_report = pd.DataFrame.from_dict(scores['detailed_scores'], orient='index')
            st.markdown(create_download_link(detailed_report), unsafe_allow_html=True)

if __name__ == "__main__":
    main()