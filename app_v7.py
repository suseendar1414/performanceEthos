import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from openai import OpenAI
import time


def calculate_detailed_kpi_scores(data):
    """
    Calculate individual scores for each KPI metric with fixed scoring logic
    """
    kpi_scores = {}
    
    # 1. Total Number of Loans Closed (Weight: 9)
    closed_loans = data['ISCLOSED'].sum()
    # Set benchmark targets
    target_loans = 1000  # Example target
    max_loans = 2000     # Example maximum
    
    # Calculate achievement score using a logarithmic scale
    achievement_score = min(100, (closed_loans / target_loans) * 100)
    kpi_scores['Total_Loans_Closed'] = {
        'raw_value': closed_loans,
        'achievement_score': achievement_score,
        'kpi_score': (achievement_score / 100) * 9,
        'weight': 9,
        'description': 'Total number of loans that have been closed'
    }
    
    # 2. Total Dollar Value (Weight: 10)
    total_value = data['LOANTOTALREVENUE__C'].sum()
    # Set revenue targets
    target_revenue = 10000000  # Example target
    max_revenue = 20000000     # Example maximum
    
    revenue_achievement = min(100, (total_value / target_revenue) * 100)
    kpi_scores['Total_Dollar_Value'] = {
        'raw_value': total_value,
        'achievement_score': revenue_achievement,
        'kpi_score': (revenue_achievement / 100) * 10,
        'weight': 10,
        'description': 'Total monetary value of all loans'
    }
    
    # 3. Loan Types (Weight: 6)
    if 'RECORDTYPEID' in data.columns:
        loan_types = data['RECORDTYPEID'].nunique()
        target_types = 5  # Example target
        
        type_achievement = min(100, (loan_types / target_types) * 100)
        kpi_scores['Loan_Types'] = {
            'raw_value': loan_types,
            'achievement_score': type_achievement,
            'kpi_score': (type_achievement / 100) * 6,
            'weight': 6,
            'description': 'Diversity of loan types handled'
        }
    
    # 4. Average Loan Size (Weight: 8)
    avg_loan = data['LOANTOTALREVENUE__C'].mean()
    target_avg = 100000  # Example target
    
    size_achievement = min(100, (avg_loan / target_avg) * 100)
    kpi_scores['Average_Loan_Size'] = {
        'raw_value': avg_loan,
        'achievement_score': size_achievement,
        'kpi_score': (size_achievement / 100) * 8,
        'weight': 8,
        'description': 'Average size of loans processed'
    }
    
    # 5. Loan Approval Rate (Weight: 7)
    approval_rate = data['ISWON'].mean() * 100  # Convert to percentage
    target_approval = 80  # Example target percentage
    
    rate_achievement = min(100, (approval_rate / target_approval) * 100)
    kpi_scores['Loan_Approval_Rate'] = {
        'raw_value': approval_rate,
        'achievement_score': rate_achievement,
        'kpi_score': (rate_achievement / 100) * 7,
        'weight': 7,
        'description': 'Percentage of loans approved'
    }
    
    # 6. Time to Close (Weight: 6)
    if 'CREATEDDATE' in data.columns and 'CLOSEDATE' in data.columns:
        try:
            data['CREATEDDATE'] = pd.to_datetime(data['CREATEDDATE']).dt.tz_localize(None)
            data['CLOSEDATE'] = pd.to_datetime(data['CLOSEDATE']).dt.tz_localize(None)
            
            closed_loans = data[data['ISCLOSED'] == True]
            if len(closed_loans) > 0:
                avg_days = (closed_loans['CLOSEDATE'] - closed_loans['CREATEDDATE']).dt.days.mean()
                target_days = 30  # Example target
                max_days = 90     # Example maximum
                
                # Inverse calculation as lower days is better
                time_achievement = max(0, min(100, ((max_days - avg_days) / (max_days - target_days)) * 100))
            else:
                avg_days = 0
                time_achievement = 0
            
            kpi_scores['Time_to_Close'] = {
                'raw_value': avg_days,
                'achievement_score': time_achievement,
                'kpi_score': (time_achievement / 100) * 6,
                'weight': 6,
                'description': 'Average time taken to close loans'
            }
        except Exception as e:
            st.warning(f"Error calculating Time to Close: {str(e)}")
    
    # 7. Conversion Rate (Weight: 6)
    conversion_rate = (data['ISCLOSED'] & data['ISWON']).mean() * 100  # Convert to percentage
    target_conversion = 70  # Example target percentage
    
    conv_achievement = min(100, (conversion_rate / target_conversion) * 100)
    kpi_scores['Conversion_Rate'] = {
        'raw_value': conversion_rate,
        'achievement_score': conv_achievement,
        'kpi_score': (conv_achievement / 100) * 6,
        'weight': 6,
        'description': 'Rate of opportunity conversion'
    }
    
    # 8. Pipeline Management (Weight: 6)
    pipeline_score = data['HASOPENACTIVITY'].mean() * 100 - data['HASOVERDUETASK'].mean() * 50
    target_pipeline = 80  # Example target
    
    pipeline_achievement = min(100, max(0, (pipeline_score / target_pipeline) * 100))
    kpi_scores['Pipeline_Management'] = {
        'raw_value': pipeline_score,
        'achievement_score': pipeline_achievement,
        'kpi_score': (pipeline_achievement / 100) * 6,
        'weight': 6,
        'description': 'Effectiveness of pipeline management'
    }
    
    # Calculate totals
    total_weight = sum(kpi['weight'] for kpi in kpi_scores.values())
    total_score = sum(kpi['kpi_score'] for kpi in kpi_scores.values())
    overall_achievement = (total_score / total_weight * 100) if total_weight > 0 else 0
    
    return {
        'detailed_scores': kpi_scores,
        'summary': {
            'total_score': total_score,
            'total_possible': total_weight,
            'overall_achievement': overall_achievement,
            'number_of_kpis_measured': len(kpi_scores)
        }
    }

# Add these visualization functions before your main function
def create_radar_chart(scores):
    """Create radar chart for KPI scores"""
    # Extract data for the radar chart
    categories = list(scores['detailed_scores'].keys())
    values = [scores['detailed_scores'][cat]['achievement_score'] for cat in categories]
    
    # Create the radar chart
    fig = go.Figure()
    
    # Add the trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Achievement Score'
    ))
    
    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="KPI Achievement Radar",
        width=600,
        height=500
    )
    
    return fig

def create_bar_chart(scores):
    """Create bar chart comparing KPI scores and weights"""
    # Prepare data
    kpi_scores_df = pd.DataFrame({
        'KPI': list(scores['detailed_scores'].keys()),
        'Score': [v['kpi_score'] for v in scores['detailed_scores'].values()],
        'Weight': [v['weight'] for v in scores['detailed_scores'].values()]
    })
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars for scores
    fig.add_trace(go.Bar(
        name='Score',
        x=kpi_scores_df['KPI'],
        y=kpi_scores_df['Score'],
        marker_color='rgb(55, 83, 109)'
    ))
    
    # Add bars for weights
    fig.add_trace(go.Bar(
        name='Weight',
        x=kpi_scores_df['KPI'],
        y=kpi_scores_df['Weight'],
        marker_color='rgb(26, 118, 255)'
    ))
    
    # Update layout
    fig.update_layout(
        title='KPI Scores vs Weights',
        xaxis_tickangle=-45,
        barmode='group',
        width=800,
        height=500
    )
    
    return fig

def create_achievement_distribution(scores):
    """Create histogram of achievement scores"""
    achievement_scores = [v['achievement_score'] for v in scores['detailed_scores'].values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=achievement_scores,
        nbinsx=10,
        name='Achievement Scores',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.update_layout(
        title='Distribution of Achievement Scores',
        xaxis_title='Achievement Score',
        yaxis_title='Frequency',
        width=600,
        height=400
    )
    
    return fig

def create_performance_summary(scores):
    """Create summary metrics visualization"""
    fig = go.Figure()
    
    # Add a gauge chart for overall achievement
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=scores['summary']['overall_achievement'],
        title={'text': "Overall Achievement"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 33], 'color': "lightgray"},
                   {'range': [33, 66], 'color': "gray"},
                   {'range': [66, 100], 'color': "darkgray"}
               ]
              }
    ))
    
    fig.update_layout(
        width=400,
        height=300
    )
    
    return fig


def create_achievement_distribution(scores):
    """Create histogram of achievement scores"""
    achievement_scores = [v['achievement_score'] for v in scores['detailed_scores'].values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=achievement_scores,
        nbinsx=10,
        name='Achievement Scores',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.update_layout(
        title='Distribution of Achievement Scores',
        xaxis_title='Achievement Score',
        yaxis_title='Frequency',
        width=600,
        height=400
    )
    
    return fig

def create_performance_summary(scores):
    """Create summary metrics visualization"""
    fig = go.Figure()
    
    # Add a gauge chart for overall achievement
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=scores['summary']['overall_achievement'],
        title={'text': "Overall Achievement"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 33], 'color': "lightgray"},
                   {'range': [33, 66], 'color': "gray"},
                   {'range': [66, 100], 'color': "darkgray"}
               ]
              }
    ))
    
    fig.update_layout(
        width=400,
        height=300
    )
    
    return fig


def create_time_trend_analysis(data):
    """Create time trend analysis of key metrics"""
    try:
        # Convert dates and create monthly metrics
        data['CREATEDDATE'] = pd.to_datetime(data['CREATEDDATE'])
        monthly_data = data.groupby(data['CREATEDDATE'].dt.to_period('M')).agg({
            'ISCLOSED': 'sum',
            'LOANTOTALREVENUE__C': 'sum',
            'ISWON': 'mean'
        }).reset_index()
        
        # Convert period to datetime for plotting
        monthly_data['CREATEDDATE'] = monthly_data['CREATEDDATE'].astype(str)
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(
            go.Bar(name='Closed Loans', x=monthly_data['CREATEDDATE'], 
                  y=monthly_data['ISCLOSED'], yaxis='y')
        )
        fig.add_trace(
            go.Line(name='Revenue', x=monthly_data['CREATEDDATE'], 
                   y=monthly_data['LOANTOTALREVENUE__C'], yaxis='y2')
        )
        
        # Set layout with two y-axes
        fig.update_layout(
            title='Monthly Performance Trends',
            yaxis=dict(title='Number of Closed Loans'),
            yaxis2=dict(title='Revenue', overlaying='y', side='right'),
            barmode='group'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error in time trend analysis: {str(e)}")
        return None

def create_download_link(df):
    """Create download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="kpi_scores.csv">Download Detailed Report</a>'


def get_llm_insights(scores, api_key):
    """
    Get LLM insights about overall KPI performance
    """
    client = OpenAI(api_key=api_key)
    
    # Prepare performance summary for LLM
    summary = f"""
    Overall Performance Metrics:
    - Overall Achievement: {scores['summary']['overall_achievement']:.1f}%
    - Total KPI Score: {scores['summary']['total_score']:.1f}/{scores['summary']['total_possible']}
    
    Individual KPI Performances:
    """
    
    for kpi, values in scores['detailed_scores'].items():
        summary += f"\n{kpi}:"
        summary += f"\n- Achievement Score: {values['achievement_score']:.1f}%"
        summary += f"\n- Raw Value: {values['raw_value']:.2f}"
        summary += f"\n- Weight: {values['weight']}"
    
    prompt = f"""
    As a Loan Officer Performance Expert, analyze these KPI scores and provide:
    1. Overall performance assessment
    2. Top 3 strengths
    3. Top 3 areas needing improvement
    4. Specific, actionable recommendations
    5. Strategic priorities for next quarter
    
    Performance Data:
    {summary}
    
    Provide your analysis in a clear, structured format with specific insights and actionable recommendations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def get_kpi_specific_insight(kpi_name, kpi_data, api_key):
    """
    Get LLM insights for a specific KPI
    """
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    As a Loan Officer Performance Expert, analyze this specific KPI:
    
    KPI: {kpi_name}
    Achievement Score: {kpi_data['achievement_score']:.1f}%
    Raw Value: {kpi_data['raw_value']:.2f}
    Weight: {kpi_data['weight']}
    Description: {kpi_data['description']}
    
    Provide:
    1. Performance assessment
    2. Specific strengths/concerns
    3. 3 actionable recommendations
    4. Industry best practices for this metric
    
    Keep the response concise and actionable.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insight: {str(e)}"

# Modify your main function to include LLM features
def main():
    st.set_page_config(page_title="Loan Officer KPI Dashboard", layout="wide")
    
    st.title("Loan Officer KPI Performance Dashboard")
    
    # Add API key input in sidebar
    st.sidebar.header("OpenAI Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your opportunities CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        scores = calculate_detailed_kpi_scores(data)
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["KPI Scores", "Visualizations", "AI Insights", "Detailed Analysis"])
        
        with tab1:
            st.header("Individual KPI Scores")
        
            for kpi, values in scores['detailed_scores'].items():
                with st.expander(f"{kpi} - Score: {values['kpi_score']:.2f}/{values['weight']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Achievement", f"{values['achievement_score']:.1f}%")
                        st.metric("Raw Value", f"{values['raw_value']:.2f}")
                    with col2:
                        st.write("Description:", values['description'])
                    
                    # Calculate normalized progress value
                        progress_value = min(1.0, max(0.0, values['achievement_score'] / 100))
                    
                    # Add color-coded progress bars
                        if progress_value < 0.4:
                            st.markdown(
                                f"""<div style="width:100%; background-color:#ff0000; height:20px; 
                                border-radius:10px; overflow:hidden;">
                                <div style="width:{progress_value*100}%; background-color:#ff4444; 
                                height:100%;"></div></div>""", 
                                unsafe_allow_html=True
                            )
                        elif progress_value < 0.7:
                            st.markdown(
                                f"""<div style="width:100%; background-color:#ffa500; height:20px; 
                                border-radius:10px; overflow:hidden;">
                                <div style="width:{progress_value*100}%; background-color:#ffcc00; 
                                height:100%;"></div></div>""", 
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""<div style="width:100%; background-color:#00ff00; height:20px; 
                                border-radius:10px; overflow:hidden;">
                                <div style="width:{progress_value*100}%; background-color:#44ff44; 
                                height:100%;"></div></div>""", 
                                unsafe_allow_html=True
                            )
                
                # Display achievement level
                achievement_level = "Needs Improvement" if progress_value < 0.4 else \
                                  "Good" if progress_value < 0.7 else "Excellent"
                st.write(f"Performance Level: **{achievement_level}**")
                
                if api_key:
                    if st.button(f"Get AI Insights for {kpi}"):
                        with st.spinner("Generating insights..."):
                            insight = get_kpi_specific_insight(kpi, values, api_key)
                            st.markdown("### AI Insights")
                            st.markdown(insight)

        
        with tab2:
            st.header("Performance Visualizations")
        
        # Create gauge chart for overall performance
            st.subheader("Overall Performance")
            gauge_fig = create_performance_summary(scores)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Create two columns for visualizations
            col1, col2 = st.columns(2)
        
            with col1:
            # Radar Chart
                st.subheader("KPI Achievement Radar")
                radar_fig = create_radar_chart(scores)
                st.plotly_chart(radar_fig, use_container_width=True)
            
            # Achievement Distribution
                st.subheader("Achievement Score Distribution")
                dist_fig = create_achievement_distribution(scores)
                st.plotly_chart(dist_fig, use_container_width=True)
        
            with col2:
            # Bar Chart
                st.subheader("KPI Scores vs Weights")
                bar_fig = create_bar_chart(scores)
                st.plotly_chart(bar_fig, use_container_width=True)
            
            # Create metrics table
                st.subheader("Key Metrics Summary")
                metrics_df = pd.DataFrame({
                    'Metric': scores['detailed_scores'].keys(),
                    'Achievement (%)': [f"{v['achievement_score']:.1f}%" for v in scores['detailed_scores'].values()],
                    'Score': [f"{v['kpi_score']:.1f}/{v['weight']}" for v in scores['detailed_scores'].values()]
                })
                st.dataframe(metrics_df, hide_index=True)

        
        with tab3:
            st.header("AI Performance Analysis")
            
            if api_key:
                if st.button("Generate Comprehensive Performance Analysis"):
                    with st.spinner("Analyzing performance data..."):
                        insights = get_llm_insights(scores, api_key)
                        st.markdown(insights)
                
                # Add custom query section
                st.subheader("Ask Custom Questions")
                user_question = st.text_input("What would you like to know about the performance metrics?")
                if user_question:
                    with st.spinner("Generating response..."):
                        client = OpenAI(api_key=api_key)
                        prompt = f"""
                        Given these performance metrics:
                        {scores['summary']}
                        
                        Answer this question: {user_question}
                        
                        Provide a clear, specific answer based on the data.
                        """
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7
                            )
                            st.markdown(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please enter your OpenAI API key in the sidebar to enable AI insights.")
        
        with tab4:
            # [Your existing detailed analysis code]
            pass

if __name__ == "__main__":
    main()