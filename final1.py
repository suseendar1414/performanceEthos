import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from openai import OpenAI
import time
import hmac
import hashlib
import os
from dotenv import load_dotenv
from plotly.subplots import make_subplots

# Load environment variables from .env file
load_dotenv()

def check_credentials(email, password):
    """Check if email and password match environment variables"""
    correct_email = os.getenv('ETHOS_USER_EMAIL')
    correct_password = os.getenv('ETHOS_USER_PASSWORD')
    
    if not correct_email or not correct_password:
        st.error("Error: Login credentials not properly configured. Please check environment variables.")
        return False
    
    return email == correct_email and password == correct_password

def login_page():
    """Display login page and handle authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Ethos KPI Dashboard Login")
        
        # Add logo/branding if needed
        st.markdown("""
            <style>
                .login-container {
                    max-width: 400px;
                    margin: 0 auto;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                .stButton>button {
                    width: 100%;
                    margin-top: 20px;
                }
            </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            email = st.text_input("Email", key="email")
            password = st.text_input("Password", type="password", key="password")
            
            if st.button("Login"):
                if check_credentials(email, password):
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid email or password")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add footer
        st.markdown("""
            <style>
                .footer {
                    position: fixed;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    padding: 10px;
                    text-align: center;
                    background-color: #f0f2f6;
                }
            </style>
            <div class="footer">
                © 2024 Ethos Lending. All rights reserved.
            </div>
        """, unsafe_allow_html=True)
        
        return False
    
    return True


def create_year_over_year_comparison(scores):
    """Create a comparison chart showing performance across years"""
    years = list(scores.keys())
    overall_achievements = [scores[year]['summary']['overall_achievement'] for year in years]
    
    fig = go.Figure()
    
    # Add line chart for overall achievement
    fig.add_trace(go.Scatter(
        x=years,
        y=overall_achievements,
        mode='lines+markers',
        name='Overall Achievement',
        line=dict(color='rgb(55, 83, 109)', width=3),
        marker=dict(size=10)
    ))
    
    # Update layout
    fig.update_layout(
        title='Year-over-Year Performance',
        xaxis_title='Year',
        yaxis_title='Overall Achievement Score',
        width=800,
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_loan_type_charts(scores, selected_year):
    """Create charts showing loan type distribution"""
    kpi_data = scores[selected_year]['detailed_scores']['Loan_Types']
    
    # Create data for charts
    loan_types = ['Conventional', 'ARM', 'FHA/VA']
    values = [kpi_data['conv_count'], kpi_data['arm_count'], kpi_data['fha_va_count']]
    
    # Create subplot with pie chart and bar chart side by side
    fig = go.Figure()
    
    # Create separate subplots for pie and bar charts
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=('Distribution by Percentage', 'Distribution by Count'),
                       specs=[[{"type": "pie"}, {"type": "bar"}]])
    
    # Add pie chart
    fig.add_trace(
        go.Pie(
            labels=loan_types,
            values=values,
            hole=0.4,
            name='Distribution',
            textinfo='label+percent',
            hovertemplate="Type: %{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=loan_types,
            y=values,
            name='Count',
            text=values,
            textposition='auto',
            hovertemplate="Type: %{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=f'Loan Type Distribution - {selected_year}',
        showlegend=False,
        width=800,
        height=400,
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    # Update y-axis for bar chart to start at 0
    fig.update_yaxes(range=[0, max(values) * 1.1], row=1, col=2)
    
    return fig

def create_radar_chart(scores, selected_year):
    """Create radar chart for KPI scores for a specific year"""
    # Extract data for the radar chart
    year_scores = scores[selected_year]['detailed_scores']
    categories = list(year_scores.keys())
    values = [year_scores[cat]['achievement_score'] for cat in categories]
    
    # Create the radar chart
    fig = go.Figure()
    
    # Add the trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f'Achievement Score {selected_year}'
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
        title=f"KPI Achievement Radar - {selected_year}",
        width=600,
        height=500
    )
    
    return fig

def create_bar_chart(scores, selected_year):
    """Create bar chart comparing KPI scores and weights for a specific year"""
    # Prepare data
    year_scores = scores[selected_year]['detailed_scores']
    kpi_scores_df = pd.DataFrame({
        'KPI': list(year_scores.keys()),
        'Score': [v['kpi_score'] for v in year_scores.values()],
        'Weight': [v['weight'] for v in year_scores.values()]
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
        title=f'KPI Scores vs Weights - {selected_year}',
        xaxis_tickangle=-45,
        barmode='group',
        width=800,
        height=500
    )
    
    return fig

def create_performance_summary(scores, selected_year):
    """Create summary metrics visualization for a specific year"""
    fig = go.Figure()
    
    # Add a gauge chart for overall achievement
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=scores[selected_year]['summary']['overall_achievement'],
        title={'text': f"Overall Achievement - {selected_year}"},
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

def calculate_time_to_close(data, year):
    """
    Calculate time to close metrics with proper validation and error handling
    Returns: (avg_days, time_achievement, warning_message)
    """
    benchmark_days = 60
    warning_message = None
    
    try:
        # Filter for closed loans in the specified year
        year_data = data[data['YEAR'] == year].copy()
        closed_loans = year_data[year_data['Closed'] == True]
        
        if len(closed_loans) == 0:
            return 0, 0, "No closed loans found for Time to Close calculation."
        
        # Calculate time difference
        closed_loans['days_to_close'] = (
            pd.to_datetime(closed_loans['Close Date']) - 
            pd.to_datetime(closed_loans['Created Date'])
        ).dt.days
        
        # Remove invalid entries
        valid_loans = closed_loans[closed_loans['days_to_close'] > 0]
        
        if len(valid_loans) == 0:
            return 0, 0, "No valid closing time data found. All entries have 0 or negative days."
        
        if len(valid_loans) < len(closed_loans):
            warning_message = f"Excluded {len(closed_loans) - len(valid_loans)} loans with invalid closing times."
        
        avg_days = valid_loans['days_to_close'].mean()
        time_achievement = (benchmark_days / avg_days) * 100 if avg_days > 0 else 0
        
        return avg_days, time_achievement, warning_message
        
    except Exception as e:
        return 0, 0, f"Error calculating Time to Close: {str(e)}"

def calculate_detailed_kpi_scores(data):
    """Calculate KPI scores with modified loan approval rate calculation"""
    try:
        # Convert dates and create year column
        data['YEAR'] = pd.to_datetime(data['Close Date']).dt.year
        data['Created_YEAR'] = pd.to_datetime(data['Created Date']).dt.year
        
        # Convert and clean numeric columns
        data['Closed'] = pd.to_numeric(data['Closed'], errors='coerce').fillna(0).astype(bool)
        data['Won'] = pd.to_numeric(data['Won'], errors='coerce').fillna(0).astype(bool)
        data['Amount'] = pd.to_numeric(data['Amount'], errors='coerce').fillna(0)
        
        # Store warnings for later display
        warnings = {}
        yearly_scores = {}
        
        for year, year_data in data.groupby('YEAR'):
            if pd.isna(year):
                continue
                
            kpi_scores = {}
            warnings[year] = []
            
            try:
                # 1. Total Number of Loans Closed (Weight: 9)
                closed_loans = int(year_data['Closed'].sum())
                benchmark_loans = 50
                achievement_score = (closed_loans / benchmark_loans) * 100 if benchmark_loans > 0 else 0
                kpi_scores['Total_Loans_Closed'] = {
                    'raw_value': closed_loans,
                    'achievement_score': float(achievement_score),
                    'kpi_score': float((achievement_score / 100) * 9),
                    'weight': 9,
                    'description': 'Total number of loans closed',
                    'benchmark': benchmark_loans
                }
                
                # 2. Total Dollar Value (Weight: 10)
                total_value = year_data['Amount'].sum()
                benchmark_revenue = 17000000
                revenue_achievement = (total_value / benchmark_revenue) * 100 if benchmark_revenue > 0 else 0
                kpi_scores['Total_Dollar_Value'] = {
                    'raw_value': float(total_value),
                    'achievement_score': float(revenue_achievement),
                    'kpi_score': float((revenue_achievement / 100) * 10),
                    'weight': 10,
                    'description': 'Total monetary value of loans',
                    'benchmark': float(benchmark_revenue)
                }
                
                # 3. Loan Types (Weight: 6)
                loan_types = year_data['Type'].fillna('Unknown').astype(str).str.upper().str.strip()
                
                conv_count = sum('CONV' in loan_type for loan_type in loan_types)
                arm_count = sum('ARM' in loan_type for loan_type in loan_types)
                fha_va_count = sum(any(x in loan_type for x in ['FHA', 'VA']) for loan_type in loan_types)
                
                total_loans = max(conv_count + arm_count + fha_va_count, 1)
                
                kpi_scores['Loan_Types'] = {
                    'raw_value': total_loans,
                    'achievement_score': float((total_loans / 50) * 100),
                    'kpi_score': float((min((total_loans / 50), 1) * 100) * 0.06),
                    'weight': 6,
                    'description': 'Distribution of loan types',
                    'benchmark': 50,
                    'conv_count': int(conv_count),
                    'arm_count': int(arm_count),
                    'fha_va_count': int(fha_va_count)
                }
                
                # 4. Average Loan Size (Weight: 8) - MODIFIED CALCULATION
                # Calculate total loan value for all loans
                total_loan_value = year_data['Amount'].sum()
                # Get total number of loans
                total_number_of_loans = len(year_data)
                # Calculate average loan size
                avg_loan = total_loan_value / total_number_of_loans if total_number_of_loans > 0 else 0
                benchmark_avg = 330000
                size_achievement = (avg_loan / benchmark_avg) * 100 if benchmark_avg > 0 else 0
                
                kpi_scores['Average_Loan_Size'] = {
                    'raw_value': float(avg_loan),
                    'achievement_score': float(size_achievement),
                    'kpi_score': float((size_achievement / 100) * 8),
                    'weight': 8,
                    'description': 'Average size of loans (Total Value / Total Loans)',
                    'benchmark': float(benchmark_avg),
                    'total_value': float(total_loan_value),
                    'total_loans': int(total_number_of_loans)  # Added for transparency
                }
                
                # 5. Loan Approval Rate (Weight: 7)
                total_loans = len(year_data)  # Total number of loans
                created_in_year = len(year_data[year_data['Created_YEAR'] == year])  # Loans created in this year
                
                # Calculate approval rate as (created in year / total loans)
                if total_loans > 0:
                    approval_rate = (created_in_year / total_loans) * 100
                else:
                    approval_rate = 0
                    warnings[year].append("No loans found for approval rate calculation")

                benchmark_approval = 85  # Benchmark remains at 85%
                rate_achievement = (approval_rate / benchmark_approval) * 100 if benchmark_approval > 0 else 0
                
                kpi_scores['Loan_Approval_Rate'] = {
                    'raw_value': float(approval_rate),
                    'achievement_score': float(rate_achievement),
                    'kpi_score': float((rate_achievement / 100) * 7),
                    'weight': 7,
                    'description': 'Percentage of loans created in current year vs total loans',
                    'benchmark': float(benchmark_approval),
                    'total_loans': int(total_loans),
                    'created_in_year': int(created_in_year)  # Added for transparency
                }
                
                # Calculate totals for the year
                total_weight = sum(kpi['weight'] for kpi in kpi_scores.values())
                total_score = sum(kpi['kpi_score'] for kpi in kpi_scores.values())
                overall_achievement = (total_score / total_weight * 100) if total_weight > 0 else 0
                
                yearly_scores[year] = {
                    'detailed_scores': kpi_scores,
                    'summary': {
                        'total_score': float(total_score),
                        'total_possible': float(total_weight),
                        'overall_achievement': float(overall_achievement),
                        'number_of_kpis_measured': len(kpi_scores)
                    },
                    'warnings': warnings[year]
                }
                
            except Exception as e:
                warnings[year].append(f"Error processing year {year}: {str(e)}")
                continue
        
        return yearly_scores
    
    except Exception as e:
        st.error(f"Error in calculate_detailed_kpi_scores: {str(e)}")
        raise

# Add this helper function at the top of your code
def format_loan_type_value(value):
    """Helper function to format loan type values for display"""
    if isinstance(value, dict):
        if 'detailed_counts' in value:
            return f"CONV: {value['detailed_counts']['Conventional']}, ARM: {value['detailed_counts']['ARM']}, FHA/VA: {value['detailed_counts']['FHA_VA']}"
        return str(value)
    return f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)

# Modify the benchmark comparison section in the main function
def create_benchmark_comparison(scores, selected_year):
    """Create benchmark comparison data without variance column"""
    benchmark_comparison = []
    
    # Define columns without variance
    columns = ['KPI', 'Actual', 'Benchmark', 'Achievement', 'Score']
    
    for kpi, values in scores[selected_year]['detailed_scores'].items():
        actual = values['raw_value']
        benchmark = values['benchmark']
        
        # Format main KPI row
        row_data = {}
        row_data['KPI'] = kpi.replace('_', ' ')
        
        # Format values based on KPI type
        if kpi == 'Total_Dollar_Value' or kpi == 'Average_Loan_Size':
            row_data['Actual'] = f"${float(actual):,.2f}"
            row_data['Benchmark'] = f"${float(benchmark):,.2f}"
        elif 'Rate' in kpi or 'Percentage' in kpi:
            row_data['Actual'] = f"{float(actual):.1f}%"
            row_data['Benchmark'] = f"{float(benchmark):.1f}%"
        else:
            row_data['Actual'] = f"{float(actual):.1f}"
            row_data['Benchmark'] = f"{float(benchmark):.1f}"
            
        row_data['Achievement'] = f"{values['achievement_score']:.1f}%"
        row_data['Score'] = f"{values['kpi_score']:.1f}/{values['weight']}"
        
        benchmark_comparison.append(row_data)
        
        # Add loan type distribution as sub-rows
        if kpi == 'Loan_Types' and actual > 0:
            # Add Conventional Loans
            conv_pct = (values['conv_count'] / actual * 100) if actual > 0 else 0
            benchmark_comparison.append({
                'KPI': '└─ Conventional',
                'Actual': f"{conv_pct:.1f}%",
                'Benchmark': "53.2%",
                'Achievement': f"{(conv_pct / 53.2 * 100):.1f}%" if conv_pct > 0 else "0.0%",
                'Score': '—'
            })
            
            # Add ARM Loans
            arm_pct = (values['arm_count'] / actual * 100) if actual > 0 else 0
            benchmark_comparison.append({
                'KPI': '└─ ARM',
                'Actual': f"{arm_pct:.1f}%",
                'Benchmark': "23.7%",
                'Achievement': f"{(arm_pct / 23.7 * 100):.1f}%" if arm_pct > 0 else "0.0%",
                'Score': '—'
            })
            
            # Add FHA/VA Loans
            fha_va_pct = (values['fha_va_count'] / actual * 100) if actual > 0 else 0
            benchmark_comparison.append({
                'KPI': '└─ FHA/VA',
                'Actual': f"{fha_va_pct:.1f}%",
                'Benchmark': "19.8%",
                'Achievement': f"{(fha_va_pct / 19.8 * 100):.1f}%" if fha_va_pct > 0 else "0.0%",
                'Score': '—'
            })
    
    return pd.DataFrame(benchmark_comparison, columns=columns)

def style_dataframe(df):
    """Apply styles to the dataframe without variance styling"""
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Style row backgrounds
    for idx in df.index:
        if df.loc[idx, 'KPI'].startswith('└─'):
            styles.loc[idx, :] = 'background-color: #f8f9fa; color: #666666'
        else:
            styles.loc[idx, 'KPI'] = 'font-weight: bold'
    
    return styles

def validate_csv_data(data):
    """Enhanced validation of CSV data before processing"""
    required_columns = ['Close Date', 'Amount', 'Type', 'Won', 'Closed', 'Created Date']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.write("Your file contains these columns:")
        st.write(list(data.columns))
        return False
    
    # Create a copy of the data for validation
    valid_data = data.copy()
    
    try:
        # Convert and clean date columns
        valid_data['Close Date'] = pd.to_datetime(valid_data['Close Date'], errors='coerce')
        valid_data['Created Date'] = pd.to_datetime(valid_data['Created Date'], errors='coerce')
        
        # Convert numeric columns, replacing NaN with 0
        valid_data['Amount'] = pd.to_numeric(valid_data['Amount'], errors='coerce').fillna(0)
        valid_data['Closed'] = pd.to_numeric(valid_data['Closed'], errors='coerce').fillna(0)
        valid_data['Won'] = pd.to_numeric(valid_data['Won'], errors='coerce').fillna(0)
        
        # Clean Type column
        valid_data['Type'] = valid_data['Type'].fillna('Unknown')
        
        # Report data quality issues
        total_rows = len(valid_data)
        valid_rows = len(valid_data.dropna(subset=['Close Date', 'Created Date']))
        
        if valid_rows < total_rows:
            st.warning(f"Found {total_rows - valid_rows} rows with missing or invalid dates. These will be excluded from analysis.")
        
        if valid_rows == 0:
            st.error("No valid data rows found after cleaning. Please check your data format.")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Error validating data: {str(e)}")
        return False


# Modify the display_kpi_details function to include the charts
def display_kpi_details(scores, selected_year, kpi_name):
    """Display KPI details with proper warning handling and charts"""
    try:
        kpi_data = scores[selected_year]['detailed_scores'][kpi_name]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Performance Metrics")
            st.metric(
                "Achievement Score",
                f"{kpi_data['achievement_score']:.1f}%"
                # Removed the delta parameter to not show points
            )
            
            if kpi_name == 'Loan_Types':
                st.subheader("Loan Type Distribution")
                st.metric("Conventional", str(kpi_data['conv_count']))
                st.metric("ARM", str(kpi_data['arm_count']))
                st.metric("FHA/VA", str(kpi_data['fha_va_count']))
                st.metric("Total Loans", str(kpi_data['raw_value']))
                
                # Add distribution charts
                st.markdown("---")
                st.subheader("Distribution Analysis")
                fig = create_loan_type_charts(scores, selected_year)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add summary statistics
                total = kpi_data['raw_value']
                if total > 0:
                    st.markdown("### Distribution Percentages")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        conv_pct = (kpi_data['conv_count'] / total) * 100
                        st.metric("Conventional", f"{conv_pct:.1f}%")
                    with col2:
                        arm_pct = (kpi_data['arm_count'] / total) * 100
                        st.metric("ARM", f"{arm_pct:.1f}%")
                    with col3:
                        fha_va_pct = (kpi_data['fha_va_count'] / total) * 100
                        st.metric("FHA/VA", f"{fha_va_pct:.1f}%")
            else:
                if kpi_name == 'Time_to_Close':
                    st.metric(
                        "Average Days to Close",
                        f"{kpi_data['raw_value']:.1f} days"
                    )
                elif kpi_name in ['Total_Dollar_Value', 'Average_Loan_Size']:
                    st.metric(
                        "Actual Value",
                        f"${kpi_data['raw_value']:,.2f}"
                    )
                else:
                    st.metric(
                        "Actual Value",
                        f"{kpi_data['raw_value']:.2f}"
                    )
        
        with col2:
            st.subheader("Benchmark Details")
            st.write(str(kpi_data['description']))
            if kpi_name == 'Time_to_Close':
                st.write(f"**Target:** {kpi_data['benchmark']} days")
            elif kpi_name in ['Total_Dollar_Value', 'Average_Loan_Size']:
                st.write(f"**Target:** ${kpi_data['benchmark']:,.2f}")
            else:
                st.write(f"**Target:** {kpi_data['benchmark']:.2f}")
        
    except Exception as e:
        st.error(f"Error displaying KPI details: {str(e)}")

def generate_kpi_insights(api_key, scores, selected_year):
    """Generate insights about KPI performance using OpenAI"""
    try:
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar to get AI-powered insights.")
            return None
            
        client = OpenAI(api_key=api_key)
        
        # Prepare the data for analysis
        year_data = scores[selected_year]
        detailed_scores = year_data['detailed_scores']
        
        # Create a structured summary of the KPI data
        kpi_summary = []
        for kpi_name, kpi_data in detailed_scores.items():
            kpi_summary.append(f"{kpi_name}:")
            kpi_summary.append(f"- Actual Value: {kpi_data['raw_value']:.2f}")
            kpi_summary.append(f"- Benchmark: {kpi_data['benchmark']}")
            kpi_summary.append(f"- Achievement Score: {kpi_data['achievement_score']:.1f}%")
            kpi_summary.append(f"- Weight: {kpi_data['weight']}")
            kpi_summary.append("")
        
        overall_achievement = year_data['summary']['overall_achievement']
        
        # Create the prompt
        prompt = f"""Given the following KPI performance data for a loan officer in {selected_year}:

Overall Achievement: {overall_achievement:.1f}%

{chr(10).join(kpi_summary)}

Please provide a brief analysis including:
1. Top performing areas
2. Areas needing improvement
3. Specific actionable recommendations
4. Key trends or patterns
5. Strategic suggestions for next quarter

Format the response with clear headers and bullet points."""

        # Get insights from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert financial and KPI analyst specializing in loan officer performance analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return None

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_llm_insights(scores, selected_year):
    """
    Get LLM insights about overall KPI performance
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI API key not found in environment variables. Please check your .env file configuration."
    
    client = OpenAI(api_key=api_key)
    
    # Prepare performance summary for LLM
    summary = f"""
    Overall Performance Metrics:
    - Overall Achievement: {scores[selected_year]['summary']['overall_achievement']:.1f}%
    - Total KPI Score: {scores[selected_year]['summary']['total_score']:.1f}/{scores[selected_year]['summary']['total_possible']}
    
    Individual KPI Performances:
    """
    
    for kpi, values in scores[selected_year]['detailed_scores'].items():
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

def get_kpi_specific_insight(kpi_name, kpi_data):
    """
    Get LLM insights for a specific KPI
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return "OpenAI API key not found in environment variables. Please check your .env file configuration."
    
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

def add_ai_insights_tab(scores, selected_year):
    """Add AI Insights tab to the dashboard with enhanced LLM analysis"""
    st.header("AI-Powered Performance Insights")
    
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.warning("OpenAI API key not found in environment variables. Please check your .env file configuration.")
            return
            
        client = OpenAI(api_key=api_key)
        
        # Create tabs for different types of insights
        overview_tab, detailed_tab = st.tabs(["Overall Performance", "KPI-Specific Insights"])
        
        with overview_tab:
            with st.spinner("Generating overall performance insights..."):
                overall_insights = get_llm_insights(scores, selected_year)
                st.markdown(overall_insights)
                
                # Add download button for overall insights
                if overall_insights:
                    st.download_button(
                        label="Download Overall Insights",
                        data=overall_insights,
                        file_name=f"overall_insights_{selected_year}.txt",
                        mime="text/plain"
                    )
        
        with detailed_tab:
            # KPI selector for detailed insights
            selected_kpi = st.selectbox(
                "Select KPI for detailed analysis",
                options=list(scores[selected_year]['detailed_scores'].keys()),
                format_func=lambda x: x.replace('_', ' ')
            )
            
            if selected_kpi:
                with st.spinner(f"Analyzing {selected_kpi.replace('_', ' ')}..."):
                    kpi_data = scores[selected_year]['detailed_scores'][selected_kpi]
                    kpi_insight = get_kpi_specific_insight(selected_kpi, kpi_data)
                    
                    # Display KPI metrics and insight
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Achievement Score", f"{kpi_data['achievement_score']:.1f}%")
                        st.metric("Weight", f"{kpi_data['weight']}")
                        st.metric("Raw Value", f"{kpi_data['raw_value']:.2f}")
                    
                    with col2:
                        st.markdown(kpi_insight)
                        
                        # Add download button for KPI-specific insights
                        if kpi_insight:
                            st.download_button(
                                label=f"Download {selected_kpi.replace('_', ' ')} Analysis",
                                data=kpi_insight,
                                file_name=f"{selected_kpi.lower()}_analysis_{selected_year}.txt",
                                mime="text/plain"
                            )
        
        # Add historical trends analysis
        if len(scores) > 1:
            st.markdown("---")
            st.subheader("Historical Performance Analysis")
            
            with st.spinner("Analyzing historical trends..."):
                # Prepare historical data summary
                historical_summary = "Historical Performance:\n"
                for year in sorted(scores.keys()):
                    historical_summary += f"\n{year}:"
                    historical_summary += f"\n- Overall Achievement: {scores[year]['summary']['overall_achievement']:.1f}%"
                    # Add more detailed metrics for better analysis
                    for kpi_name, kpi_data in scores[year]['detailed_scores'].items():
                        historical_summary += f"\n  {kpi_name}: {kpi_data['achievement_score']:.1f}%"
                
                # Get historical insights
                historical_prompt = f"""
                As a Loan Officer Performance Expert, analyze these historical performance trends:
                {historical_summary}
                
                Please provide:
                1. Key performance trends over time
                2. Notable year-over-year improvements or declines
                3. Consistent strengths and persistent challenges
                4. Long-term strategic recommendations
                5. Industry context and benchmarking insights
                
                Focus on actionable insights and specific recommendations for sustained improvement.
                """
                
                try:
                    historical_response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": historical_prompt}],
                        temperature=0.7,
                        max_tokens=800
                    )
                    st.markdown(historical_response.choices[0].message.content)
                    
                    # Add download button for historical analysis
                    historical_insight = historical_response.choices[0].message.content
                    st.download_button(
                        label="Download Historical Analysis",
                        data=historical_insight,
                        file_name=f"historical_analysis_{min(scores.keys())}-{max(scores.keys())}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error analyzing historical trends: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")

def main():
    st.set_page_config(page_title="Ethos Lending Mortgages Under Management (MUM) by Loan Officer", layout="wide")
    
    # Add credentials input in the sidebar when not authenticated
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.credentials = None

    # Sidebar for credentials
    with st.sidebar:
        st.header("Dashboard Settings")
        
        if not st.session_state.authenticated:
            st.subheader("Login Credentials")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if email and password:  # Basic validation
                    st.session_state.credentials = {'email': email, 'password': password}
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Please enter both email and password")
        else:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.credentials = None
                st.rerun()
    
    # Main dashboard content
    if not st.session_state.authenticated:
        st.title("Ethos KPI Dashboard")
        st.info("Please login using the sidebar to access the dashboard.")
        return
    
    st.title("Ethos Lending Mortgages Under Management (MUM) by Loan Officer")
    
    # File upload
    st.sidebar.markdown("---")  # Separator
    st.sidebar.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your opportunities CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully. Validating data...")
            
            # Validate data
            if not validate_csv_data(data):
                return
            
            # Process data
            try:
                with st.spinner("Processing data and calculating KPIs..."):
                    scores = calculate_detailed_kpi_scores(data)
                    
                if not scores:
                    st.error("No valid data to analyze after processing.")
                    return
                
                st.success("Data processed successfully!")
                
                # Add year selection in sidebar
                available_years = sorted(list(scores.keys()))
                selected_year = st.sidebar.selectbox(
                    "Select Year for Analysis",
                    available_years,
                    index=len(available_years)-1  # Default to most recent year
                )
                
                # Create tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Year Overview", 
                    "KPI Scores", 
                    "Visualizations", 
                    "Year Comparison", 
                    "AI Insights"
                ])
                
                with tab1:
                    st.header(f"Performance Overview - {selected_year}")
                    
                    # Create three columns for summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Overall Achievement",
                            f"{scores[selected_year]['summary']['overall_achievement']:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Total KPIs Measured",
                            scores[selected_year]['summary']['number_of_kpis_measured']
                        )
                    
                    with col3:
                        st.metric(
                            "Total Score",
                            f"{scores[selected_year]['summary']['total_score']:.1f}/{scores[selected_year]['summary']['total_possible']}"
                        )
                    
                    # Add year-over-year comparison chart
                    st.subheader("Performance Trends")
                    yoy_fig = create_year_over_year_comparison(scores)
                    st.plotly_chart(yoy_fig, use_container_width=True)
                
                with tab2:
                    st.header(f"Individual KPI Scores - {selected_year}")
    
                    # Add a summary table at the top
                    st.subheader("KPI Performance vs Benchmarks")
                    comparison_df = create_benchmark_comparison(scores, selected_year)
                    
                    def style_dataframe(df):
                        """Apply styles to the dataframe"""
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        
                        # Style row backgrounds
                        for idx in df.index:
                            if df.loc[idx, 'KPI'].startswith('└─'):
                                styles.loc[idx, :] = 'background-color: #f8f9fa; color: #666666'
                            else:
                                styles.loc[idx, 'KPI'] = 'font-weight: bold'
                        
                        return styles
                    
                    styled_df = comparison_df.style.apply(style_dataframe, axis=None)
                    st.dataframe(styled_df, hide_index=True)
                    
                    # Detailed KPI Breakdowns
                    st.markdown("---")
                    st.subheader("Detailed KPI Analysis")
                    
                    for kpi, values in scores[selected_year]['detailed_scores'].items():
                        with st.expander(f"{kpi.replace('_', ' ')}"):
                            display_kpi_details(scores, selected_year, kpi)
                
                with tab3:
                    st.header(f"Performance Visualizations - {selected_year}")
                    
                    # Create two columns for visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar Chart
                        radar_fig = create_radar_chart(scores, selected_year)
                        st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")
                        
                        # Performance Summary
                        gauge_fig = create_performance_summary(scores, selected_year)
                        st.plotly_chart(gauge_fig, use_container_width=True, key="gauge_chart")
                    
                    with col2:
                        # Bar Chart
                        bar_fig = create_bar_chart(scores, selected_year)
                        st.plotly_chart(bar_fig, use_container_width=True, key="bar_chart")
                        
                        # Add loan type distribution chart if data exists
                        if 'Loan_Types' in scores[selected_year]['detailed_scores']:
                            loan_type_fig = create_loan_type_charts(scores, selected_year)
                            st.plotly_chart(loan_type_fig, use_container_width=True, key="loan_type_chart")
                
                with tab4:
                    st.header("Year-over-Year Analysis")
                    
                    # Create comparison table
                    comparison_data = []
                    for year in available_years:
                        year_summary = scores[year]['summary']
                        comparison_data.append({
                            'Year': year,
                            'Overall Achievement': f"{year_summary['overall_achievement']:.1f}%",
                            'Total Score': f"{year_summary['total_score']:.1f}",
                            'KPIs Measured': year_summary['number_of_kpis_measured']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, hide_index=True)
                    
                    # Add year-over-year comparison charts
                    st.subheader("Performance Trends")
                    yoy_fig = create_year_over_year_comparison(scores)
                    st.plotly_chart(yoy_fig, use_container_width=True, key="yoy_comparison")
                    
                    # Additional year-by-year KPI comparison
                    st.subheader("KPI Performance by Year")
                    
                    # Create multi-year KPI comparison chart
                    kpi_comparison_data = []
                    for year in available_years:
                        for kpi, values in scores[year]['detailed_scores'].items():
                            kpi_comparison_data.append({
                                'Year': year,
                                'KPI': kpi,
                                'Achievement Score': values['achievement_score']
                            })
                    
                    kpi_comparison_df = pd.DataFrame(kpi_comparison_data)
                    
                    # Create heatmap for KPI performance across years
                    kpi_heatmap = go.Figure(data=go.Heatmap(
                        z=kpi_comparison_df.pivot(index='Year', columns='KPI', values='Achievement Score'),
                        x=kpi_comparison_df['KPI'].unique(),
                        y=kpi_comparison_df['Year'].unique(),
                        colorscale='Viridis',
                        text=kpi_comparison_df.pivot(index='Year', columns='KPI', values='Achievement Score').round(1),
                        texttemplate='%{text}%',
                        textfont={"size": 10},
                        hoverongaps=False
                    ))
                    
                    kpi_heatmap.update_layout(
                        title='KPI Achievement Scores Across Years',
                        width=800,
                        height=400
                    )
                    
                    st.plotly_chart(kpi_heatmap, use_container_width=True, key="kpi_heatmap")
                
                with tab5:
                    add_ai_insights_tab(scores, selected_year)
                
            except Exception as e:
                st.error(f"Error calculating KPI scores: {str(e)}")
                st.write("Please check your data format and contents.")
                return
                
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            if 'data' in locals():
                st.write("Your file contains these columns:")
                st.write(list(data.columns))
            return

        # Display any warnings
        if 'scores' in locals() and selected_year in scores and scores[selected_year]['warnings']:
            with st.expander("Data Processing Warnings"):
                for warning in scores[selected_year]['warnings']:
                    st.warning(warning)

if __name__ == "__main__":
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.credentials = None
    main()
