import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from openai import OpenAI
import time

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

def calculate_detailed_kpi_scores(data):
    """Calculate individual scores for each KPI metric based on loan metrics, grouped by year"""
    # Convert dates and create year column
    data['YEAR'] = pd.to_datetime(data['Close Date']).dt.year
    
    # Convert float values to boolean (1.0 = True, 0.0 = False)
    data['Closed'] = data['Closed'].astype(float).fillna(0).astype(bool)
    data['Won'] = data['Won'].astype(float).fillna(0).astype(bool)
    
    # Group scores by year
    yearly_scores = {}
    
    for year, year_data in data.groupby('YEAR'):
        kpi_scores = {}
        
        # 1. Total Number of Loans Closed (Weight: 9)
        closed_loans = int(year_data['Closed'].sum())  # Convert to integer for count
        benchmark_loans = 50  # Annual benchmark
        achievement_score = (closed_loans / benchmark_loans) * 100  # Removed min(100, ...)
        kpi_scores['Total_Loans_Closed'] = {
            'raw_value': closed_loans,
            'achievement_score': achievement_score,
            'kpi_score': (achievement_score / 100) * 9,
            'weight': 9,
            'description': f'Total number of loans closed in {year}',
            'benchmark': benchmark_loans
        }
        
        # 2. Total Dollar Value (Weight: 10)
        total_value = year_data['Amount'].fillna(0).sum()
        benchmark_revenue = 17000000  # Annual benchmark
        revenue_achievement = (total_value / benchmark_revenue) * 100  # Removed min(100, ...)
        kpi_scores['Total_Dollar_Value'] = {
            'raw_value': total_value,
            'achievement_score': revenue_achievement,
            'kpi_score': (revenue_achievement / 100) * 10,
            'weight': 10,
            'description': f'Total monetary value of all loans in {year}',
            'benchmark': benchmark_revenue
        }
        
        # 3. Loan Types (Weight: 6)
        if 'Type' in year_data.columns:
            loan_types = year_data['Type'].value_counts(normalize=True) * 100
            benchmark_types = {
                'Conventional': 53.2,
                'ARM': 23.7,
                'FHA_VA': 19.8
            }
            # Allow exceeding benchmarks
            type_achievement = sum(loan_types.get(type_name, 0) 
                                 for type_name, benchmark_pct in benchmark_types.items())
            
            kpi_scores['Loan_Types'] = {
                'raw_value': len(loan_types),
                'achievement_score': type_achievement,
                'kpi_score': (type_achievement / 100) * 6,
                'weight': 6,
                'description': f'Diversity of loan types handled in {year}',
                'benchmark': benchmark_types
            }
        
        # 4. Average Loan Size (Weight: 8)
        avg_loan = year_data['Amount'].fillna(0).mean()
        benchmark_avg = 330000
        size_achievement = (avg_loan / benchmark_avg) * 100  # Removed min(100, ...)
        kpi_scores['Average_Loan_Size'] = {
            'raw_value': avg_loan,
            'achievement_score': size_achievement,
            'kpi_score': (size_achievement / 100) * 8,
            'weight': 8,
            'description': f'Average size of loans processed in {year}',
            'benchmark': benchmark_avg
        }
        
        # 5. Loan Approval Rate (Weight: 7)
        won_count = year_data['Won'].sum()
        total_count = len(year_data)
        approval_rate = (won_count / total_count * 100) if total_count > 0 else 0
        benchmark_approval = 85
        rate_achievement = (approval_rate / benchmark_approval) * 100  # Removed min(100, ...)
        kpi_scores['Loan_Approval_Rate'] = {
            'raw_value': approval_rate,
            'achievement_score': rate_achievement,
            'kpi_score': (rate_achievement / 100) * 7,
            'weight': 7,
            'description': f'Percentage of loans approved in {year}',
            'benchmark': benchmark_approval
        }
        
        # 6. Time to Close (Weight: 6)
        if 'Created Date' in year_data.columns and 'Close Date' in year_data.columns:
            try:
                benchmark_days = 60
                
                year_data['Created Date'] = pd.to_datetime(year_data['Created Date'])
                year_data['Close Date'] = pd.to_datetime(year_data['Close Date'])
                
                # Filter for closed loans
                closed_loans = year_data[year_data['Closed'] == True]
                
                if len(closed_loans) > 0:
                    avg_days = (closed_loans['Close Date'] - closed_loans['Created Date']).dt.days.mean()
                    avg_days = max(0, avg_days)  # Ensure non-negative
                    time_achievement = ((benchmark_days - avg_days) / benchmark_days) * 100  # Allow exceeding 100%
                else:
                    avg_days = 0
                    time_achievement = 0
                
                kpi_scores['Time_to_Close'] = {
                    'raw_value': avg_days,
                    'achievement_score': time_achievement,
                    'kpi_score': (time_achievement / 100) * 6,
                    'weight': 6,
                    'description': f'Average time taken to close loans in {year}',
                    'benchmark': benchmark_days
                }
            except Exception as e:
                st.warning(f"Error calculating Time to Close for {year}: {str(e)}")
                kpi_scores['Time_to_Close'] = {
                    'raw_value': 0,
                    'achievement_score': 0,
                    'kpi_score': 0,
                    'weight': 6,
                    'description': f'Average time taken to close loans in {year}',
                    'benchmark': 60
                }
        
        # Calculate totals for the year
        total_weight = sum(kpi['weight'] for kpi in kpi_scores.values())
        total_score = sum(kpi['kpi_score'] for kpi in kpi_scores.values())
        overall_achievement = (total_score / total_weight * 100) if total_weight > 0 else 0
        
        yearly_scores[year] = {
            'detailed_scores': kpi_scores,
            'summary': {
                'total_score': total_score,
                'total_possible': total_weight,
                'overall_achievement': overall_achievement,
                'number_of_kpis_measured': len(kpi_scores)
            }
        }
    
    return yearly_scores

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

def add_ai_insights_tab():
    """Add AI Insights tab to the dashboard"""
    # Get OpenAI API key from sidebar
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    
    if api_key:
        try:
            # Generate insights
            insights = generate_kpi_insights(api_key, scores, selected_year)
            
            if insights:
                # Display insights in a formatted way
                st.header("AI-Powered Performance Insights")
                st.markdown(insights)
                
                # Add a download button for the insights
                def get_binary_file_downloader_html(text):
                    b64 = base64.b64encode(text.encode()).decode()
                    filename = f"kpi_insights_{selected_year}.txt"
                    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Insights Report</a>'
                    return href
                
                st.markdown("---")
                st.markdown(get_binary_file_downloader_html(insights), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error in AI analysis: {str(e)}")
            st.write("Please check your OpenAI API key and try again.")
    else:
        st.info("Enter your OpenAI API key in the sidebar to get AI-powered insights about your KPI performance.")

# Update the main function to include year selection and yearly visualizations
def main():
    st.set_page_config(page_title="Ethos KPI Dashboard", layout="wide")
    
    st.title("Ethos KPI Performance Dashboard")
    
    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.file_uploader("Upload your opportunities CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            data = pd.read_csv(uploaded_file)
            
            # Convert dates
            data['Close Date'] = pd.to_datetime(data['Close Date'], errors='coerce')
            valid_data = data.dropna(subset=['Close Date'])
            
            if len(valid_data) == 0:
                st.error("No valid Close Date values found in the data. Please check your data format.")
                return
            
            # Convert numeric columns
            valid_data['Amount'] = pd.to_numeric(valid_data['Amount'], errors='coerce').fillna(0)
            valid_data['Closed'] = pd.to_numeric(valid_data['Closed'], errors='coerce').fillna(0)
            valid_data['Won'] = pd.to_numeric(valid_data['Won'], errors='coerce').fillna(0)
            
            # Calculate scores
            global scores, selected_year  # Make these available to the AI insights function
            scores = calculate_detailed_kpi_scores(valid_data)
            
            if not scores:
                st.error("No valid data to analyze. Please check your data format.")
                return
                
            # Add year selection in sidebar
            available_years = sorted(list(scores.keys()))
            selected_year = st.sidebar.selectbox(
                "Select Year for Analysis",
                available_years,
                index=len(available_years)-1  # Default to most recent year
            )
            
            # Create tabs including the new AI Insights tab
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Year Overview", "KPI Scores", "Visualizations", "Year Comparison", "AI Insights"])
            
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
                st.plotly_chart(yoy_fig, use_container_width=True, key="yoy_overview")
            
            with tab2:
                st.header(f"Individual KPI Scores - {selected_year}")
                
                # Add a summary table at the top
                st.subheader("KPI Performance vs Benchmarks")
                benchmark_comparison = []
                
                for kpi, values in scores[selected_year]['detailed_scores'].items():
                    actual = values['raw_value']
                    benchmark = values['benchmark']
                    
                    if isinstance(benchmark, dict):
                        # For loan types which have multiple benchmarks
                        benchmark_str = " / ".join([f"{k}: {v}%" for k, v in benchmark.items()])
                        actual_str = " / ".join([f"{k}: {actual:.1f}%" for k in benchmark.keys()])
                    else:
                        # Format numbers based on their magnitude
                        if kpi == 'Total_Dollar_Value':
                            benchmark_str = f"${benchmark:,.0f}"
                            actual_str = f"${actual:,.0f}"
                        elif kpi == 'Average_Loan_Size':
                            benchmark_str = f"${benchmark:,.0f}"
                            actual_str = f"${actual:,.0f}"
                        elif 'Rate' in kpi or 'Percentage' in kpi:
                            benchmark_str = f"{benchmark:.1f}%"
                            actual_str = f"{actual:.1f}%"
                        else:
                            benchmark_str = f"{benchmark:,.1f}"
                            actual_str = f"{actual:,.1f}"
                    
                    # Calculate variance from benchmark
                    if isinstance(benchmark, dict):
                        variance = values['achievement_score'] - 100  # Use achievement score for complex benchmarks
                    else:
                        variance = ((actual / benchmark) - 1) * 100
                    
                    benchmark_comparison.append({
                        'KPI': kpi.replace('_', ' '),
                        'Actual': actual_str,
                        'Benchmark': benchmark_str,
                        'Achievement': f"{values['achievement_score']:.1f}%",
                        'Variance': f"{variance:+.1f}%",
                        'Score': f"{values['kpi_score']:.1f}/{values['weight']}"
                    })
                
                comparison_df = pd.DataFrame(benchmark_comparison)
                
                # Style the dataframe
                def color_variance(val):
                    val_num = float(val.strip('%').strip('+'))
                    if val_num > 0:
                        return 'background-color: #a8f0a8'  # Light green
                    elif val_num < 0:
                        return 'background-color: #ffb3b3'  # Light red
                    return ''
                
                styled_df = comparison_df.style.apply(lambda x: [''] * len(x) if x.name != 'Variance' 
                                                    else [color_variance(v) for v in x], axis=1)
                
                st.dataframe(styled_df, hide_index=True)
                
                # Detailed KPI Breakdowns
                st.markdown("---")
                st.subheader("Detailed KPI Analysis")
                
                for kpi, values in scores[selected_year]['detailed_scores'].items():
                    with st.expander(f"{kpi.replace('_', ' ')}"):
                        col1, col2, col3 = st.columns([2,2,1])
                        
                        with col1:
                            st.subheader("Performance Metrics")
                            st.metric(
                                "Achievement Score", 
                                f"{values['achievement_score']:.1f}%",
                                f"{((values['achievement_score']/100) * values['weight']):.1f} points"
                            )
                            
                            actual = values['raw_value']
                            benchmark = values['benchmark']
                            
                            if isinstance(benchmark, dict):
                                st.write("**Distribution Comparison:**")
                                dist_df = pd.DataFrame({
                                    'Category': benchmark.keys(),
                                    'Benchmark': benchmark.values(),
                                    'Actual': [actual] * len(benchmark)
                                })
                                st.dataframe(dist_df)
                            else:
                                if 'Dollar' in kpi or 'Loan_Size' in kpi:
                                    st.metric(
                                        "Actual Value", 
                                        f"${actual:,.2f}",
                                        f"${(actual - benchmark):,.2f} vs benchmark"
                                    )
                                else:
                                    st.metric(
                                        "Actual Value",
                                        f"{actual:,.2f}",
                                        f"{(actual - benchmark):,.2f} vs benchmark"
                                    )
                        
                        with col2:
                            st.subheader("Benchmark Details")
                            st.write(values['description'])
                            if isinstance(benchmark, dict):
                                st.write("**Target Distribution:**")
                                for k, v in benchmark.items():
                                    st.write(f"- {k}: {v}%")
                            else:
                                if 'Dollar' in kpi or 'Loan_Size' in kpi:
                                    st.write(f"**Target:** ${benchmark:,.2f}")
                                else:
                                    st.write(f"**Target:** {benchmark:,.2f}")
                        
                        with col3:
                            st.subheader("Weight")
                            st.metric(
                                "KPI Weight",
                                f"{values['weight']}",
                                f"of {scores[selected_year]['summary']['total_possible']}"
                            )
                        
                        # Progress visualization
                        st.markdown("---")
                        st.write("**Progress to Target:**")
                        progress_value = min(1.0, max(0.0, values['achievement_score'] / 100))
                        
                        # Create a more detailed progress bar with performance levels
                        col1, col2, col3 = st.columns([1,2,1])
                        with col2:
                            performance_color = (
                                "#ff4444" if progress_value < 0.4 else
                                "#ffcc00" if progress_value < 0.7 else
                                "#44ff44"
                            )
                            
                            performance_level = (
                                "Needs Improvement" if progress_value < 0.4 else
                                "Good" if progress_value < 0.7 else
                                "Excellent"
                            )
                            
                            st.markdown(
                                f"""
                                <div style="width:100%; background-color:#f0f0f0; height:30px; border-radius:15px; overflow:hidden;">
                                    <div style="width:{progress_value*100}%; background-color:{performance_color}; height:100%; 
                                    text-align:center; line-height:30px; color:white;">
                                        {values['achievement_score']:.1f}%
                                    </div>
                                </div>
                                <p style="text-align:center; margin-top:5px;">Performance Level: <strong>{performance_level}</strong></p>
                                """, 
                                unsafe_allow_html=True
                            )
            
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
                add_ai_insights_tab()
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.write("Please ensure your CSV file contains the following columns:")
            st.write("- 'Close Date'")
            st.write("- 'Amount'")
            st.write("- 'Type'")
            st.write("- 'Won'")
            st.write("- 'Closed'")
            st.write("- 'Created Date'")
            
            # Display the actual columns in the uploaded file
            st.write("\nYour file contains these columns:")
            if 'data' in locals():
                st.write(list(data.columns))

if __name__ == "__main__":
    main()
