import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Economic Data Statistical Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FRED API configuration
FRED_API_KEY = "e373038c7a1042761613c33902c3156d"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

@st.cache_data
def fetch_fred_data(series_id, start_date="2010-01-01"):
    """Fetch data from FRED API"""
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': start_date
    }
    
    try:
        response = requests.get(FRED_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        observations = data['observations']
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna()
        
        return df[['date', 'value']].rename(columns={'value': series_id})
    except Exception as e:
        st.error(f"Error fetching data for {series_id}: {str(e)}")
        return pd.DataFrame()

def calculate_descriptive_stats(data, var_name):
    """Calculate comprehensive descriptive statistics"""
    return {
        'Count': len(data),
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Mode': stats.mode(data, keepdims=True)[0][0] if len(data) > 0 else np.nan,
        'Standard Deviation': np.std(data, ddof=1),
        'Variance': np.var(data, ddof=1),
        'Minimum': np.min(data),
        'Maximum': np.max(data),
        'Range': np.max(data) - np.min(data),
        'Q1 (25th percentile)': np.percentile(data, 25),
        'Q3 (75th percentile)': np.percentile(data, 75),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data)
    }

def main():
    st.title("🏛️ Economic Data Statistical Analysis Dashboard")
    st.markdown("### *A Comprehensive Statistical Toolkit Using FRED Economic Data*")
    
    st.markdown("""
    **Welcome to DATA 3010/03!** This dashboard demonstrates the statistical analysis capabilities 
    you'll master by the end of this course. We're analyzing real economic data from the Federal Reserve Economic Data (FRED) database.
    """)
    
    # Sidebar for data selection
    st.sidebar.header("📊 Data Selection")
    
    # Predefined economic indicators
    economic_indicators = {
        'GDP': 'GDP',
        'Unemployment Rate': 'UNRATE', 
        'Consumer Price Index': 'CPIAUCSL',
        'Federal Funds Rate': 'FEDFUNDS',
        'Industrial Production': 'INDPRO',
        'Personal Consumption Expenditures': 'PCE',
        'Housing Starts': 'HOUST',
        'S&P 500': 'SP500'
    }
    
    var1_name = st.sidebar.selectbox("Select First Variable:", list(economic_indicators.keys()), index=0)
    var2_name = st.sidebar.selectbox("Select Second Variable:", list(economic_indicators.keys()), index=1)
    
    var1_code = economic_indicators[var1_name]
    var2_code = economic_indicators[var2_name]
    
    start_date = st.sidebar.date_input("Start Date:", datetime(2015, 1, 1))
    
    # Fetch data
    with st.spinner("Fetching economic data from FRED..."):
        df1 = fetch_fred_data(var1_code, start_date.strftime('%Y-%m-%d'))
        df2 = fetch_fred_data(var2_code, start_date.strftime('%Y-%m-%d'))
    
    if df1.empty or df2.empty:
        st.error("Could not fetch data. Please check your selection.")
        return
    
    # Merge data
    df_merged = pd.merge(df1, df2, on='date', how='inner')
    
    if df_merged.empty:
        st.error("No overlapping data found for selected variables.")
        return
    
    st.success(f"✅ Successfully loaded {len(df_merged)} observations")
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Time Series", "📊 Descriptive Stats", "📉 Distributions", 
        "🔗 Correlation Analysis", "📏 Confidence Intervals", "📋 Summary Report"
    ])
    
    with tab1:
        st.header("Time Series Analysis")
        
        # Interactive time series plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{var1_name} Over Time', f'{var2_name} Over Time'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df_merged['date'], y=df_merged[var1_code], 
                      name=var1_name, line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df_merged['date'], y=df_merged[var2_code], 
                      name=var2_name, line=dict(color='red', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Economic Indicators Time Series")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.subheader("Recent Data (Last 10 Observations)")
        recent_data = df_merged.tail(10).copy()
        recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)
    
    with tab2:
        st.header("Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {var1_name}")
            stats1 = calculate_descriptive_stats(df_merged[var1_code].dropna(), var1_name)
            stats_df1 = pd.DataFrame(list(stats1.items()), columns=['Statistic', 'Value'])
            stats_df1['Value'] = stats_df1['Value'].round(4)
            st.dataframe(stats_df1, use_container_width=True)
        
        with col2:
            st.subheader(f"📊 {var2_name}")
            stats2 = calculate_descriptive_stats(df_merged[var2_code].dropna(), var2_name)
            stats_df2 = pd.DataFrame(list(stats2.items()), columns=['Statistic', 'Value'])
            stats_df2['Value'] = stats_df2['Value'].round(4)
            st.dataframe(stats_df2, use_container_width=True)
        
        # Box plots comparison
        st.subheader("Box Plot Comparison")
        fig_box = make_subplots(rows=1, cols=2, subplot_titles=(var1_name, var2_name))
        
        fig_box.add_trace(
            go.Box(y=df_merged[var1_code], name=var1_name, marker_color='blue'),
            row=1, col=1
        )
        fig_box.add_trace(
            go.Box(y=df_merged[var2_code], name=var2_name, marker_color='red'),
            row=1, col=2
        )
        
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab3:
        st.header("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Distribution of {var1_name}")
            fig_hist1 = px.histogram(
                df_merged, x=var1_code, nbins=30,
                title=f"Histogram: {var1_name}",
                marginal="box"
            )
            fig_hist1.update_traces(marker_color='blue', opacity=0.7)
            st.plotly_chart(fig_hist1, use_container_width=True)
            
            # Normality test
            shapiro_stat1, shapiro_p1 = stats.shapiro(df_merged[var1_code].dropna()[:5000])  # Limit for shapiro
            st.write(f"**Shapiro-Wilk Normality Test:**")
            st.write(f"Statistic: {shapiro_stat1:.4f}, p-value: {shapiro_p1:.4f}")
            st.write("✅ Normal" if shapiro_p1 > 0.05 else "❌ Not Normal")
        
        with col2:
            st.subheader(f"Distribution of {var2_name}")
            fig_hist2 = px.histogram(
                df_merged, x=var2_code, nbins=30,
                title=f"Histogram: {var2_name}",
                marginal="box"
            )
            fig_hist2.update_traces(marker_color='red', opacity=0.7)
            st.plotly_chart(fig_hist2, use_container_width=True)
            
            # Normality test
            shapiro_stat2, shapiro_p2 = stats.shapiro(df_merged[var2_code].dropna()[:5000])
            st.write(f"**Shapiro-Wilk Normality Test:**")
            st.write(f"Statistic: {shapiro_stat2:.4f}, p-value: {shapiro_p2:.4f}")
            st.write("✅ Normal" if shapiro_p2 > 0.05 else "❌ Not Normal")
        
        # Q-Q plots
        st.subheader("Q-Q Plots (Normal Distribution)")
        fig_qq = make_subplots(rows=1, cols=2, subplot_titles=(f"Q-Q Plot: {var1_name}", f"Q-Q Plot: {var2_name}"))
        
        # Q-Q plot for variable 1
        qq1 = stats.probplot(df_merged[var1_code].dropna(), dist="norm")
        fig_qq.add_trace(
            go.Scatter(x=qq1[0][0], y=qq1[0][1], mode='markers', name=var1_name, marker_color='blue'),
            row=1, col=1
        )
        fig_qq.add_trace(
            go.Scatter(x=qq1[0][0], y=qq1[1][1] + qq1[1][0]*qq1[0][0], mode='lines', 
                      name='Normal Line', line=dict(color='blue', dash='dash')),
            row=1, col=1
        )
        
        # Q-Q plot for variable 2
        qq2 = stats.probplot(df_merged[var2_code].dropna(), dist="norm")
        fig_qq.add_trace(
            go.Scatter(x=qq2[0][0], y=qq2[0][1], mode='markers', name=var2_name, marker_color='red'),
            row=1, col=2
        )
        fig_qq.add_trace(
            go.Scatter(x=qq2[0][0], y=qq2[1][1] + qq2[1][0]*qq2[0][0], mode='lines', 
                      name='Normal Line', line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        fig_qq.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_qq, use_container_width=True)
    
    with tab4:
        st.header("Correlation and Regression Analysis")
        
        # Remove any rows with missing values for correlation analysis
        clean_data = df_merged[[var1_code, var2_code]].dropna()
        
        if len(clean_data) < 2:
            st.error("Not enough data points for correlation analysis.")
            return
        
        # Correlation coefficient
        correlation = clean_data[var1_code].corr(clean_data[var2_code])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Correlation Coefficient", f"{correlation:.4f}")
        col2.metric("R-squared", f"{correlation**2:.4f}")
        
        # Interpret correlation strength
        if abs(correlation) >= 0.7:
            strength = "Strong"
        elif abs(correlation) >= 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        col3.metric("Correlation Strength", strength)
        
        # Scatter plot with regression line
        st.subheader("Scatter Plot with Regression Line")
        
        fig_scatter = px.scatter(
            clean_data, x=var1_code, y=var2_code,
            title=f"Relationship between {var1_name} and {var2_name}"
        )
        
        # Add manual regression line using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[var1_code], clean_data[var2_code])
        x_range = np.linspace(clean_data[var1_code].min(), clean_data[var1_code].max(), 100)
        y_pred = slope * x_range + intercept
        
        fig_scatter.add_trace(
            go.Scatter(x=x_range, y=y_pred, mode='lines', name='Regression Line',
                      line=dict(color='red', width=2))
        )
        
        fig_scatter.update_traces(marker_size=8, marker_opacity=0.6)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Linear regression statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[var1_code], clean_data[var2_code])
        
        st.subheader("Linear Regression Results")
        reg_results = {
            'Slope (β₁)': slope,
            'Intercept (β₀)': intercept,
            'R-value': r_value,
            'R-squared': r_value**2,
            'P-value': p_value,
            'Standard Error': std_err
        }
        
        reg_df = pd.DataFrame(list(reg_results.items()), columns=['Parameter', 'Value'])
        reg_df['Value'] = reg_df['Value'].round(6)
        st.dataframe(reg_df, use_container_width=True)
        
        # Regression equation
        st.write(f"**Regression Equation:**")
        st.latex(f"{var2_name} = {intercept:.4f} + {slope:.4f} \\times {var1_name}")
    
    with tab5:
        st.header("Confidence Intervals")
        
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
        alpha = 1 - confidence_level/100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Confidence Interval for {var1_name}")
            data1 = df_merged[var1_code].dropna()
            mean1 = np.mean(data1)
            sem1 = stats.sem(data1)  # Standard error of mean
            ci1 = stats.t.interval(confidence_level/100, len(data1)-1, loc=mean1, scale=sem1)
            
            st.write(f"**Mean:** {mean1:.4f}")
            st.write(f"**{confidence_level}% Confidence Interval:** [{ci1[0]:.4f}, {ci1[1]:.4f}]")
            st.write(f"**Margin of Error:** ±{(ci1[1]-ci1[0])/2:.4f}")
            
            # Visualization
            fig_ci1 = go.Figure()
            
            # Add normal distribution curve
            x_range = np.linspace(mean1 - 4*sem1*np.sqrt(len(data1)), 
                                mean1 + 4*sem1*np.sqrt(len(data1)), 100)
            y_range = stats.norm.pdf(x_range, mean1, sem1*np.sqrt(len(data1)))
            
            fig_ci1.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', 
                                       name='Distribution', line=dict(color='blue')))
            
            # Add confidence interval
            fig_ci1.add_vline(x=ci1[0], line=dict(color='red', dash='dash'), 
                            annotation_text=f"CI Lower: {ci1[0]:.2f}")
            fig_ci1.add_vline(x=ci1[1], line=dict(color='red', dash='dash'), 
                            annotation_text=f"CI Upper: {ci1[1]:.2f}")
            fig_ci1.add_vline(x=mean1, line=dict(color='green'), 
                            annotation_text=f"Mean: {mean1:.2f}")
            
            fig_ci1.update_layout(title=f"{confidence_level}% Confidence Interval for {var1_name}", 
                                height=300)
            st.plotly_chart(fig_ci1, use_container_width=True)
        
        with col2:
            st.subheader(f"Confidence Interval for {var2_name}")
            data2 = df_merged[var2_code].dropna()
            mean2 = np.mean(data2)
            sem2 = stats.sem(data2)
            ci2 = stats.t.interval(confidence_level/100, len(data2)-1, loc=mean2, scale=sem2)
            
            st.write(f"**Mean:** {mean2:.4f}")
            st.write(f"**{confidence_level}% Confidence Interval:** [{ci2[0]:.4f}, {ci2[1]:.4f}]")
            st.write(f"**Margin of Error:** ±{(ci2[1]-ci2[0])/2:.4f}")
            
            # Visualization
            fig_ci2 = go.Figure()
            
            x_range = np.linspace(mean2 - 4*sem2*np.sqrt(len(data2)), 
                                mean2 + 4*sem2*np.sqrt(len(data2)), 100)
            y_range = stats.norm.pdf(x_range, mean2, sem2*np.sqrt(len(data2)))
            
            fig_ci2.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', 
                                       name='Distribution', line=dict(color='red')))
            
            fig_ci2.add_vline(x=ci2[0], line=dict(color='blue', dash='dash'), 
                            annotation_text=f"CI Lower: {ci2[0]:.2f}")
            fig_ci2.add_vline(x=ci2[1], line=dict(color='blue', dash='dash'), 
                            annotation_text=f"CI Upper: {ci2[1]:.2f}")
            fig_ci2.add_vline(x=mean2, line=dict(color='green'), 
                            annotation_text=f"Mean: {mean2:.2f}")
            
            fig_ci2.update_layout(title=f"{confidence_level}% Confidence Interval for {var2_name}", 
                                height=300)
            st.plotly_chart(fig_ci2, use_container_width=True)
    
    with tab6:
        st.header("📋 Executive Summary Report")
        
        st.markdown(f"""
        ## Statistical Analysis Report
        **Variables Analyzed:** {var1_name} vs {var2_name}  
        **Time Period:** {df_merged['date'].min().strftime('%Y-%m-%d')} to {df_merged['date'].max().strftime('%Y-%m-%d')}  
        **Sample Size:** {len(df_merged)} observations
        
        ### Key Findings:
        
        #### Descriptive Statistics Summary:
        - **{var1_name}**: Mean = {np.mean(df_merged[var1_code]):.2f}, SD = {np.std(df_merged[var1_code]):.2f}
        - **{var2_name}**: Mean = {np.mean(df_merged[var2_code]):.2f}, SD = {np.std(df_merged[var2_code]):.2f}
        
        #### Correlation Analysis:
        - **Correlation Coefficient**: {correlation:.4f}
        - **Relationship Strength**: {strength}
        - **R-squared**: {correlation**2:.4f} ({correlation**2*100:.1f}% of variance explained)
        
        #### Distribution Characteristics:
        - **{var1_name} Normality**: {"Normal" if shapiro_p1 > 0.05 else "Non-normal"} (p = {shapiro_p1:.4f})
        - **{var2_name} Normality**: {"Normal" if shapiro_p2 > 0.05 else "Non-normal"} (p = {shapiro_p2:.4f})
        
        #### Statistical Significance:
        - **Correlation p-value**: {p_value:.6f}
        - **Result**: {"Statistically significant" if p_value < 0.05 else "Not statistically significant"} at α = 0.05
        
        ### Conclusions:
        {f"There is a {strength.lower()} {'positive' if correlation > 0 else 'negative'} correlation between {var1_name} and {var2_name}." if abs(correlation) > 0.1 else f"There is little to no linear relationship between {var1_name} and {var2_name}."}
        
        ---
        *This analysis was generated using Python, Streamlit, and the FRED API - demonstrating the power of modern statistical computing tools.*
        """)
        
        # Download button for data
        csv = df_merged.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"economic_analysis_{var1_code}_{var2_code}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()