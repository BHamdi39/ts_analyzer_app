import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import urllib.request
warnings.filterwarnings('ignore')

# Time series libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from prophet import Prophet
import pmdarima as pm

# PDF generation with Unicode support
from fpdf import FPDF
import plotly.io as pio
import base64
import io
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Professional Time Series Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem;
        background-color: #ecf0f1;
        border-left: 5px solid #3498db;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'time_column' not in st.session_state:
    st.session_state.time_column = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'fitted_model' not in st.session_state:
    st.session_state.fitted_model = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

def main():
    st.markdown('<div class="main-header">📈 Professional Time Series Analysis Suite</div>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox("Choose Analysis Section", [
        "📁 Data Upload & Preprocessing",
        "🔍 Exploratory Data Analysis", 
        "🧪 Statistical Tests",
        "🤖 Time Series Models",
        "🔮 Forecasting & Results",
        "📄 Export & Reporting"
    ])
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ App Info")
    st.sidebar.info("Professional time series analysis with Stata-like functionality")
    
    if st.session_state.data is not None:
        st.sidebar.markdown("### 📊 Current Dataset")
        st.sidebar.write(f"**Shape:** {st.session_state.data.shape}")
        if st.session_state.target_column:
            st.sidebar.write(f"**Target:** {st.session_state.target_column}")
    
    # Route to appropriate section
    if page == "📁 Data Upload & Preprocessing":
        data_upload_section()
    elif page == "🔍 Exploratory Data Analysis":
        eda_section()
    elif page == "🧪 Statistical Tests":
        statistical_tests_section()
    elif page == "🤖 Time Series Models":
        modeling_section()
    elif page == "🔮 Forecasting & Results":
        forecasting_section()
    elif page == "📄 Export & Reporting":
        export_section()

def data_upload_section():
    st.markdown('<div class="section-header">📁 Data Upload & Preprocessing</div>', 
                unsafe_allow_html=True)
    
    # File upload with improved UI
    st.markdown("### Upload Your Time Series Data")
    uploaded_file = st.file_uploader(
        "Choose your data file", 
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel files containing your time series data. Ensure your data has a date column and at least one numeric variable."
    )
    
    if uploaded_file is not None:
        try:
            # Show upload progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Reading file...")
            progress_bar.progress(25)
            
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            progress_bar.progress(50)
            status_text.text("Processing data...")
            
            st.session_state.data = df
            
            progress_bar.progress(100)
            status_text.text("✅ File uploaded successfully!")
            
            st.success(f"✅ File uploaded successfully! **{uploaded_file.name}** - Shape: {df.shape}")
            
            # Data preview with enhanced display
            st.markdown("### 📊 Data Preview")
            with st.expander("Click to view data sample", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Enhanced data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                missing_count = df.isnull().sum().sum()
                st.metric("❓ Missing Values", f"{missing_count:,}")
            with col4:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("🔢 Numeric Columns", numeric_cols)
            
            # Data types information
            st.markdown("### 📋 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Column selection with improved UI
            st.markdown("### ⚙️ Time Series Setup")
            
            col1, col2 = st.columns(2)
            with col1:
                time_col = st.selectbox(
                    "🗓️ Select Time Column",
                    [""] + df.columns.tolist(),
                    help="Choose the column containing dates/time values"
                )
                if time_col:
                    st.session_state.time_column = time_col
                    # Show sample values
                    st.write("**Sample values:**")
                    st.write(df[time_col].head().tolist())
                
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    target_col = st.selectbox(
                        "🎯 Select Target Variable",
                        [""] + numeric_cols,
                        help="Choose the numeric variable you want to analyze"
                    )
                    if target_col:
                        st.session_state.target_column = target_col
                        # Show basic stats
                        st.write("**Basic statistics:**")
                        stats = df[target_col].describe()
                        st.write(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                else:
                    st.error("❌ No numeric columns found for analysis!")
            
            # Data preprocessing with validation
            if st.session_state.time_column and st.session_state.target_column:
                st.markdown("---")
                if st.button("🔧 Preprocess Data", type="primary"):
                    with st.spinner("Processing data..."):
                        try:
                            # Convert time column
                            df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
                            df = df.sort_values(time_col)
                            
                            # Handle duplicates
                            initial_shape = df.shape[0]
                            df = df.drop_duplicates(subset=[time_col])
                            duplicates_removed = initial_shape - df.shape[0]
                            
                            # Set index
                            df.set_index(time_col, inplace=True)
                            
                            # Remove rows with missing target values
                            df = df.dropna(subset=[target_col])
                            
                            st.session_state.data = df
                            
                            st.success("✅ Data preprocessed successfully!")
                            
                            # Show processed data info
                            st.markdown("### 📈 Processed Data Information")
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.write(f"**📅 Date range:** {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
                                st.write(f"**📊 Final shape:** {df.shape}")
                                if duplicates_removed > 0:
                                    st.write(f"**🔄 Duplicates removed:** {duplicates_removed}")
                            
                            with info_col2:
                                try:
                                    freq = pd.infer_freq(df.index)
                                    st.write(f"**⏱️ Inferred frequency:** {freq if freq else 'Irregular'}")
                                except:
                                    st.write(f"**⏱️ Frequency:** Could not infer")
                                
                                missing_target = df[target_col].isnull().sum()
                                st.write(f"**❓ Missing values in target:** {missing_target}")
                                
                        except Exception as e:
                            st.error(f"❌ Preprocessing failed: {str(e)}")
                            st.write("**Troubleshooting tips:**")
                            st.write("- Ensure your time column contains valid date formats")
                            st.write("- Check for special characters in date strings")
                            st.write("- Verify your target column contains numeric values")
                            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.write("**Possible solutions:**")
            st.write("- Check file format (CSV or XLSX)")
            st.write("- Ensure file is not corrupted")
            st.write("- Verify file permissions")

def eda_section():
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload and preprocess data first!")
        return
    
    df = st.session_state.data
    target_col = st.session_state.target_column
    
    if target_col not in df.columns:
        st.error("❌ Target column not found. Please reconfigure in Data Upload section.")
        return
    
    # Summary statistics with enhanced display
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    stats = df[target_col].describe()
    with col1:
        st.metric("📈 Mean", f"{stats['mean']:.4f}")
    with col2:
        st.metric("📏 Std Dev", f"{stats['std']:.4f}")
    with col3:
        st.metric("📉 Min", f"{stats['min']:.4f}")
    with col4:
        st.metric("📊 Max", f"{stats['max']:.4f}")
    with col5:
        st.metric("🎯 Count", f"{stats['count']:.0f}")
    
    # Time series plot with enhanced features
    st.markdown("### 📈 Time Series Visualization")
    
    # Plot options
    plot_col1, plot_col2 = st.columns([3, 1])
    
    with plot_col2:
        show_trend = st.checkbox("Show trend line", False)
        show_points = st.checkbox("Show data points", False)
        plot_height = st.slider("Plot height", 400, 800, 500)
    
    with plot_col1:
        fig = px.line(df, x=df.index, y=target_col, 
                      title=f"Time Series of {target_col}")
        
        if show_points:
            fig.update_traces(mode='lines+markers')
        
        if show_trend:
            # Add trend line
            from scipy import stats
            x_numeric = np.arange(len(df))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df[target_col])
            trend_line = slope * x_numeric + intercept
            
            fig.add_trace(go.Scatter(
                x=df.index, y=trend_line,
                mode='lines', name='Trend',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            xaxis_title="Time", 
            yaxis_title=target_col,
            height=plot_height,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribution Analysis")
        bins = st.slider("Number of bins", 10, 50, 30, key="hist_bins")
        fig = px.histogram(df, x=target_col, nbins=bins, 
                          title="Distribution of Values")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📅 Seasonal Patterns")
        # Create monthly boxplot if enough data
        if len(df) > 24:  # At least 2 years of monthly data
            df_copy = df.copy()
            df_copy['Month'] = df_copy.index.month
            fig = px.box(df_copy, x='Month', y=target_col, 
                        title="Monthly Distribution")
            fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for seasonal analysis")
    
    # Time series decomposition with error handling
    st.markdown("### 🔬 Time Series Decomposition")
    
    try:
        if len(df) >= 24:  # Need at least 2 seasonal periods
            # Determine appropriate period
            decomp_period = st.selectbox(
                "Decomposition period",
                [12, 4, 52, 7],
                help="Choose seasonal period: 12=monthly, 4=quarterly, 52=weekly, 7=daily"
            )
            
            decomposition = seasonal_decompose(
                df[target_col].dropna(), 
                model='additive', 
                period=min(decomp_period, len(df)//2)
            )
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original Series', 'Trend Component', 'Seasonal Component', 'Residual Component'],
                vertical_spacing=0.08
            )
            
            # Original series
            fig.add_trace(go.Scatter(
                x=df.index, y=df[target_col], 
                name='Original', line=dict(color='blue')
            ), row=1, col=1)
            
            # Trend
            fig.add_trace(go.Scatter(
                x=df.index, y=decomposition.trend, 
                name='Trend', line=dict(color='red')
            ), row=2, col=1)
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=df.index, y=decomposition.seasonal, 
                name='Seasonal', line=dict(color='green')
            ), row=3, col=1)
            
            # Residual
            fig.add_trace(go.Scatter(
                x=df.index, y=decomposition.resid, 
                name='Residual', line=dict(color='orange')
            ), row=4, col=1)
            
            fig.update_layout(height=800, title_text="Time Series Decomposition", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Decomposition summary
            st.markdown("#### Decomposition Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_change = (decomposition.trend.dropna().iloc[-1] - decomposition.trend.dropna().iloc[0])
                st.metric("📈 Overall Trend", f"{trend_change:.2f}")
            
            with col2:
                seasonal_strength = decomposition.seasonal.std()
                st.metric("🔄 Seasonal Strength", f"{seasonal_strength:.2f}")
            
            with col3:
                residual_std = decomposition.resid.std()
                st.metric("📊 Residual Std", f"{residual_std:.2f}")
                
        else:
            st.warning("⚠️ Need at least 24 observations for reliable decomposition")
            
    except Exception as e:
        st.error(f"❌ Decomposition failed: {str(e)}")
        st.write("**Possible reasons:**")
        st.write("- Insufficient data points")
        st.write("- Too many missing values")
        st.write("- Inappropriate seasonal period")

def statistical_tests_section():
    st.markdown('<div class="section-header">🧪 Statistical Tests</div>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload and preprocess data first!")
        return
    
    df = st.session_state.data
    target_col = st.session_state.target_column
    
    if target_col not in df.columns:
        st.error("❌ Target column not found.")
        return
    
    data_clean = df[target_col].dropna()
    
    if len(data_clean) < 10:
        st.error("❌ Not enough data points for statistical tests (minimum 10 required)")
        return
    
    # Unit root tests with enhanced display
    st.markdown("### 🧪 Stationarity Tests")
    st.write("**Testing whether your time series is stationary (constant mean and variance over time)**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Augmented Dickey-Fuller Test")
        st.write("**H₀:** Series has unit root (non-stationary)")
        st.write("**H₁:** Series is stationary")
        
        try:
            adf_result = adfuller(data_clean, autolag='AIC')
            
            # Create results dataframe
            adf_df = pd.DataFrame({
                'Metric': ['Test Statistic', 'p-value', 'Lags Used', 'Observations'],
                'Value': [f"{adf_result[0]:.4f}", f"{adf_result[1]:.4f}", 
                         adf_result[2], adf_result[3]]
            })
            
            st.dataframe(adf_df, use_container_width=True)
            
            # Critical values
            st.write("**Critical Values:**")
            crit_vals = pd.DataFrame({
                'Significance Level': ['1%', '5%', '10%'],
                'Critical Value': [f"{adf_result[4]['1%']:.4f}", 
                                 f"{adf_result[4]['5%']:.4f}", 
                                 f"{adf_result[4]['10%']:.4f}"]
            })
            st.dataframe(crit_vals, use_container_width=True)
            
            # Interpretation
            if adf_result[1] <= 0.05:
                st.success("✅ **Conclusion:** Series is stationary (reject H₀)")
            else:
                st.warning("⚠️ **Conclusion:** Series is non-stationary (fail to reject H₀)")
                
        except Exception as e:
            st.error(f"❌ ADF test failed: {str(e)}")
    
    with col2:
        st.markdown("#### 📊 KPSS Test")
        st.write("**H₀:** Series is stationary")
        st.write("**H₁:** Series has unit root (non-stationary)")
        
        try:
            kpss_result = kpss(data_clean, regression='c', nlags='auto')
            
            kpss_df = pd.DataFrame({
                'Metric': ['Test Statistic', 'p-value', 'Lags Used'],
                'Value': [f"{kpss_result[0]:.4f}", f"{kpss_result[1]:.4f}", kpss_result[2]]
            })
            
            st.dataframe(kpss_df, use_container_width=True)
            
            # Critical values
            st.write("**Critical Values:**")
            kpss_crit = pd.DataFrame({
                'Significance Level': ['10%', '5%', '2.5%', '1%'],
                'Critical Value': [f"{kpss_result[3]['10%']:.4f}",
                                 f"{kpss_result[3]['5%']:.4f}",
                                 f"{kpss_result[3]['2.5%']:.4f}",
                                 f"{kpss_result[3]['1%']:.4f}"]
            })
            st.dataframe(kpss_crit, use_container_width=True)
            
            # Interpretation
            if kpss_result[1] <= 0.05:
                st.warning("⚠️ **Conclusion:** Series is non-stationary (reject H₀)")
            else:
                st.success("✅ **Conclusion:** Series is stationary (fail to reject H₀)")
                
        except Exception as e:
            st.error(f"❌ KPSS test failed: {str(e)}")
    
    # Combined interpretation
    st.markdown("---")
    st.markdown("#### 🎯 Combined Test Interpretation")
    
    try:
        adf_stationary = adfuller(data_clean)[1] <= 0.05
        kpss_stationary = kpss(data_clean, regression='c')[1] > 0.05
        
        if adf_stationary and kpss_stationary:
            st.success("🎉 **Both tests agree:** Series is stationary")
        elif not adf_stationary and not kpss_stationary:
            st.error("🚨 **Both tests agree:** Series is non-stationary - consider differencing")
        else:
            st.warning("⚠️ **Tests disagree:** Results are inconclusive - manual inspection recommended")
            
    except Exception as e:
        st.write("Could not perform combined interpretation")
    
    # Autocorrelation analysis with enhanced plots
    st.markdown("### 🔄 Autocorrelation Analysis")
    
    max_lags = min(40, len(data_clean)//4)
    selected_lags = st.slider("Number of lags to display", 5, max_lags, min(20, max_lags))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Autocorrelation Function (ACF)")
        try:
            acf_vals, acf_confint = acf(data_clean, nlags=selected_lags, alpha=0.05)
            
            # Create ACF plot
            fig = go.Figure()
            
            # ACF bars
            fig.add_trace(go.Bar(
                x=list(range(len(acf_vals))),
                y=acf_vals,
                name='ACF',
                marker_color='blue'
            ))
            
            # Confidence intervals
            confidence_level = 1.96/np.sqrt(len(data_clean))
            fig.add_hline(y=confidence_level, line_dash="dash", line_color="red", 
                         annotation_text="95% Confidence")
            fig.add_hline(y=-confidence_level, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_color="black")
            
            fig.update_layout(
                title="Autocorrelation Function", 
                xaxis_title="Lag", 
                yaxis_title="ACF",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"ACF calculation failed: {str(e)}")
    
    with col2:
        st.markdown("#### 📈 Partial Autocorrelation Function (PACF)")
        try:
            pacf_vals, pacf_confint = pacf(data_clean, nlags=selected_lags, alpha=0.05)
            
            # Create PACF plot
            fig = go.Figure()
            
            # PACF bars
            fig.add_trace(go.Bar(
                x=list(range(len(pacf_vals))),
                y=pacf_vals,
                name='PACF',
                marker_color='green'
            ))
            
            # Confidence intervals
            confidence_level = 1.96/np.sqrt(len(data_clean))
            fig.add_hline(y=confidence_level, line_dash="dash", line_color="red",
                         annotation_text="95% Confidence")
            fig.add_hline(y=-confidence_level, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_color="black")
            
            fig.update_layout(
                title="Partial Autocorrelation Function", 
                xaxis_title="Lag", 
                yaxis_title="PACF",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"PACF calculation failed: {str(e)}")
    
    # Additional diagnostics
    st.markdown("### 📊 Additional Diagnostics")
    
    # Ljung-Box test for white noise
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box = acorr_ljungbox(data_clean, lags=10, return_df=True)
        
        st.markdown("#### 🎲 Ljung-Box Test (White Noise Test)")
        st.write("**H₀:** Residuals are independently distributed (white noise)")
        st.write("**H₁:** Residuals are not independently distributed")
        
        st.dataframe(ljung_box[['lb_stat', 'lb_pvalue']].head(), use_container_width=True)
        
        if any(ljung_box['lb_pvalue'] < 0.05):
            st.warning("⚠️ Evidence against white noise hypothesis")
        else:
            st.success("✅ Residuals appear to be white noise")
            
    except Exception as e:
        st.write(f"Could not perform Ljung-Box test: {str(e)}")

def modeling_section():
    st.markdown('<div class="section-header">🤖 Time Series Models</div>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload and preprocess data first!")
        return
    
    df = st.session_state.data
    target_col = st.session_state.target_column
    
    if target_col not in df.columns:
        st.error("❌ Target column not found.")
        return
    
    data_clean = df[target_col].dropna()
    
    if len(data_clean) < 10:
        st.error("❌ Not enough data points for modeling (minimum 10 required)")
        return
    
    # Model selection with enhanced UI
    st.markdown("### 🎯 Model Selection")
    
    model_type = st.selectbox(
        "Choose Model Type",
        ["Auto ARIMA", "Manual ARIMA", "Prophet", "Exponential Smoothing"],
        help="Select the time series model you want to fit to your data"
    )
    
    # Train/test split option
    st.markdown("### 📊 Data Splitting")
    use_split = st.checkbox("Use train/test split for validation", True)
    
    if use_split:
        split_ratio = st.slider("Training data percentage", 60, 95, 80)
        split_point = int(len(data_clean) * split_ratio / 100)
        train_data = data_clean[:split_point]
        test_data = data_clean[split_point:]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Training samples", len(train_data))
        with col2:
            st.metric("🧪 Testing samples", len(test_data))
    else:
        train_data = data_clean
        test_data = None
    
    # Model fitting based on selection
    if model_type == "Auto ARIMA":
        st.markdown("### 🚀 Auto ARIMA Model")
        st.write("Automatically selects optimal ARIMA parameters using information criteria")
        
        # Auto ARIMA configuration
        col1, col2 = st.columns(2)
        with col1:
            seasonal = st.checkbox("Include seasonal components", True)
            max_p = st.number_input("Max AR order (p)", 1, 5, 3)
            max_q = st.number_input("Max MA order (q)", 1, 5, 3)
            
        with col2:
            max_d = st.number_input("Max differencing (d)", 0, 2, 2)
            if seasonal:
                max_P = st.number_input("Max seasonal AR (P)", 0, 2, 1)
                max_Q = st.number_input("Max seasonal MA (Q)", 0, 2, 1)
                seasonal_period = st.number_input("Seasonal period", 2, 52, 12)
        
        if st.button("🚀 Fit Auto ARIMA", type="primary"):
            with st.spinner("Fitting Auto ARIMA model... This may take a few minutes."):
                try:
                    # Configure auto_arima parameters
                    arima_params = {
                        'seasonal': seasonal,
                        'stepwise': True,
                        'suppress_warnings': True,
                        'error_action': 'ignore',
                        'max_p': max_p,
                        'max_q': max_q,
                        'max_d': max_d,
                        'trace': True
                    }
                    
                    if seasonal:
                        arima_params.update({
                            'max_P': max_P,
                            'max_Q': max_Q,
                            'max_D': 1,
                            'seasonal_period': seasonal_period
                        })
                    
                    model = pm.auto_arima(train_data, **arima_params)
                    
                    st.session_state.fitted_model = model
                    st.session_state.model_type = "auto_arima"
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    st.success(f"✅ Model fitted successfully!")
                    
                    # Display model information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Model Order", f"ARIMA{model.order}")
                        if seasonal:
                            st.metric("🔄 Seasonal Order", f"{model.seasonal_order}")
                    with col2:
                        st.metric("📈 AIC", f"{model.aic():.2f}")
                        st.metric("📉 BIC", f"{model.bic():.2f}")
                    
                    # Model summary
                    st.markdown("#### 📋 Model Summary")
                    with st.expander("View detailed model summary"):
                        st.text(str(model.summary()))
                    
                    # Residual analysis
                    residuals = model.resid()
                    plot_residual_analysis(residuals)
                    
                except Exception as e:
                    st.error(f"❌ Auto ARIMA fitting failed: {str(e)}")
                    st.write("**Troubleshooting tips:**")
                    st.write("- Try reducing max_p, max_q parameters")
                    st.write("- Disable seasonal components if data is not seasonal")
                    st.write("- Ensure sufficient data points")
    
    elif model_type == "Manual ARIMA":
        st.markdown("### ⚙️ Manual ARIMA Model")
        st.write("Specify ARIMA parameters manually for precise control")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR Order (p)", min_value=0, max_value=10, value=1,
                               help="Number of autoregressive terms")
        with col2:
            d = st.number_input("Differencing (d)", min_value=0, max_value=3, value=1,
                               help="Number of differences needed for stationarity")
        with col3:
            q = st.number_input("MA Order (q)", min_value=0, max_value=10, value=1,
                               help="Number of moving average terms")
        
        # Seasonal ARIMA option
        seasonal_arima = st.checkbox("Include seasonal components")
        if seasonal_arima:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                P = st.number_input("Seasonal AR (P)", 0, 3, 1)
            with col2:
                D = st.number_input("Seasonal Diff (D)", 0, 2, 1)
            with col3:
                Q = st.number_input("Seasonal MA (Q)", 0, 3, 1)
            with col4:
                s = st.number_input("Seasonal Period (s)", 2, 52, 12)
        
        if st.button("🚀 Fit Manual ARIMA", type="primary"):
            with st.spinner("Fitting ARIMA model..."):
                try:
                    if seasonal_arima:
                        model = ARIMA(train_data, order=(p, d, q), seasonal_order=(P, D, Q, s))
                    else:
                        model = ARIMA(train_data, order=(p, d, q))
                    
                    fitted_model = model.fit()
                    
                    st.session_state.fitted_model = fitted_model
                    st.session_state.model_type = "manual_arima"
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    order_str = f"({p},{d},{q})"
                    if seasonal_arima:
                        order_str += f"x({P},{D},{Q},{s})"
                    
                    st.success(f"✅ ARIMA{order_str} model fitted successfully!")
                    
                    # Model metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📈 AIC", f"{fitted_model.aic:.2f}")
                        st.metric("📉 BIC", f"{fitted_model.bic:.2f}")
                    with col2:
                        st.metric("📊 Log-Likelihood", f"{fitted_model.llf:.2f}")
                        st.metric("🎯 Parameters", len(fitted_model.params))
                    
                    # Model summary
                    st.markdown("#### 📋 Model Summary")
                    with st.expander("View detailed model summary"):
                        st.text(str(fitted_model.summary()))
                    
                    # Residual analysis
                    plot_residual_analysis(fitted_model.resid)
                    
                except Exception as e:
                    st.error(f"❌ Manual ARIMA fitting failed: {str(e)}")
                    st.write("**Common issues:**")
                    st.write("- Model parameters may not be suitable for this data")
                    st.write("- Try different (p,d,q) combinations")
                    st.write("- Check if data needs more/less differencing")
    
    elif model_type == "Prophet":
        st.markdown("### 🔮 Prophet Model")
        st.write("Facebook's robust forecasting algorithm for business time series")
        
        # Prophet configuration
        col1, col2 = st.columns(2)
        with col1:
            yearly_seasonality = st.selectbox("Yearly seasonality", [True, False, "auto"], index=0)
            weekly_seasonality = st.selectbox("Weekly seasonality", [True, False, "auto"], index=2)
        with col2:
            daily_seasonality = st.selectbox("Daily seasonality", [True, False, "auto"], index=1)
            changepoint_prior_scale = st.slider("Changepoint prior scale", 0.001, 0.5, 0.05,
                                               help="Higher values allow more flexible trend changes")
        
        if st.button("🚀 Fit Prophet Model", type="primary"):
            with st.spinner("Fitting Prophet model..."):
                try:
                    # Prepare data for Prophet
                    prophet_df = pd.DataFrame({
                        'ds': train_data.index,
                        'y': train_data.values
                    })
                    
                    # Create and configure model
                    model = Prophet(
                        yearly_seasonality=yearly_seasonality,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=daily_seasonality,
                        changepoint_prior_scale=changepoint_prior_scale
                    )
                    
                    model.fit(prophet_df)
                    
                    st.session_state.fitted_model = model
                    st.session_state.model_type = "prophet"
                    st.session_state.prophet_df = prophet_df
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    st.success("✅ Prophet model fitted successfully!")
                    
                    # Model insights
                    st.markdown("#### 📊 Model Components")
                    
                    # Make prediction for current data to show components
                    future = model.make_future_dataframe(periods=0, freq='D')
                    forecast = model.predict(future)
                    
                    # Plot components
                    try:
                        import matplotlib.pyplot as plt
                        fig = model.plot_components(forecast)
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.write(f"Could not display components: {str(e)}")
                    
                    # Model performance metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        mape = np.mean(np.abs((train_data.values - forecast['yhat'][:len(train_data)]) / train_data.values)) * 100
                        st.metric("📊 MAPE (%)", f"{mape:.2f}")
                    with col2:
                        mae = np.mean(np.abs(train_data.values - forecast['yhat'][:len(train_data)]))
                        st.metric("📈 MAE", f"{mae:.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Prophet model fitting failed: {str(e)}")
                    st.write("**Possible solutions:**")
                    st.write("- Ensure your data has a datetime index")
                    st.write("- Check for sufficient data points")
                    st.write("- Try adjusting seasonality parameters")
    
    elif model_type == "Exponential Smoothing":
        st.markdown("### 📈 Exponential Smoothing")
        st.write("Classic forecasting method with trend and seasonal components")
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Exponential Smoothing configuration
        col1, col2 = st.columns(2)
        with col1:
            trend = st.selectbox("Trend component", [None, "add", "mul"], 
                                help="None=no trend, add=additive, mul=multiplicative")
            seasonal = st.selectbox("Seasonal component", [None, "add", "mul"],
                                   help="None=no seasonal, add=additive, mul=multiplicative")
        with col2:
            if seasonal:
                seasonal_periods = st.number_input("Seasonal periods", 2, 52, 12)
            else:
                seasonal_periods = None
            damped = st.checkbox("Damped trend", False)
        
        if st.button("🚀 Fit Exponential Smoothing", type="primary"):
            with st.spinner("Fitting Exponential Smoothing model..."):
                try:
                    model = ExponentialSmoothing(
                        train_data,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods,
                        damped_trend=damped
                    )
                    
                    fitted_model = model.fit(optimized=True)
                    
                    st.session_state.fitted_model = fitted_model
                    st.session_state.model_type = "exp_smoothing"
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    st.success("✅ Exponential Smoothing model fitted successfully!")
                    
                    # Model parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📈 AIC", f"{fitted_model.aic:.2f}")
                        st.metric("📉 BIC", f"{fitted_model.bic:.2f}")
                    with col2:
                        st.metric("🎯 Alpha (level)", f"{fitted_model.params['smoothing_level']:.3f}")
                        if trend:
                            st.metric("📊 Beta (trend)", f"{fitted_model.params['smoothing_trend']:.3f}")
                    
                    # Model summary
                    st.markdown("#### 📋 Model Summary")
                    with st.expander("View model parameters"):
                        st.write(fitted_model.params)
                    
                    # Residual analysis
                    plot_residual_analysis(fitted_model.resid)
                    
                except Exception as e:
                    st.error(f"❌ Exponential Smoothing fitting failed: {str(e)}")
                    st.write("**Troubleshooting:**")
                    st.write("- Try different trend/seasonal combinations")
                    st.write("- Ensure seasonal_periods matches your data frequency")
                    st.write("- Check for sufficient data points")

def plot_residual_analysis(residuals):
    """Plot residual analysis for model diagnostics"""
    st.markdown("#### 🔍 Residual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals time series
        fig = px.line(x=range(len(residuals)), y=residuals, 
                      title="Residuals Over Time")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title="Time", yaxis_title="Residuals")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals distribution
        fig = px.histogram(x=residuals, nbins=30, title="Residual Distribution")
        fig.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Mean", f"{np.mean(residuals):.4f}")
    with col2:
        st.metric("📏 Std Dev", f"{np.std(residuals):.4f}")
    with col3:
        # Jarque-Bera test for normality
        from scipy import stats
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        if jb_pvalue > 0.05:
            st.metric("✅ Normality", "Normal")
        else:
            st.metric("⚠️ Normality", "Non-normal")

def forecasting_section():
    st.markdown('<div class="section-header">🔮 Forecasting & Results</div>', 
                unsafe_allow_html=True)
    
    if 'fitted_model' not in st.session_state:
        st.warning("⚠️ Please fit a model first in the Time Series Models section!")
        return
    
    model = st.session_state.fitted_model
    model_type = st.session_state.model_type
    
    # Forecast parameters with enhanced UI
    st.markdown("### 🎯 Forecast Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_periods = st.number_input("Number of periods to forecast", 
                                         min_value=1, max_value=100, value=12)
    with col2:
        confidence_level = st.selectbox("Confidence level", [80, 90, 95, 99], index=2)
    
    # Forecast frequency
    freq_options = {
        'Daily': 'D',
        'Weekly': 'W', 
        'Monthly': 'M',
        'Quarterly': 'Q',
        'Yearly': 'Y'
    }
    forecast_freq = st.selectbox("Forecast frequency", list(freq_options.keys()), index=2)
    
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                if model_type in ["auto_arima", "manual_arima"]:
                    # ARIMA forecasting
                    forecast = model.forecast(steps=forecast_periods)
                    
                    if hasattr(model, 'get_forecast'):
                        forecast_obj = model.get_forecast(steps=forecast_periods)
                        conf_int = forecast_obj.conf_int(alpha=(100-confidence_level)/100)
                    else:
                        # For pmdarima auto_arima
                        forecast, conf_int = model.predict(n_periods=forecast_periods, 
                                                         return_conf_int=True, 
                                                         alpha=(100-confidence_level)/100)
                    
                    # Create forecast dates
                    last_date = st.session_state.train_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.DateOffset(1), 
                        periods=forecast_periods, 
                        freq=freq_options[forecast_freq]
                    )
                    
                    # Create forecast dataframe
                    if isinstance(conf_int, np.ndarray):
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecast,
                            'Lower CI': conf_int[:, 0],
                            'Upper CI': conf_int[:, 1]
                        })
                    else:
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast': forecast,
                            'Lower CI': conf_int.iloc[:, 0],
                            'Upper CI': conf_int.iloc[:, 1]
                        })
                    
                    # Plot forecast
                    plot_forecast(st.session_state.train_data, st.session_state.test_data, 
                                forecast_df, confidence_level)
                
                elif model_type == "prophet":
                    # Prophet forecasting
                    future = model.make_future_dataframe(
                        periods=forecast_periods, 
                        freq=freq_options[forecast_freq]
                    )
                    forecast = model.predict(future)
                    
                    # Extract forecast for new periods only
                    forecast_subset = forecast.tail(forecast_periods)
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_subset['ds'],
                        'Forecast': forecast_subset['yhat'],
                        'Lower CI': forecast_subset['yhat_lower'],
                        'Upper CI': forecast_subset['yhat_upper']
                    })
                    
                    # Plot Prophet forecast
                    try:
                        fig = model.plot(forecast)
                        plt.title("Prophet Forecast")
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.write(f"Could not display Prophet plot: {str(e)}")
                    
                    plot_forecast(st.session_state.train_data, st.session_state.test_data, 
                                forecast_df, confidence_level, model_type="prophet")
                
                elif model_type == "exp_smoothing":
                    # Exponential Smoothing forecasting
                    forecast = model.forecast(steps=forecast_periods)
                    
                    # Generate forecast dates
                    last_date = st.session_state.train_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.DateOffset(1),
                        periods=forecast_periods,
                        freq=freq_options[forecast_freq]
                    )
                    
                    # Simple confidence intervals (approximate)
                    std_error = np.std(model.resid)
                    z_score = {80: 1.28, 90: 1.64, 95: 1.96, 99: 2.58}[confidence_level]
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast': forecast,
                        'Lower CI': forecast - z_score * std_error,
                        'Upper CI': forecast + z_score * std_error
                    })
                    
                    plot_forecast(st.session_state.train_data, st.session_state.test_data,
                                forecast_df, confidence_level)
                
                st.session_state.forecast_results = forecast_df
                
                # Display forecast table
                st.markdown("### 📊 Forecast Results")
                st.dataframe(forecast_df.round(4), use_container_width=True)
                
                # Forecast summary statistics
                display_forecast_summary(forecast_df)
                
                # Model validation if test data exists
                if st.session_state.test_data is not None and len(st.session_state.test_data) > 0:
                    validate_forecast(model, st.session_state.test_data, model_type)
                
            except Exception as e:
                st.error(f"❌ Forecasting failed: {str(e)}")
                st.write("**Possible solutions:**")
                st.write("- Check model fitting was successful")
                st.write("- Try smaller forecast horizon")
                st.write("- Verify data frequency settings")

def plot_forecast(train_data, test_data, forecast_df, confidence_level, model_type="arima"):
    """Create comprehensive forecast visualization"""
    st.markdown("### 📈 Forecast Visualization")
    
    fig = go.Figure()
    
    # Historical training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data.values,
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=2)
    ))
    
    # Test data (if available)
    if test_data is not None and len(test_data) > 0:
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Test Data',
            line=dict(color='green', width=2)
        ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['Upper CI'].tolist() + forecast_df['Lower CI'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{confidence_level}% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"Time Series Forecast ({model_type.upper()})",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        height=600,
        legend=dict(x=0.01, y=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_forecast_summary(forecast_df):
    """Display forecast summary statistics"""
    st.markdown("### 📊 Forecast Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Mean Forecast", f"{forecast_df['Forecast'].mean():.2f}")
    with col2:
        st.metric("📏 Forecast Std", f"{forecast_df['Forecast'].std():.2f}")
    with col3:
        st.metric("📉 Min Forecast", f"{forecast_df['Forecast'].min():.2f}")
    with col4:
        st.metric("📊 Max Forecast", f"{forecast_df['Forecast'].max():.2f}")
    
    # Forecast trend analysis
    if len(forecast_df) > 1:
        first_value = forecast_df['Forecast'].iloc[0]
        last_value = forecast_df['Forecast'].iloc[-1]
        trend = "📈 Increasing" if last_value > first_value else "📉 Decreasing"
        change_pct = ((last_value - first_value) / first_value) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔄 Forecast Trend", trend)
        with col2:
            st.metric("📊 Total Change (%)", f"{change_pct:.1f}%")

def validate_forecast(model, test_data, model_type):
    """Validate forecast accuracy against test data"""
    st.markdown("### ✅ Model Validation")
    
    try:
        if model_type in ["auto_arima", "manual_arima"]:
            # Generate predictions for test period
            if hasattr(model, 'predict'):
                predictions = model.predict(n_periods=len(test_data))
            else:
                predictions = model.forecast(steps=len(test_data))
        
        elif model_type == "prophet":
            # Prophet validation
            future = model.make_future_dataframe(periods=len(test_data), freq='D')
            forecast = model.predict(future)
            predictions = forecast['yhat'].tail(len(test_data)).values
        
        elif model_type == "exp_smoothing":
            predictions = model.forecast(steps=len(test_data))
        
        # Calculate accuracy metrics
        mae = np.mean(np.abs(test_data.values - predictions))
        mse = np.mean((test_data.values - predictions) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 MAE", f"{mae:.3f}")
        with col2:
            st.metric("📈 RMSE", f"{rmse:.3f}")
        with col3:
            st.metric("📉 MAPE (%)", f"{mape:.2f}%")
        with col4:
            # Calculate R-squared
            ss_res = np.sum((test_data.values - predictions) ** 2)
            ss_tot = np.sum((test_data.values - np.mean(test_data.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            st.metric("🎯 R²", f"{r_squared:.3f}")
        
        # Plot actual vs predicted
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=predictions,
            mode='lines+markers', 
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Model Validation: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Validation failed: {str(e)}")

def export_section():
    st.markdown('<div class="section-header">📄 Export & Reporting</div>', 
                unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please complete the analysis first!")
        return
    
    st.markdown("### 📊 Export Options")
    
    # Export forecast data
    if 'forecast_results' in st.session_state and st.session_state.forecast_results is not None:
        st.markdown("#### 📈 Forecast Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Forecast Data (CSV)", type="secondary"):
                csv = st.session_state.forecast_results.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV File",
                    data=csv,
                    file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="csv_download"
                )
        
        with col2:
            if st.button("📈 Download Forecast Data (Excel)", type="secondary"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    st.session_state.forecast_results.to_excel(writer, sheet_name='Forecast', index=False)
                    if st.session_state.data is not None:
                        st.session_state.data.to_excel(writer, sheet_name='Original_Data')
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=buffer.getvalue(),
                    file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.ms-excel",
                    key="excel_download"
                )
    
    # PDF Report Generation with enhanced options
    st.markdown("#### 📄 PDF Report Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", "Time Series Analysis Report")
        author_name = st.text_input("Author", "Data Analyst")
        
    with col2:
        include_forecast = st.checkbox("Include Forecast Results", True)
        include_diagnostics = st.checkbox("Include Model Diagnostics", True)
        include_data_summary = st.checkbox("Include Data Summary", True)
        include_methodology = st.checkbox("Include Methodology", True)
    
    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Generating comprehensive PDF report..."):
            try:
                # Generate PDF with all options
                pdf_data = generate_pdf_report(
                    title=report_title,
                    author=author_name,
                    include_forecast=include_forecast,
                    include_diagnostics=include_diagnostics,
                    include_data_summary=include_data_summary,
                    include_methodology=include_methodology
                )
                
                st.success("✅ PDF report generated successfully!")
                
                # Download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"time_series_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )
                
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")
                
                # Enhanced error diagnosis
                error_msg = str(e).lower()
                if 'charmap' in error_msg or 'codec' in error_msg:
                    st.write("**Unicode encoding issue detected:**")
                    st.write("- Your data contains special characters that cannot be encoded")
                    st.write("- The system is attempting to use Unicode-compatible fonts")
                    st.write("- If the issue persists, try simplifying text content")
                elif 'font' in error_msg:
                    st.write("**Font loading issue:**")
                    st.write("- Unable to load Unicode font from internet")
                    st.write("- Falling back to basic ASCII character support")
                else:
                    st.write("**General PDF generation error:**")
                    st.write("- Check your data for any unusual characters")
                    st.write("- Try reducing the amount of content included")
                
                st.write("**Troubleshooting steps:**")
                st.write("1. Ensure stable internet connection for font download")
                st.write("2. Try unchecking some report sections")
                st.write("3. Check data for special characters")

def generate_pdf_report(title, author, include_forecast, include_diagnostics, include_data_summary, include_methodology):
    """Generate a comprehensive PDF report with Unicode support and error handling"""
    
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.setup_unicode_font()
        
        def setup_unicode_font(self):
            """Download and setup DejaVu font for Unicode support"""
            font_path = "DejaVuSansCondensed.ttf"
            
            # Try to download DejaVu font if not exists
            if not os.path.exists(font_path):
                try:
                    url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSansCondensed.ttf"
                    urllib.request.urlretrieve(url, font_path)
                except Exception as e:
                    print(f"Could not download Unicode font: {e}")
                    self.unicode_available = False
                    return
            
            # Add Unicode font
            try:
                self.add_font('DejaVu', '', font_path, uni=True)
                self.add_font('DejaVu', 'B', font_path, uni=True)  # Bold variant
                self.unicode_available = True
            except Exception as e:
                print(f"Could not load Unicode font: {e}")
                self.unicode_available = False
        
        def header(self):
            # Set font for header
            if hasattr(self, 'unicode_available') and self.unicode_available:
                self.set_font('DejaVu', 'B', 16)
            else:
                self.set_font('Arial', 'B', 16)
            
            # Clean and add title
            clean_title = self.clean_text(title)
            self.cell(0, 10, clean_title, 0, 1, 'C')
            
            # Add author and date
            self.set_font('DejaVu', '', 10) if self.unicode_available else self.set_font('Arial', '', 10)
            self.cell(0, 5, f'By: {self.clean_text(author)}', 0, 1, 'C')
            self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
            self.ln(10)
        
        def footer(self):
            self.set_y(-15)
            if hasattr(self, 'unicode_available') and self.unicode_available:
                self.set_font('DejaVu', '', 8)
            else:
                self.set_font('Arial', '', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        def clean_text(self, text):
            """Clean text to remove problematic Unicode characters"""
            if not isinstance(text, str):
                text = str(text)
            
            # Replace common problematic characters
            replacements = {
                '"': '"', '"': '"',    # Smart quotes → regular quotes
                ''': "'", ''': "'",    # Smart apostrophes → regular apostrophes
                '—': '-', '–': '-',    # Em/en dashes → hyphens
                '…': '...',            # Ellipsis → three dots
                '°': ' degrees',       # Degree symbol
                '±': '+/-',            # Plus-minus symbol
                '×': 'x', '÷': '/',   # Math symbols
                '€': 'EUR', '£': 'GBP', '¥': 'JPY',  # Currency symbols
                'α': 'alpha', 'β': 'beta', 'γ': 'gamma',  # Greek letters
                '²': '^2', '³': '^3',  # Superscripts
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            # If no Unicode font available, remove remaining non-ASCII characters
            if not (hasattr(self, 'unicode_available') and self.unicode_available):
                text = ''.join(char if ord(char) < 128 else '?' for char in text)
            
            return text
        
        def add_section_header(self, header_text):
            """Add a section header with consistent formatting"""
            self.ln(10)
            if self.unicode_available:
                self.set_font('DejaVu', 'B', 14)
            else:
                self.set_font('Arial', 'B', 14)
            self.cell(0, 10, self.clean_text(header_text), 0, 1, 'L')
            self.ln(5)
        
        def add_paragraph(self, text):
            """Add a paragraph with proper text cleaning"""
            if self.unicode_available:
                self.set_font('DejaVu', '', 10)
            else:
                self.set_font('Arial', '', 10)
            
            # Split long text into lines
            cleaned_text = self.clean_text(text)
            words = cleaned_text.split(' ')
            line = ''
            
            for word in words:
                if self.get_string_width(line + word + ' ') < 190:  # Page width limit
                    line += word + ' '
                else:
                    if line:
                        self.cell(0, 6, line.strip(), 0, 1, 'L')
                    line = word + ' '
            
            if line:
                self.cell(0, 6, line.strip(), 0, 1, 'L')
            self.ln(3)
    
    # Create PDF instance
    pdf = PDF()
    pdf.add_page()
    
    # Executive Summary
    pdf.add_section_header("Executive Summary")
    summary_text = f"""
    This report presents a comprehensive time series analysis of the dataset containing {st.session_state.data.shape[0]} observations 
    of the variable '{st.session_state.target_column}'. The analysis includes exploratory data analysis, statistical testing, 
    model fitting, and forecasting results.
    """
    pdf.add_paragraph(summary_text)
    
    # Data Summary Section
    if include_data_summary and st.session_state.data is not None:
        pdf.add_section_header("Data Summary")
        
        target_col = st.session_state.target_column
        data_clean = st.session_state.data[target_col].dropna()
        
        # Basic statistics
        pdf.add_paragraph(f"Dataset contains {len(st.session_state.data)} total observations with {len(data_clean)} valid values for analysis.")
        pdf.add_paragraph(f"Time period: {st.session_state.data.index.min().strftime('%Y-%m-%d')} to {st.session_state.data.index.max().strftime('%Y-%m-%d')}")
        
        pdf.ln(5)
        
        # Descriptive statistics table
        pdf.set_font('DejaVu', 'B', 10) if pdf.unicode_available else pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, 'Descriptive Statistics', 0, 1, 'L')
        
        pdf.set_font('DejaVu', '', 9) if pdf.unicode_available else pdf.set_font('Arial', '', 9)
        
        # Table header
        pdf.cell(40, 6, 'Statistic', 1, 0, 'C')
        pdf.cell(40, 6, 'Value', 1, 1, 'C')
        
        # Statistics
        stats = [
            ('Mean', f'{data_clean.mean():.4f}'),
            ('Standard Deviation', f'{data_clean.std():.4f}'),
            ('Minimum', f'{data_clean.min():.4f}'),
            ('Maximum', f'{data_clean.max():.4f}'),
            ('Median', f'{data_clean.median():.4f}'),
            ('Missing Values', f'{st.session_state.data[target_col].isnull().sum()}')
        ]
        
        for stat_name, stat_value in stats:
            pdf.cell(40, 6, stat_name, 1, 0, 'L')
            pdf.cell(40, 6, str(stat_value), 1, 1, 'C')
    
    # Methodology Section
    if include_methodology:
        pdf.add_section_header("Methodology")
        
        methodology_text = """
        The analysis follows a systematic approach to time series modeling:
        
        1. Data Preprocessing: Data cleaning, missing value handling, and time series setup
        2. Exploratory Data Analysis: Statistical summaries and visualization
        3. Stationarity Testing: Augmented Dickey-Fuller and KPSS tests
        4. Model Selection: Comparison of ARIMA, Prophet, and Exponential Smoothing models
        5. Forecasting: Generate predictions with confidence intervals
        6. Validation: Assess model performance using standard metrics
        """
        pdf.add_paragraph(methodology_text)
    
    # Model Results Section
    if 'fitted_model' in st.session_state:
        pdf.add_section_header("Model Results")
        
        model_type = st.session_state.model_type
        
        if model_type == "auto_arima":
            model = st.session_state.fitted_model
            pdf.add_paragraph(f"Auto ARIMA model was fitted with order {model.order}")
            if hasattr(model, 'seasonal_order'):
                pdf.add_paragraph(f"Seasonal order: {model.seasonal_order}")
            pdf.add_paragraph(f"AIC: {model.aic():.3f}, BIC: {model.bic():.3f}")
            
        elif model_type == "manual_arima":
            model = st.session_state.fitted_model
            pdf.add_paragraph(f"Manual ARIMA model fitted successfully")
            pdf.add_paragraph(f"AIC: {model.aic:.3f}, BIC: {model.bic:.3f}")
            pdf.add_paragraph(f"Log-likelihood: {model.llf:.3f}")
            
        elif model_type == "prophet":
            pdf.add_paragraph("Prophet model was fitted with automatic seasonality detection")
            pdf.add_paragraph("The model captures trend and seasonal patterns in the data")
            
        elif model_type == "exp_smoothing":
            model = st.session_state.fitted_model
            pdf.add_paragraph("Exponential Smoothing model fitted successfully")
            pdf.add_paragraph(f"AIC: {model.aic:.3f}, BIC: {model.bic:.3f}")
    
    # Forecast Results Section
    if include_forecast and 'forecast_results' in st.session_state:
        pdf.add_section_header("Forecast Results")
        
        forecast_df = st.session_state.forecast_results
        pdf.add_paragraph(f"Generated {len(forecast_df)} period forecast with confidence intervals")
        
        # Forecast summary
        pdf.add_paragraph(f"Mean forecast value: {forecast_df['Forecast'].mean():.3f}")
        pdf.add_paragraph(f"Forecast range: {forecast_df['Forecast'].min():.3f} to {forecast_df['Forecast'].max():.3f}")
        
        # Forecast table (first 10 rows)
        pdf.ln(5)
        pdf.set_font('DejaVu', 'B', 10) if pdf.unicode_available else pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 8, 'Forecast Table (First 10 Periods)', 0, 1, 'L')
        
        pdf.set_font('DejaVu', '', 8) if pdf.unicode_available else pdf.set_font('Arial', '', 8)
        
        # Table header
        pdf.cell(40, 6, 'Date', 1, 0, 'C')
        pdf.cell(30, 6, 'Forecast', 1, 0, 'C')
        pdf.cell(30, 6, 'Lower CI', 1, 0, 'C')
        pdf.cell(30, 6, 'Upper CI', 1, 1, 'C')
        
        # Table data (first 10 rows)
        for i in range(min(10, len(forecast_df))):
            row = forecast_df.iloc[i]
            date_str = pdf.clean_text(str(row['Date'])[:10] if pd.notnull(row['Date']) else 'N/A')
            forecast_val = f'{row["Forecast"]:.3f}' if pd.notnull(row['Forecast']) else 'N/A'
            lower_ci = f'{row["Lower CI"]:.3f}' if pd.notnull(row['Lower CI']) else 'N/A'
            upper_ci = f'{row["Upper CI"]:.3f}' if pd.notnull(row['Upper CI']) else 'N/A'
            
            pdf.cell(40, 6, date_str, 1, 0, 'C')
            pdf.cell(30, 6, forecast_val, 1, 0, 'C')
            pdf.cell(30, 6, lower_ci, 1, 0, 'C')
            pdf.cell(30, 6, upper_ci, 1, 1, 'C')
    
    # Model Diagnostics Section
    if include_diagnostics and 'fitted_model' in st.session_state:
        pdf.add_section_header("Model Diagnostics")
        
        diagnostics_text = """
        Model validation was performed using standard diagnostic procedures:
        - Residual analysis to check for patterns and outliers
        - Normality testing of residuals
        - Autocorrelation analysis of residuals
        - Out-of-sample validation when test data is available
        """
        pdf.add_paragraph(diagnostics_text)
    
    # Conclusions and Recommendations
    pdf.add_section_header("Conclusions and Recommendations")
    
    conclusions_text = """
    Based on the analysis performed, the following conclusions and recommendations are made:
    
    1. The fitted model provides reasonable forecasts for the given time series
    2. Model diagnostics should be reviewed to ensure assumptions are met
    3. Regular model updates are recommended as new data becomes available
    4. Forecast uncertainty increases with longer prediction horizons
    5. Consider ensemble methods for improved forecast accuracy
    """
    pdf.add_paragraph(conclusions_text)
    
    # Footer information
    pdf.ln(10)
    pdf.set_font('DejaVu', '', 8) if pdf.unicode_available else pdf.set_font('Arial', '', 8)
    pdf.cell(0, 5, f'Report generated using Professional Time Series Analysis Suite', 0, 1, 'C')
    pdf.cell(0, 5, f'Analysis completed on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}', 0, 1, 'C')
    
    return bytes(pdf.output())

if __name__ == "__main__":
    main()
