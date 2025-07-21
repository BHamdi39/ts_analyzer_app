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
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
    st.warning("Prophet not available. Install with: pip install prophet")

# Use statsforecast instead of pmdarima for better compatibility
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
except ImportError:
    StatsForecast = None
    AutoARIMA = None
    st.warning("Statsforecast not available. Install with: pip install statsforecast")

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
    .data-preview-container {
        border: 2px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        background-color: #fafafa;
        margin: 1rem 0;
    }
    .sample-values-box {
        background-color: #e8f4fd;
        border: 1px solid #b3d9ff;
        border-radius: 0.3rem;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'time_column' not in st.session_state:
    st.session_state.time_column = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'fitted_model' not in st.session_state:
    st.session_state.fitted_model = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False

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
    
    # Data status indicator
    if st.session_state.data is not None:
        st.sidebar.markdown("### 📊 Dataset Status")
        st.sidebar.write(f"**Shape:** {st.session_state.data.shape}")
        if st.session_state.target_column:
            st.sidebar.write(f"**Target:** {st.session_state.target_column}")
        
        # Preprocessing status
        if st.session_state.data_preprocessed:
            st.sidebar.success("✅ Data preprocessed")
        else:
            st.sidebar.warning("⚠️ Data needs preprocessing")
    
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

def analyze_date_patterns(date_series):
    """Analyze date patterns and frequency"""
    try:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series, infer_datetime_format=True)
        
        # Basic date info
        date_range = date_series.max() - date_series.min()
        
        # Try to infer frequency
        freq = pd.infer_freq(date_series)
        
        # Calculate time differences
        time_diffs = date_series.diff().dropna()
        
        # Most common time difference
        mode_diff = time_diffs.mode()
        median_diff = time_diffs.median()
        
        # Detect gaps
        expected_freq = median_diff
        gaps = time_diffs[time_diffs > expected_freq * 1.5]
        
        return {
            'min_date': date_series.min(),
            'max_date': date_series.max(),
            'date_range': date_range,
            'inferred_freq': freq,
            'median_interval': median_diff,
            'mode_interval': mode_diff[0] if len(mode_diff) > 0 else None,
            'gaps_count': len(gaps),
            'total_periods': len(date_series),
            'unique_dates': date_series.nunique(),
            'duplicates': len(date_series) - date_series.nunique()
        }
    except Exception as e:
        return {'error': str(e)}

def get_sample_value_ranges(df, column):
    """Get comprehensive sample value information"""
    try:
        if column not in df.columns:
            return {'error': 'Column not found'}
        
        series = df[column].dropna()
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric data analysis
            return {
                'data_type': 'numeric',
                'min_value': series.min(),
                'max_value': series.max(),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'q25': series.quantile(0.25),
                'q75': series.quantile(0.75),
                'sample_values': series.sample(min(10, len(series))).tolist(),
                'unique_count': series.nunique(),
                'zero_count': (series == 0).sum(),
                'negative_count': (series < 0).sum(),
                'positive_count': (series > 0).sum()
            }
        else:
            # Categorical/text data analysis
            value_counts = series.value_counts()
            return {
                'data_type': 'categorical',
                'unique_count': series.nunique(),
                'most_common': value_counts.head(5).to_dict(),
                'sample_values': series.sample(min(10, len(series))).tolist(),
                'avg_length': series.astype(str).str.len().mean() if not pd.api.types.is_datetime64_any_dtype(series) else None,
                'null_count': series.isnull().sum()
            }
    except Exception as e:
        return {'error': str(e)}

def data_upload_section():
    st.markdown('<div class="section-header">📁 Data Upload & Preprocessing</div>', 
                unsafe_allow_html=True)
    
    # File upload with improved UI
    st.markdown("### 📤 Upload Your Time Series Data")
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
            
            status_text.text("📖 Reading file...")
            progress_bar.progress(25)
            
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            progress_bar.progress(50)
            status_text.text("🔍 Analyzing data structure...")
            
            st.session_state.data = df.copy()
            st.session_state.data_preprocessed = False
            
            progress_bar.progress(100)
            status_text.text("✅ File uploaded successfully!")
            
            st.success(f"✅ File uploaded successfully! **{uploaded_file.name}** - Shape: {df.shape}")
            
            # ENHANCED DATA PREVIEW SECTION
            st.markdown('<div class="data-preview-container">', unsafe_allow_html=True)
            st.markdown("### 📊 **Complete Data Preview & Analysis**")
            
            # Data preview options
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                show_all_rows = st.checkbox("🔍 Show all rows", False, 
                                          help="Display complete dataset (be careful with large files)")
                preview_rows = st.selectbox("Preview rows", [10, 25, 50, 100, 500], index=1)
                
            with col2:
                show_all_cols = st.checkbox("📋 Show all columns", True,
                                          help="Display all columns in the dataset")
                
            with col3:
                show_dtypes = st.checkbox("🏷️ Show data types", True)
            
            # Display data with options
            st.markdown("#### 📋 **Dataset Preview**")
            
            if show_all_rows:
                st.markdown(f"**Displaying all {len(df):,} rows:**")
                display_df = df
            else:
                st.markdown(f"**Displaying first {preview_rows} rows:**")
                display_df = df.head(preview_rows)
            
            # Configure display options
            if show_all_cols:
                st.dataframe(display_df, use_container_width=True, height=400)
            else:
                # Show only first 10 columns if too many
                if len(df.columns) > 10:
                    st.dataframe(display_df.iloc[:, :10], use_container_width=True, height=400)
                    st.info(f"Showing first 10 of {len(df.columns)} columns. Enable 'Show all columns' to see more.")
                else:
                    st.dataframe(display_df, use_container_width=True, height=400)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced data info with 4 columns for better layout
            st.markdown("### 📈 **Dataset Overview**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Rows", f"{len(df):,}")
            with col2:
                st.metric("📋 Total Columns", len(df.columns))
            with col3:
                missing_count = df.isnull().sum().sum()
                missing_pct = (missing_count / (len(df) * len(df.columns))) * 100
                st.metric("❓ Missing Values", f"{missing_count:,} ({missing_pct:.1f}%)")
            with col4:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("🔢 Numeric Columns", numeric_cols)
            
            # ENHANCED COLUMN INFORMATION
            st.markdown("### 📋 **Detailed Column Information**")
            
            # Create comprehensive column info
            col_info_data = []
            for col in df.columns:
                col_data = {
                    'Column': col,
                    'Type': str(df[col].dtype),
                    'Non-Null Count': f"{df[col].count():,}",
                    'Missing Count': f"{df[col].isnull().sum():,}",
                    'Missing %': f"{(df[col].isnull().sum() / len(df) * 100):.1f}%",
                    'Unique Values': f"{df[col].nunique():,}",
                    'Memory Usage': f"{df[col].memory_usage(deep=True) / 1024:.1f} KB"
                }
                
                # Add sample values for each column
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null_data = df[col].dropna()
                    if len(non_null_data) > 0:
                        col_data['Min'] = f"{non_null_data.min():.2f}"
                        col_data['Max'] = f"{non_null_data.max():.2f}"
                        col_data['Mean'] = f"{non_null_data.mean():.2f}"
                    else:
                        col_data['Min'] = col_data['Max'] = col_data['Mean'] = 'N/A'
                else:
                    col_data['Min'] = col_data['Max'] = col_data['Mean'] = 'N/A'
                    
                col_info_data.append(col_data)
            
            col_info_df = pd.DataFrame(col_info_data)
            st.dataframe(col_info_df, use_container_width=True, height=400)
            
            # SAMPLE VALUES ANALYSIS FOR EACH COLUMN
            st.markdown("### 🔍 **Sample Values & Range Analysis**")
            
            selected_col_for_analysis = st.selectbox(
                "Select column for detailed analysis:", 
                df.columns.tolist(),
                help="Choose a column to see detailed sample values and statistics"
            )
            
            if selected_col_for_analysis:
                col_analysis = get_sample_value_ranges(df, selected_col_for_analysis)
                
                if 'error' not in col_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="sample-values-box">', unsafe_allow_html=True)
                        st.markdown(f"#### 📊 **{selected_col_for_analysis}** Analysis")
                        
                        if col_analysis['data_type'] == 'numeric':
                            st.write(f"**Data Type:** Numeric ({df[selected_col_for_analysis].dtype})")
                            st.write(f"**Range:** {col_analysis['min_value']:.3f} to {col_analysis['max_value']:.3f}")
                            st.write(f"**Mean:** {col_analysis['mean']:.3f}")
                            st.write(f"**Median:** {col_analysis['median']:.3f}")
                            st.write(f"**Std Deviation:** {col_analysis['std']:.3f}")
                            st.write(f"**25th Percentile:** {col_analysis['q25']:.3f}")
                            st.write(f"**75th Percentile:** {col_analysis['q75']:.3f}")
                        else:
                            st.write(f"**Data Type:** {col_analysis['data_type'].title()}")
                            st.write(f"**Unique Values:** {col_analysis['unique_count']:,}")
                            if col_analysis.get('avg_length'):
                                st.write(f"**Average Length:** {col_analysis['avg_length']:.1f} characters")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 📝 **Sample Values**")
                        sample_values = col_analysis['sample_values']
                        
                        # Display sample values in a nice format
                        for i, val in enumerate(sample_values[:10], 1):
                            st.write(f"**{i}.** {val}")
                        
                        if col_analysis['data_type'] == 'categorical' and 'most_common' in col_analysis:
                            st.markdown("#### 🏆 **Most Common Values**")
                            for val, count in col_analysis['most_common'].items():
                                pct = (count / len(df)) * 100
                                st.write(f"• **{val}:** {count:,} ({pct:.1f}%)")
                else:
                    st.error(f"❌ Error analyzing column: {col_analysis.get('error', 'Unknown error')}")
            
            # TIME SERIES SETUP with Enhanced Date Analysis
            st.markdown("### ⚙️ **Time Series Configuration**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🗓️ **Date/Time Column Selection**")
                time_col = st.selectbox(
                    "Select Time Column",
                    [""] + df.columns.tolist(),
                    help="Choose the column containing dates/time values"
                )
                if time_col:
                    st.session_state.time_column = time_col
                    
                    # ENHANCED DATE ANALYSIS
                    st.markdown("##### 📅 **Date Column Analysis**")
                    date_analysis = analyze_date_patterns(df[time_col])
                    
                    if 'error' not in date_analysis:
                        st.success("✅ Date column successfully analyzed!")
                        
                        # Date range information
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**📅 Start Date:** {date_analysis['min_date']}")
                            st.write(f"**📅 End Date:** {date_analysis['max_date']}")
                            st.write(f"**📏 Total Range:** {date_analysis['date_range']}")
                            
                        with col_b:
                            st.write(f"**📊 Total Periods:** {date_analysis['total_periods']:,}")
                            st.write(f"**🔄 Unique Dates:** {date_analysis['unique_dates']:,}")
                            if date_analysis['duplicates'] > 0:
                                st.warning(f"**⚠️ Duplicate Dates:** {date_analysis['duplicates']:,}")
                        
                        # Time interval analysis
                        st.markdown("##### ⏱️ **Time Interval Analysis**")
                        col_x, col_y = st.columns(2)
                        
                        with col_x:
                            freq_display = date_analysis['inferred_freq'] if date_analysis['inferred_freq'] else "Irregular"
                            st.info(f"**🔍 Detected Frequency:** {freq_display}")
                            
                            if date_analysis['median_interval']:
                                st.write(f"**📊 Median Interval:** {date_analysis['median_interval']}")
                            
                        with col_y:
                            if date_analysis['gaps_count'] > 0:
                                st.warning(f"**⚠️ Potential Gaps:** {date_analysis['gaps_count']} detected")
                            else:
                                st.success("✅ **No gaps detected**")
                                
                            if date_analysis['mode_interval']:
                                st.write(f"**📈 Most Common Interval:** {date_analysis['mode_interval']}")
                    else:
                        st.error(f"❌ Error analyzing dates: {date_analysis['error']}")
                        st.write("**💡 Suggestions:**")
                        st.write("- Check if column contains valid date formats")
                        st.write("- Common formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY")
                        st.write("- Ensure dates are not stored as text with special characters")
                
            with col2:
                st.markdown("#### 🎯 **Target Variable Selection**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    target_col = st.selectbox(
                        "Select Target Variable",
                        [""] + numeric_cols,
                        help="Choose the numeric variable you want to analyze"
                    )
                    if target_col:
                        st.session_state.target_column = target_col
                        
                        # ENHANCED TARGET ANALYSIS
                        st.markdown("##### 📊 **Target Variable Analysis**")
                        target_analysis = get_sample_value_ranges(df, target_col)
                        
                        if 'error' not in target_analysis and target_analysis['data_type'] == 'numeric':
                            st.success("✅ Target variable successfully analyzed!")
                            
                            # Quick stats
                            col_i, col_j = st.columns(2)
                            with col_i:
                                st.metric("📈 Mean", f"{target_analysis['mean']:.2f}")
                                st.metric("📊 Median", f"{target_analysis['median']:.2f}")
                                
                            with col_j:
                                st.metric("📏 Std Dev", f"{target_analysis['std']:.2f}")
                                st.metric("🔢 Unique Values", f"{target_analysis['unique_count']:,}")
                            
                            # Value distribution
                            if target_analysis['negative_count'] > 0:
                                st.write(f"**➖ Negative Values:** {target_analysis['negative_count']:,}")
                            if target_analysis['zero_count'] > 0:
                                st.write(f"**0️⃣ Zero Values:** {target_analysis['zero_count']:,}")
                            st.write(f"**➕ Positive Values:** {target_analysis['positive_count']:,}")
                            
                            # Range info
                            st.info(f"**📊 Value Range:** {target_analysis['min_value']:.3f} to {target_analysis['max_value']:.3f}")
                else:
                    st.error("❌ No numeric columns found for analysis!")
                    st.write("**💡 Suggestions:**")
                    st.write("- Ensure your file contains at least one numeric column")
                    st.write("- Check that numbers are not stored as text")
                    st.write("- Remove any special characters from numeric columns")
            
            # Data preprocessing with enhanced validation
            if st.session_state.time_column and st.session_state.target_column:
                st.markdown("---")
                st.markdown("### 🔧 **Data Preprocessing**")
                
                # Preprocessing options
                col1, col2 = st.columns(2)
                with col1:
                    handle_duplicates = st.selectbox("Handle duplicate dates:", 
                                                   ["Remove duplicates", "Keep first", "Keep last", "Average values"])
                    fill_missing = st.selectbox("Handle missing values:", 
                                               ["Drop missing", "Forward fill", "Backward fill", "Linear interpolation"])
                
                with col2:
                    sort_by_date = st.checkbox("Sort by date", True)
                    validate_frequency = st.checkbox("Validate date frequency", True)
                
                if st.button("🔧 Preprocess Data", type="primary"):
                    with st.spinner("🔄 Processing data..."):
                        try:
                            # Make a copy of the data for processing
                            processed_df = df.copy()
                            processing_log = []
                            
                            # Convert time column with better error handling
                            try:
                                processed_df[time_col] = pd.to_datetime(processed_df[time_col], infer_datetime_format=True)
                                processing_log.append("✅ Date column converted successfully")
                            except:
                                # Try different date parsing methods
                                try:
                                    processed_df[time_col] = pd.to_datetime(processed_df[time_col], format='%Y-%m-%d')
                                    processing_log.append("✅ Date column parsed with YYYY-MM-DD format")
                                except:
                                    try:
                                        processed_df[time_col] = pd.to_datetime(processed_df[time_col], dayfirst=True)
                                        processing_log.append("✅ Date column parsed with day-first format")
                                    except Exception as date_error:
                                        st.error(f"❌ Could not parse date column: {str(date_error)}")
                                        st.write("**Troubleshooting tips:**")
                                        st.write("- Ensure dates are in a standard format (YYYY-MM-DD, MM/DD/YYYY, etc.)")
                                        st.write("- Remove any special characters or extra spaces")
                                        st.write("- Check for inconsistent date formats in the same column")
                                        return
                            
                            # Sort by time column if requested
                            if sort_by_date:
                                processed_df = processed_df.sort_values(time_col)
                                processing_log.append("✅ Data sorted by date")
                            
                            # Handle duplicates
                            initial_rows = len(processed_df)
                            if handle_duplicates == "Remove duplicates":
                                processed_df = processed_df.drop_duplicates(subset=[time_col])
                            elif handle_duplicates == "Keep first":
                                processed_df = processed_df.drop_duplicates(subset=[time_col], keep='first')
                            elif handle_duplicates == "Keep last":
                                processed_df = processed_df.drop_duplicates(subset=[time_col], keep='last')
                            elif handle_duplicates == "Average values":
                                # Group by date and average numeric columns
                                numeric_cols_to_avg = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                                if numeric_cols_to_avg:
                                    processed_df = processed_df.groupby(time_col).agg({
                                        col: 'mean' if col in numeric_cols_to_avg else 'first' 
                                        for col in processed_df.columns if col != time_col
                                    }).reset_index()
                                    processing_log.append("✅ Duplicate dates averaged")
                            
                            duplicates_removed = initial_rows - len(processed_df)
                            if duplicates_removed > 0:
                                processing_log.append(f"✅ {duplicates_removed:,} duplicate rows handled")
                            
                            # Set datetime index
                            processed_df.set_index(time_col, inplace=True)
                            processing_log.append("✅ DateTime index set")
                            
                            # Handle missing values
                            missing_before = processed_df[target_col].isnull().sum()
                            if missing_before > 0:
                                if fill_missing == "Drop missing":
                                    processed_df = processed_df.dropna(subset=[target_col])
                                elif fill_missing == "Forward fill":
                                    processed_df[target_col] = processed_df[target_col].fillna(method='ffill')
                                elif fill_missing == "Backward fill":
                                    processed_df[target_col] = processed_df[target_col].fillna(method='bfill')
                                elif fill_missing == "Linear interpolation":
                                    processed_df[target_col] = processed_df[target_col].interpolate(method='linear')
                                
                                missing_after = processed_df[target_col].isnull().sum()
                                processing_log.append(f"✅ Missing values: {missing_before:,} → {missing_after:,}")
                            
                            # Store processed data
                            st.session_state.processed_data = processed_df
                            st.session_state.data_preprocessed = True
                            
                            st.success("✅ Data preprocessed successfully!")
                            
                            # Show processing log
                            st.markdown("#### 📋 **Processing Summary**")
                            for log_entry in processing_log:
                                st.write(log_entry)
                            
                            # Show processed data info
                            st.markdown("#### 📈 **Processed Data Summary**")
                            info_col1, info_col2, info_col3 = st.columns(3)
                            
                            with info_col1:
                                st.write(f"**📅 Date Range:**")
                                st.write(f"From: {processed_df.index.min().strftime('%Y-%m-%d')}")
                                st.write(f"To: {processed_df.index.max().strftime('%Y-%m-%d')}")
                                
                            with info_col2:
                                st.write(f"**📊 Final Shape:** {processed_df.shape}")
                                st.write(f"**🎯 Target Values:** {processed_df[target_col].count():,}")
                                
                            with info_col3:
                                try:
                                    freq = pd.infer_freq(processed_df.index)
                                    st.write(f"**⏱️ Frequency:** {freq if freq else 'Irregular'}")
                                except:
                                    st.write(f"**⏱️ Frequency:** Could not infer")
                                
                                missing_final = processed_df[target_col].isnull().sum()
                                st.write(f"**❓ Missing in Target:** {missing_final:,}")
                            
                            # Show final preview
                            st.markdown("#### 👀 **Final Processed Data Preview**")
                            st.dataframe(processed_df.head(10), use_container_width=True)
                                
                        except Exception as e:
                            st.error(f"❌ Preprocessing failed: {str(e)}")
                            st.write("**Troubleshooting tips:**")
                            st.write("- Ensure your time column contains valid date formats")
                            st.write("- Check for special characters in date strings")
                            st.write("- Verify your target column contains numeric values")
                            st.write("- Try different missing value handling options")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.write("**Possible solutions:**")
            st.write("- Check file format (CSV or XLSX)")
            st.write("- Ensure file is not corrupted")
            st.write("- Verify file permissions")
            st.write("- Try a smaller file first to test the system")

# Rest of the functions remain the same...
def eda_section():
    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', 
                unsafe_allow_html=True)
    
    # Check if data is preprocessed
    if not st.session_state.data_preprocessed or st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and preprocess data first!")
        return
    
    df = st.session_state.processed_data
    target_col = st.session_state.target_column
    
    if target_col not in df.columns:
        st.error("❌ Target column not found. Please reconfigure in Data Upload section.")
        return
    
    # Validate datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        st.error("❌ Data index is not properly set as datetime. Please re-run preprocessing.")
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
        # Create seasonal analysis with proper datetime index handling
        if len(df) > 24:  # At least 2 years of data
            try:
                df_copy = df.copy()
                # Safely extract month from datetime index
                df_copy['Month'] = df_copy.index.month
                df_copy['Year'] = df_copy.index.year
                
                # Create monthly boxplot
                fig = px.box(df_copy, x='Month', y=target_col, 
                            title="Monthly Distribution")
                fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not create seasonal plot: {str(e)}")
                # Fallback: simple time series plot
                fig = px.line(df, x=df.index, y=target_col, title="Time Series Plot")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for seasonal analysis (need at least 24 observations)")
    
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
            
            # Ensure we have enough data for the selected period
            min_periods = max(2 * decomp_period, 10)
            if len(df) >= min_periods:
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
                    trend_vals = decomposition.trend.dropna()
                    if len(trend_vals) > 1:
                        trend_change = (trend_vals.iloc[-1] - trend_vals.iloc[0])
                        st.metric("📈 Overall Trend", f"{trend_change:.2f}")
                
                with col2:
                    seasonal_strength = decomposition.seasonal.std()
                    st.metric("🔄 Seasonal Strength", f"{seasonal_strength:.2f}")
                
                with col3:
                    residual_std = decomposition.resid.std()
                    st.metric("📊 Residual Std", f"{residual_std:.2f}")
            else:
                st.warning(f"⚠️ Need at least {min_periods} observations for period {decomp_period}")
                
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
    
    # Check if data is preprocessed
    if not st.session_state.data_preprocessed or st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and preprocess data first!")
        return
    
    df = st.session_state.processed_data
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
    
    # Check if data is preprocessed
    if not st.session_state.data_preprocessed or st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and preprocess data first!")
        return
    
    df = st.session_state.processed_data
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
        ["Auto ARIMA (statsforecast)", "Manual ARIMA", "Prophet", "Exponential Smoothing"],
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
    if model_type == "Auto ARIMA (statsforecast)":
        st.markdown("### 🚀 Auto ARIMA Model (statsforecast)")
        st.write("Using Nixtla's statsforecast - faster and more compatible than pmdarima")
        
        # Configuration options
        col1, col2 = st.columns(2)
        with col1:
            season_length = st.number_input("Season Length", 2, 52, 12)
            max_p = st.number_input("Max AR order (p)", 1, 5, 3)
            max_q = st.number_input("Max MA order (q)", 1, 5, 3)
            
        with col2:
            max_d = st.number_input("Max differencing (d)", 0, 2, 2)
            
        if st.button("🚀 Fit Auto ARIMA", type="primary"):
            with st.spinner("Fitting Auto ARIMA model..."):
                try:
                    if StatsForecast is None or AutoARIMA is None:
                        st.error("❌ statsforecast not installed. Please install with: pip install statsforecast")
                        return
                        
                    # Prepare data
                    sf_df = pd.DataFrame({
                        'unique_id': ['series_1'] * len(train_data),
                        'ds': train_data.index,
                        'y': train_data.values
                    })
                    
                    # Configure model
                    model = AutoARIMA(
                        season_length=season_length,
                        max_p=max_p,
                        max_q=max_q,
                        max_d=max_d
                    )
                    
                    # Fit model
                    sf = StatsForecast(
                        models=[model],
                        freq='M'  # Adjust based on your data frequency
                    )
                    
                    sf.fit(sf_df)
                    
                    st.session_state.fitted_model = sf
                    st.session_state.model_type = "statsforecast_arima"
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    st.success("✅ Auto ARIMA model fitted successfully!")
                    
                    # Display model info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("📊 Model", "AutoARIMA (statsforecast)")
                        st.metric("🔄 Season Length", season_length)
                    with col2:
                        st.metric("📈 Training Samples", len(train_data))
                        st.metric("🎯 Max Parameters", f"p={max_p}, d={max_d}, q={max_q}")
                    
                except Exception as e:
                    st.error(f"❌ Model fitting failed: {str(e)}")

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

    elif model_type == "Prophet":
        st.markdown("### 🔮 Prophet Model")
        
        if Prophet is None:
            st.error("❌ Prophet not available. Please install with: pip install prophet")
            return
            
        st.write("Facebook's robust forecasting algorithm for business time series")
        
        # Prophet configuration
        col1, col2 = st.columns(2)
        with col1:
            yearly_seasonality = st.selectbox("Yearly seasonality", [True, False, "auto"], index=0)
            weekly_seasonality = st.selectbox("Weekly seasonality", [True, False, "auto"], index=2)
        with col2:
            daily_seasonality = st.selectbox("Daily seasonality", [True, False, "auto"], index=1)
            changepoint_prior_scale = st.slider("Changepoint prior scale", 0.001, 0.5, 0.05)
        
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

    elif model_type == "Exponential Smoothing":
        st.markdown("### 📈 Exponential Smoothing")
        st.write("Classic forecasting method with trend and seasonal components")
        
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Exponential Smoothing configuration
        col1, col2 = st.columns(2)
        with col1:
            trend = st.selectbox("Trend component", [None, "add", "mul"])
            seasonal = st.selectbox("Seasonal component", [None, "add", "mul"])
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
                    
                    # Residual analysis
                    plot_residual_analysis(fitted_model.resid)
                    
                except Exception as e:
                    st.error(f"❌ Exponential Smoothing fitting failed: {str(e)}")

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
        try:
            from scipy import stats
            jb_stat, jb_pvalue = stats.jarque_bera(residuals)
            if jb_pvalue > 0.05:
                st.metric("✅ Normality", "Normal")
            else:
                st.metric("⚠️ Normality", "Non-normal")
        except:
            st.metric("❓ Normality", "N/A")

def forecasting_section():
    st.markdown('<div class="section-header">🔮 Forecasting & Results</div>', 
                unsafe_allow_html=True)
    
    if 'fitted_model' not in st.session_state or st.session_state.fitted_model is None:
        st.warning("⚠️ Please fit a model first in the Time Series Models section!")
        return
    
    model = st.session_state.fitted_model
    model_type = st.session_state.model_type
    
    # Forecast parameters
    st.markdown("### 🎯 Forecast Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_periods = st.number_input("Number of periods to forecast", 
                                         min_value=1, max_value=100, value=12)
    with col2:
        confidence_level = st.selectbox("Confidence level", [80, 90, 95, 99], index=2)
    
    if st.button("🔮 Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                if model_type == "statsforecast_arima":
                    # Statsforecast forecasting
                    forecasts = model.predict(h=forecast_periods)
                    
                    forecast_df = pd.DataFrame({
                        'Date': pd.date_range(start=st.session_state.train_data.index[-1] + pd.DateOffset(1), 
                                            periods=forecast_periods, freq='M'),
                        'Forecast': forecasts['AutoARIMA'].values,
                        'Lower CI': forecasts['AutoARIMA'].values,  # Simplified
                        'Upper CI': forecasts['AutoARIMA'].values   # Simplified
                    })
                    
                elif model_type == "manual_arima":
                    # Manual ARIMA forecasting
                    forecast = model.forecast(steps=forecast_periods)
                    forecast_obj = model.get_forecast(steps=forecast_periods)
                    conf_int = forecast_obj.conf_int(alpha=(100-confidence_level)/100)
                    
                    forecast_dates = pd.date_range(
                        start=st.session_state.train_data.index[-1] + pd.DateOffset(1),
                        periods=forecast_periods,
                        freq='M'
                    )
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast': forecast,
                        'Lower CI': conf_int.iloc[:, 0],
                        'Upper CI': conf_int.iloc[:, 1]
                    })
                
                elif model_type == "prophet":
                    # Prophet forecasting
                    future = model.make_future_dataframe(periods=forecast_periods, freq='M')
                    forecast = model.predict(future)
                    
                    forecast_subset = forecast.tail(forecast_periods)
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_subset['ds'],
                        'Forecast': forecast_subset['yhat'],
                        'Lower CI': forecast_subset['yhat_lower'],
                        'Upper CI': forecast_subset['yhat_upper']
                    })
                
                elif model_type == "exp_smoothing":
                    # Exponential Smoothing forecasting
                    forecast = model.forecast(steps=forecast_periods)
                    
                    forecast_dates = pd.date_range(
                        start=st.session_state.train_data.index[-1] + pd.DateOffset(1),
                        periods=forecast_periods,
                        freq='M'
                    )
                    
                    # Simple confidence intervals
                    std_error = np.std(model.resid)
                    z_score = {80: 1.28, 90: 1.64, 95: 1.96, 99: 2.58}[confidence_level]
                    
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast': forecast,
                        'Lower CI': forecast - z_score * std_error,
                        'Upper CI': forecast + z_score * std_error
                    })
                
                st.session_state.forecast_results = forecast_df
                
                # Plot forecast
                plot_forecast(st.session_state.train_data, st.session_state.test_data, 
                            forecast_df, confidence_level, model_type)
                
                # Display forecast table
                st.markdown("### 📊 Forecast Results")
                st.dataframe(forecast_df.round(4), use_container_width=True)
                
                # Forecast summary
                display_forecast_summary(forecast_df)
                
            except Exception as e:
                st.error(f"❌ Forecasting failed: {str(e)}")

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

def export_section():
    st.markdown('<div class="section-header">📄 Export & Reporting</div>', 
                unsafe_allow_html=True)
    
    if not st.session_state.data_preprocessed or st.session_state.processed_data is None:
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
                    if st.session_state.processed_data is not None:
                        st.session_state.processed_data.to_excel(writer, sheet_name='Original_Data')
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=buffer.getvalue(),
                    file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.ms-excel",
                    key="excel_download"
                )
    
    # PDF Report Generation
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
                pdf_data = generate_pdf_report(
                    title=report_title,
                    author=author_name,
                    include_forecast=include_forecast,
                    include_diagnostics=include_diagnostics,
                    include_data_summary=include_data_summary,
                    include_methodology=include_methodology
                )
                
                st.success("✅ PDF report generated successfully!")
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"time_series_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )
                
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")

def generate_pdf_report(title, author, include_forecast, include_diagnostics, include_data_summary, include_methodology):
    """Generate PDF report with Unicode support"""
    
    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.setup_unicode_font()
        
        def setup_unicode_font(self):
            """Setup Unicode font support"""
            font_path = "DejaVuSansCondensed.ttf"
            
            if not os.path.exists(font_path):
                try:
                    url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSansCondensed.ttf"
                    urllib.request.urlretrieve(url, font_path)
                except:
                    self.unicode_available = False
                    return
            
            try:
                self.add_font('DejaVu', '', font_path, uni=True)
                self.unicode_available = True
            except:
                self.unicode_available = False
        
        def header(self):
            if hasattr(self, 'unicode_available') and self.unicode_available:
                self.set_font('DejaVu', 'B', 16)
            else:
                self.set_font('Arial', 'B', 16)
            
            clean_title = self.clean_text(title)
            self.cell(0, 10, clean_title, 0, 1, 'C')
            
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
            """Clean text for PDF compatibility"""
            if not isinstance(text, str):
                text = str(text)
            
            replacements = {
                '"': '"', '"': '"',
                ''': "'", ''': "'",
                '—': '-', '–': '-',
                '…': '...',
                '°': ' degrees',
                '±': '+/-'
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            if not (hasattr(self, 'unicode_available') and self.unicode_available):
                text = ''.join(char if ord(char) < 128 else '?' for char in text)
            
            return text
    
    pdf = PDF()
    pdf.add_page()
    
    # Add content sections
    if include_data_summary and st.session_state.processed_data is not None:
        pdf.cell(0, 10, 'Data Summary', 0, 1, 'L')
        pdf.ln(5)
        
        target_col = st.session_state.target_column
        data_clean = st.session_state.processed_data[target_col].dropna()
        
        pdf.cell(0, 8, f'Dataset shape: {st.session_state.processed_data.shape}', 0, 1, 'L')
        pdf.cell(0, 8, f'Target variable: {target_col}', 0, 1, 'L')
        pdf.cell(0, 8, f'Mean: {data_clean.mean():.4f}', 0, 1, 'L')
        pdf.cell(0, 8, f'Standard Deviation: {data_clean.std():.4f}', 0, 1, 'L')
        pdf.ln(10)
    
    if include_forecast and 'forecast_results' in st.session_state:
        pdf.cell(0, 10, 'Forecast Results', 0, 1, 'L')
        pdf.ln(5)
        
        forecast_df = st.session_state.forecast_results
        pdf.cell(0, 8, f'Forecast periods: {len(forecast_df)}', 0, 1, 'L')
        pdf.cell(0, 8, f'Mean forecast: {forecast_df["Forecast"].mean():.3f}', 0, 1, 'L')
    
    return bytes(pdf.output())

if __name__ == "__main__":
    main()
