import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Page configuration
st.set_page_config(page_title="Distribution Fitting Tool", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Righteous&display=swap');
    
    .main-header {
        font-size: 5.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-top: 4rem;
        margin-bottom: 2rem;
        font-family: 'Righteous', cursive;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .welcome-text {
        text-align: center;
        color: #555;
        font-size: 1.4rem;
        margin-bottom: 4rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .data-option-card {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
        cursor: pointer;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .data-option-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Dictionary of distributions
DISTRIBUTIONS = {
    'Normal': {'obj': stats.norm, 'params': ['loc (mean)', 'scale (std)'], 'ranges': [(0, 100), (0.1, 50)]},
    'Gamma': {'obj': stats.gamma, 'params': ['a (shape)', 'loc', 'scale'], 'ranges': [(0.1, 10), (0, 20), (0.1, 10)]},
    'Weibull': {'obj': stats.weibull_min, 'params': ['c (shape)', 'loc', 'scale'], 'ranges': [(0.1, 10), (0, 20), (0.1, 10)]},
    'Exponential': {'obj': stats.expon, 'params': ['loc', 'scale'], 'ranges': [(0, 20), (0.1, 10)]},
    'Log-Normal': {'obj': stats.lognorm, 'params': ['s (shape)', 'loc', 'scale'], 'ranges': [(0.1, 3), (0, 10), (0.1, 10)]},
    'Beta': {'obj': stats.beta, 'params': ['a (alpha)', 'b (beta)', 'loc', 'scale'], 'ranges': [(0.1, 10), (0.1, 10), (0, 1), (0.1, 10)]},
    'Uniform': {'obj': stats.uniform, 'params': ['loc (start)', 'scale (width)'], 'ranges': [(0, 50), (0.1, 50)]},
    'Chi-Square': {'obj': stats.chi2, 'params': ['df', 'loc', 'scale'], 'ranges': [(1, 30), (0, 20), (0.1, 10)]},
    'Student-t': {'obj': stats.t, 'params': ['df', 'loc', 'scale'], 'ranges': [(1, 30), (0, 50), (0.1, 20)]},
    'Rayleigh': {'obj': stats.rayleigh, 'params': ['loc', 'scale'], 'ranges': [(0, 20), (0.1, 10)]},
    'Pareto': {'obj': stats.pareto, 'params': ['b (shape)', 'loc', 'scale'], 'ranges': [(0.1, 5), (0, 10), (0.1, 10)]},
    'Cauchy': {'obj': stats.cauchy, 'params': ['loc', 'scale'], 'ranges': [(0, 50), (0.1, 20)]}
}

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'fitting_mode' not in st.session_state:
    st.session_state.fitting_mode = 'automatic'
if 'selected_distribution' not in st.session_state:
    st.session_state.selected_distribution = 'Normal'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'input_method_selected' not in st.session_state:
    st.session_state.input_method_selected = None

# ============================================================================
# HOME PAGE - DATA SELECTION
# ============================================================================
if not st.session_state.data_loaded:
    
    st.markdown('<p class="main-header">DISTRIBUTION FITTING TOOL</p>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Fit statistical distributions to your data</p>', unsafe_allow_html=True)
    
    st.markdown("### Choose how to input your data:")
    
    # Three columns for data input options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✍️ Manual Entry", use_container_width=True, type="secondary", key="select_manual"):
            st.session_state.input_method_selected = "manual"
            st.rerun()
    
    with col2:
        if st.button("📁 Upload CSV", use_container_width=True, type="secondary", key="select_csv"):
            st.session_state.input_method_selected = "csv"
            st.rerun()
    
    with col3:
        if st.button("🎲 Generate Sample", use_container_width=True, type="secondary", key="select_sample"):
            st.session_state.input_method_selected = "sample"
            st.rerun()
    
    # Show input form based on selection
    if st.session_state.input_method_selected:
        st.markdown("---")
        
        if st.session_state.input_method_selected == "manual":
            st.markdown("#### ✍️ Enter your data")
            data_input = st.text_area(
                "Enter comma-separated values:",
                value="12.5, 14.3, 13.8, 15.1, 12.9, 14.7, 13.2, 15.5, 14.1, 13.6",
                height=150
            )
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Load Data", use_container_width=True, type="primary"):
                    try:
                        st.session_state.data = np.array([float(x.strip()) for x in data_input.split(',')])
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {len(st.session_state.data)} points")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Invalid input format: {str(e)}")
        
        elif st.session_state.input_method_selected == "csv":
            st.markdown("#### 📁 Upload CSV file")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    column = st.selectbox("Select data column:", df.columns)
                    
                    col_a, col_b, col_c = st.columns([1, 1, 1])
                    with col_b:
                        if st.button("Load Data", use_container_width=True, type="primary"):
                            try:
                                # Convert column to numeric, coercing errors to NaN
                                numeric_data = pd.to_numeric(df[column], errors='coerce')
                                # Remove NaN values
                                clean_data = numeric_data.dropna().values
                                
                                if len(clean_data) == 0:
                                    st.error("❌ No valid numeric data found in this column")
                                else:
                                    st.session_state.data = clean_data
                                    st.session_state.data_loaded = True
                                    non_numeric_count = len(df[column]) - len(clean_data)
                                    if non_numeric_count > 0:
                                        st.warning(f"⚠️ Removed {non_numeric_count} non-numeric values")
                                    st.success(f"✅ Loaded {len(clean_data)} valid data points")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error processing column: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        else:  # sample
            st.markdown("#### 🎲 Generate sample data")
            sample_size = st.number_input("Number of samples:", 50, 5000, 500, 50)
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("Generate", use_container_width=True, type="primary"):
                    st.session_state.data = stats.gamma.rvs(5, loc=1, scale=1, size=sample_size)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Generated {sample_size} points")
                    st.rerun()

# ============================================================================
# MAIN APP - AFTER DATA IS LOADED
# ============================================================================
else:
    st.markdown('<p class="main-header">📊 Distribution Fitting Tool</p>', unsafe_allow_html=True)
    
    # Prominent "Load New Data" button at the top
    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        if st.button("🔄 Load New Data", use_container_width=True, type="primary", key="load_new_top"):
            st.session_state.data_loaded = False
            st.session_state.input_method_selected = None
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar with data info
    with st.sidebar:
        st.header("📊 Current Data")
        data = st.session_state.data
        st.metric("Points", len(data))
        st.metric("Mean", f"{np.mean(data):.2f}")
        st.metric("Std Dev", f"{np.std(data):.2f}")
        st.metric("Range", f"{np.ptp(data):.2f}")
    
    # Fitting mode selector
    st.markdown("### Fitting Mode")
    mode_col1, mode_col2, mode_col3 = st.columns([1, 1, 2])
    with mode_col1:
        if st.button("🤖 Automatic Fitting", use_container_width=True, type="primary" if st.session_state.fitting_mode == 'automatic' else "secondary"):
            st.session_state.fitting_mode = 'automatic'
            st.rerun()
    with mode_col2:
        if st.button("🎛️ Manual Fitting", use_container_width=True, type="primary" if st.session_state.fitting_mode == 'manual' else "secondary"):
            st.session_state.fitting_mode = 'manual'
            st.rerun()
    
    st.markdown("---")
    
    data = st.session_state.data
    
    # ============================================================================
    # AUTOMATIC FITTING MODE
    # ============================================================================
    if st.session_state.fitting_mode == 'automatic':
        
        # Distribution selector and fit button in one row
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_dist = st.selectbox(
                "Select Distribution:",
                list(DISTRIBUTIONS.keys()),
                index=list(DISTRIBUTIONS.keys()).index(st.session_state.selected_distribution)
            )
            # Update session state when selection changes
            st.session_state.selected_distribution = selected_dist
        with col2:
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            fit_button = st.button("🔄 Fit", use_container_width=True, type="primary")
        
        # Perform fitting
        if fit_button:
            try:
                dist_obj = DISTRIBUTIONS[selected_dist]['obj']
                params = dist_obj.fit(data)
                st.session_state.fitted_params = params
                st.session_state.fitted_dist_name = selected_dist
                st.session_state.fitted_dist_obj = dist_obj
            except Exception as e:
                st.error(f"Fitting failed: {e}")
        
        # Display results
        if 'fitted_params' in st.session_state:
            st.markdown("---")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Histogram
            ax.hist(data, bins=30, density=True, alpha=0.7, color='#1f77b4', edgecolor='black', label='Data')
            
            # Fitted curve
            x_min, x_max = data.min(), data.max()
            x_range = x_max - x_min
            x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 300)
            
            fitted_dist = st.session_state.fitted_dist_obj(*st.session_state.fitted_params)
            pdf = fitted_dist.pdf(x)
            
            ax.plot(x, pdf, 'r-', linewidth=3, label=f'{st.session_state.fitted_dist_name} Fit')
            
            ax.set_xlabel('Value', fontsize=14, fontweight='bold')
            ax.set_ylabel('Density', fontsize=14, fontweight='bold')
            ax.set_title(f'{st.session_state.fitted_dist_name} Distribution Fit', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
            # Results in columns
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Fitted Parameters")
                param_names = DISTRIBUTIONS[st.session_state.fitted_dist_name]['params']
                for name, value in zip(param_names, st.session_state.fitted_params):
                    st.metric(name, f"{value:.4f}")
            
            with col2:
                st.markdown("#### 📈 Goodness of Fit")
                ks_stat, ks_p = stats.kstest(data, fitted_dist.cdf)
                
                # Calculate MSE
                hist_heights, bin_edges = np.histogram(data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf_at_centers = fitted_dist.pdf(bin_centers)
                mse = np.mean((hist_heights - pdf_at_centers)**2)
                
                st.metric("KS Statistic (lower is better)", f"{ks_stat:.4f}")
                st.metric("KS p-value (higher is better)", f"{ks_p:.4f}")
                st.metric("Mean Squared Error", f"{mse:.6f}")
    
    # ============================================================================
    # MANUAL FITTING MODE
    # ============================================================================
    else:
        
        # Distribution selector
        selected_dist = st.selectbox(
            "Select Distribution:",
            list(DISTRIBUTIONS.keys()),
            index=list(DISTRIBUTIONS.keys()).index(st.session_state.selected_distribution)
        )
        # Update session state when selection changes
        st.session_state.selected_distribution = selected_dist
        
        dist_info = DISTRIBUTIONS[selected_dist]
        dist_obj = dist_info['obj']
        param_names = dist_info['params']
        param_ranges = dist_info['ranges']
        
        st.markdown("---")
        
        # Two column layout: sliders on left, plot on right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎛️ Adjust Parameters")
            manual_params = []
            for i, (name, (min_val, max_val)) in enumerate(zip(param_names, param_ranges)):
                # Check if we have fitted params from automatic mode for this distribution
                if ('fitted_params' in st.session_state and 
                    st.session_state.fitted_dist_name == selected_dist and
                    i < len(st.session_state.fitted_params)):
                    # Use the fitted parameter as default
                    fitted_val = float(st.session_state.fitted_params[i])
                    
                    # Expand range to accommodate fitted value if needed
                    if fitted_val < min_val:
                        min_val = fitted_val - abs(fitted_val) * 0.5
                    if fitted_val > max_val:
                        max_val = fitted_val + abs(fitted_val) * 0.5
                    
                    default_val = fitted_val
                else:
                    # Use middle of range if no fitted params
                    default_val = (min_val + max_val) / 2
                
                param_value = st.slider(
                    name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=(max_val - min_val) / 200,
                    key=f"slider_{selected_dist}_{i}"
                )
                manual_params.append(param_value)
        
        with col2:
            st.markdown("#### 📊 Live Preview")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Histogram
            ax.hist(data, bins=30, density=True, alpha=0.7, color='#2ca02c', edgecolor='black', label='Data')
            
            # Manual curve
            try:
                x_min, x_max = data.min(), data.max()
                x_range = x_max - x_min
                x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 300)
                
                manual_dist = dist_obj(*manual_params)
                pdf = manual_dist.pdf(x)
                
                # Use red color like automatic mode for consistency
                ax.plot(x, pdf, 'r-', linewidth=3, label=f'{selected_dist} (Manual)')
                
                ax.set_xlabel('Value', fontsize=14, fontweight='bold')
                ax.set_ylabel('Density', fontsize=14, fontweight='bold')
                ax.set_title(f'Manual {selected_dist} Distribution Fitting', fontsize=16, fontweight='bold')
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Fit quality metrics below plot
                st.markdown("---")
                st.markdown("#### 📈 Fit Quality")
                col_a, col_b, col_c = st.columns(3)
                
                ks_stat, ks_p = stats.kstest(data, manual_dist.cdf)
                hist_heights, bin_edges = np.histogram(data, bins=30, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                pdf_at_centers = manual_dist.pdf(bin_centers)
                mse = np.mean((hist_heights - pdf_at_centers)**2)
                
                col_a.metric("KS Statistic", f"{ks_stat:.4f}")
                col_b.metric("KS p-value", f"{ks_p:.4f}")
                col_c.metric("MSE", f"{mse:.6f}")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 20px;'>"
        "Statistical Distribution Fitting Tool | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )