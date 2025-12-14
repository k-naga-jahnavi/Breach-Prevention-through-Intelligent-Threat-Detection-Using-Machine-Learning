from sklearn.metrics import confusion_matrix
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pipeline
import matplotlib.pyplot as plt
import plotly.express as px
import auth

# Page config with enhanced styling
st.set_page_config(
    page_title="Cyber Threat Detection Framework", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS Styling with Professional Purple Theme and Transparent Backgrounds
st.markdown("""
<style>
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stSidebar { 
        background: linear-gradient(180deg, rgba(142, 68, 173, 0.9) 0%, rgba(155, 89, 182, 0.9) 100%);
        border-right: 3px solid #e74c3c;
        backdrop-filter: blur(10px);
    }
    
    /* Premium Cards - More Transparent */
    .premium-card {
        background: rgba(255,255,255,0.88);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        backdrop-filter: blur(15px);
        border-left: 4px solid #9b59b6;
    }
    
    /* Metric Highlights - More Transparent */
    .metric-highlight {
        background: linear-gradient(135deg, rgba(155, 89, 182, 0.85) 0%, rgba(142, 68, 173, 0.85) 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 8px;
        box-shadow: 0 6px 20px rgba(155, 89, 182, 0.3);
        border: 1px solid rgba(255,255,255,0.4);
        transition: transform 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-highlight:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(155, 89, 182, 0.4);
    }
    
    /* Headers */
    .header-gradient {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Buttons - More Transparent */
    .stButton>button {
        background: linear-gradient(135deg, rgba(155, 89, 182, 0.9) 0%, rgba(142, 68, 173, 0.9) 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(5px);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(142, 68, 173, 0.95) 0%, rgba(125, 60, 152, 0.95) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(155, 89, 182, 0.4);
        color: white;
    }
    
    /* User Info Box - More Transparent */
    .user-info-box {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.85) 0%, rgba(41, 128, 185, 0.85) 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        border: 1px solid rgba(255,255,255,0.4);
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Elements - More Transparent */
    .sidebar-section {
        background: rgba(255,255,255,0.12);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 3px solid #e74c3c;
        backdrop-filter: blur(15px);
    }
    
    /* Input fields - More Transparent */
    .stTextInput>div>div>input {
        background: rgba(255,255,255,0.85);
        border: 2px solid #9b59b6;
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }
    
    .stSelectbox>div>div>select {
        background: rgba(255,255,255,0.85);
        border: 2px solid #9b59b6;
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }
    
    /* Checkbox */
    .stCheckbox>label {
        color: white;
        font-weight: 500;
    }
    
    /* File uploader - More Transparent */
    .stFileUploader>div>div>div {
        background: rgba(255,255,255,0.85);
        border: 2px dashed #9b59b6;
        border-radius: 8px;
        backdrop-filter: blur(5px);
    }
    
    /* Dataframe containers - More Transparent */
    .dataframe {
        background: rgba(255,255,255,0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Plotly chart containers - More Transparent */
    .js-plotly-plot .plotly .modebar {
        background: rgba(255,255,255,0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Success and error messages - More Transparent */
    .stSuccess {
        background: rgba(46, 204, 113, 0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: rgba(231, 76, 60, 0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Expander - More Transparent */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.9) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Breach Alert Styling */
    .breach-alert-high {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.9) 0%, rgba(192, 57, 43, 0.9) 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #c0392b;
    }
    
    .breach-alert-medium {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.9) 0%, rgba(230, 126, 34, 0.9) 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #d35400;
    }
    
    .breach-alert-low {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.9) 0%, rgba(41, 128, 185, 0.9) 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
if 'authenticator' not in st.session_state:
    st.session_state.authenticator = auth.setup_authentication()

# Check authentication
authenticator = st.session_state.authenticator
if not authenticator.authenticated:
    auth.show_login_form()
    st.stop()

# User is authenticated - show main application
user_info = authenticator.get_user_info()

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'full_df' not in st.session_state:
    st.session_state.full_df = None
if 'breach_results' not in st.session_state:
    st.session_state.breach_results = None

# Header with user info
col1, col2, col3 = st.columns([2, 3, 1])
with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 20px;'>
        <h1 class='header-gradient' style='font-size: 2.5em; margin-bottom: 10px;'>🛡️ Cyber Threat Detection Framework</h1>
        <h3 style='color: #e0e0e0; font-weight: 300;'>Advanced ML Ensembles + GNN for Cyber Security Analytics</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='user-info-box'>
        <p style='margin: 0; font-weight: 600;'>{user_info['name']}</p>
        <p style='margin: 0; font-size: 0.8em; opacity: 0.9;'>{user_info['role']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout"):
        authenticator.logout()
        st.rerun()

# Sidebar with enhanced styling
with st.sidebar:
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(155, 89, 182, 0.9) 0%, rgba(142, 68, 173, 0.9) 100%); padding: 20px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; backdrop-filter: blur(10px);'>
        <h3 style='margin: 0; color: white;'>⚙️ Configuration</h3>
        <p style='margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em;'>Welcome, {user_info['name']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Source Section
    st.markdown("""
    <div class='sidebar-section'>
        <h4 style='color: white; margin-bottom: 10px;'>📁 DATA SOURCE</h4>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("**Upload Dataset (CSV)**", type="csv",
                                   help="Must include 'label' column for classification")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis Configuration
    st.markdown("""
    <div class='sidebar-section'>
        <h4 style='color: white; margin-bottom: 10px;'>🔧 ANALYSIS SETTINGS</h4>
    """, unsafe_allow_html=True)
    
    detection_type = st.selectbox("**Detection Mode**",
                                ["Anomaly Detection", "Intrusion Detection", "Mixed Threat Analysis"])
    
    use_pca = st.checkbox("**Use PCA for Dimensionality Reduction**", value=True)
    
    analyze_btn = st.button("**🚀 Analyze Threats**", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Breach Protection Settings
    st.markdown("""
    <div class='sidebar-section'>
        <h4 style='color: white; margin-bottom: 10px;'>🛡️ BREACH PROTECTION</h4>
    """, unsafe_allow_html=True)
    
    sensitivity = st.slider("**Detection Sensitivity**", 1, 10, 7,
                           help="Higher values detect more potential breaches but may increase false positives")
    
    min_confidence = st.slider("**Minimum Confidence**", 0.1, 1.0, 0.7,
                              help="Minimum confidence threshold for breach classification")
    
    auto_quarantine = st.checkbox("**Auto-Quarantine Suspicious Nodes**", value=True)
    block_malicious_ips = st.checkbox("**Block Malicious IPs**", value=True)
    alert_security_team = st.checkbox("**Real-time Security Alerts**", value=True)
    backup_critical_data = st.checkbox("**Backup Critical Data**", value=False)
    
    detect_breaches_btn = st.button("**🚨 Detect & Protect**", type="secondary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

def display_breach_analysis_results(breach_results):
    """Display breach analysis results with protection status"""
    
    # Breach Overview
    st.markdown("""
    <div class='premium-card'>
        <h3 style='color: #e74c3c; text-align: center;'>📊 Breach Detection Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Potential Breaches", breach_results['potential_breaches'])
    with col2:
        st.metric("High Risk Incidents", breach_results['high_risk_incidents'])
    with col3:
        st.metric("Protected Assets", breach_results['protected_assets'])
    with col4:
        st.metric("Prevention Rate", f"{breach_results['prevention_rate']:.1f}%")
    
    # Breach Types
    if 'breach_types_chart' in breach_results and breach_results['breach_types_chart']:
        st.plotly_chart(breach_results['breach_types_chart'], use_container_width=True)
    
    # Detailed Breach Analysis
    if 'detailed_analysis' in breach_results and breach_results['detailed_analysis']['risk_scores']:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #e74c3c; text-align: center;'>🔍 Detailed Breach Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create dataframe for detailed analysis
        risk_data = []
        for incident in breach_results['detailed_analysis']['risk_scores'][:10]:  # Show top 10
            risk_data.append({
                'Type': incident['type'].replace('_', ' ').title(),
                'Description': incident['description'],
                'Confidence': f"{incident['adjusted_confidence']:.2f}",
                'Risk Score': f"{incident['risk_score']:.1f}",
                'Severity': incident['severity']
            })
        
        if risk_data:
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True)
    
    # Protection Actions Taken
    st.markdown("""
    <div class='premium-card'>
        <h3 style='color: #27ae60; text-align: center;'>✅ Protection Actions Taken</h3>
    </div>
    """, unsafe_allow_html=True)
    
    protection_data = []
    for action, status in breach_results['protection_actions'].items():
        protection_data.append({
            'Action': action.replace('_', ' ').title(),
            'Status': '✅ Implemented' if status else '❌ Not Applied',
            'Impact': 'High' if status else 'Low'
        })
    
    protection_df = pd.DataFrame(protection_data)
    st.dataframe(protection_df, use_container_width=True)
    
    # Real-time Alerts
    if breach_results['alerts']:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #e67e22; text-align: center;'>🚨 Security Alerts</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for alert in breach_results['alerts']:
            alert_class = "breach-alert-high" if alert['level'] in ['CRITICAL', 'HIGH'] else "breach-alert-medium" if alert['level'] == 'MEDIUM' else "breach-alert-low"
            st.markdown(f"""
            <div class='{alert_class}'>
                <strong>{alert['level']}</strong>: {alert['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # Download Protection Report
    if st.button("📥 Download Protection Report"):
        pipeline.generate_protection_report(breach_results)

# Main content
if analyze_btn:
    if uploaded_file is None:
        st.error("❌ Please upload a CSV dataset to proceed.")
        st.stop()

    with st.spinner('🔍 Loading and preprocessing data...'):
        try:
            X, y, full_df, label_encoder, geo_cols = pipeline.load_and_preprocess_data(uploaded_file)
            st.success(f"✅ Data loaded successfully! Shape: {X.shape}")

            with st.expander("📊 Dataset Preview"):
                st.dataframe(full_df.head(5))
                st.write("**Label distribution:**")
                st.write(full_df['label'].value_counts())
                
                if 'threat_label' in full_df.columns:
                    st.write("**Threat class distribution:**")
                    st.write(full_df['threat_label'].value_counts())

        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()

    with st.spinner('🤖 Splitting data and training models...'):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # UPDATED: Pass full_df to train_models for GNN processing
            results = pipeline.train_models(X_train, y_train, X_test, y_test, use_pca=use_pca, full_df=full_df)

            st.session_state.results = results
            st.session_state.y_test = y_test
            st.session_state.label_encoder = label_encoder
            st.session_state.full_df = full_df

        except Exception as e:
            st.error(f"❌ Error training models: {e}")
            st.stop()

    st.markdown("""
    <div class='premium-card'>
        <h2 style='color: #8e44ad; text-align: center; margin-bottom: 20px;'>📈 Model Performance Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # Accuracy table
    acc_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy (%)': [results[m]['accuracy'] for m in results.keys()]
    }).sort_values('Accuracy (%)', ascending=False)

    st.dataframe(acc_df.style.highlight_max(axis=0, color='#d7bde2'), use_container_width=True)

    # Plot metrics
    fig_acc, fig_metrics = pipeline.plot_metrics(results, label_encoder)
    st.plotly_chart(fig_acc, use_container_width=True)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # ALL CONFUSION MATRICES
    st.markdown("""
    <div class='premium-card'>
        <h2 style='color: #8e44ad; text-align: center; margin-bottom: 20px;'>🎯 All Confusion Matrices</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.results and st.session_state.y_test is not None:
        results = st.session_state.results
        y_test = st.session_state.y_test
        label_encoder = st.session_state.label_encoder
        
        # Display all confusion matrices in a grid
        model_names = list(results.keys())
        
        # Create 2 columns for the confusion matrices
        cols = st.columns(2)
        for idx, model_name in enumerate(model_names):
            with cols[idx % 2]:
                st.markdown(f"**{model_name}**")
                try:
                    with st.spinner(f"Generating confusion matrix for {model_name}..."):
                        fig_cm = pipeline.plot_confusion_matrix(results, model_name, y_test, label_encoder)
                        if fig_cm:
                            st.pyplot(fig_cm)
                            plt.close(fig_cm)
                        else:
                            st.warning(f"Could not generate confusion matrix for {model_name}")
                except Exception as e:
                    st.error(f"Error generating confusion matrix for {model_name}: {e}")
                    try:
                        y_pred = results[model_name]['predictions']
                        cm = confusion_matrix(y_test, y_pred)
                        st.text("Confusion Matrix (text version):")
                        st.write(cm)
                    except:
                        st.warning("Confusion matrix data unavailable")

    # Feature Importance Analysis
    st.markdown("""
    <div class='premium-card'>
        <h2 style='color: #8e44ad; text-align: center; margin-bottom: 20px;'>🔍 Feature Importance Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show feature importance for tree-based models
    tree_models = ['Random Forest', 'XGBoost', 'Extra Trees', 'Hybrid GNN+XGBoost']
    for model_name in tree_models:
        if model_name in results and results[model_name]['model'] is not None:
            try:
                model = results[model_name]['model']
                feature_names = X.columns.tolist()
                fig_importance = pipeline.generate_feature_importance_plot(model, feature_names)
                if fig_importance:
                    st.markdown(f"**{model_name} - Feature Importance**")
                    st.pyplot(fig_importance)
                    plt.close(fig_importance)
            except Exception as e:
                st.warning(f"Could not generate feature importance for {model_name}: {e}")

    # Geo Analysis (only if geo columns exist)
    if any(col in st.session_state.full_df.columns for col in ['src_country', 'dst_country', 'src_lat', 'src_lon']):
        st.markdown("""
        <div class='premium-card'>
            <h2 style='color: #8e44ad; text-align: center; margin-bottom: 20px;'>🌍 Advanced Geographical Threat Analysis</h2>
        </div>
        """, unsafe_allow_html=True)
        
        full_df = st.session_state.full_df

        if 'src_country' in full_df.columns or 'dst_country' in full_df.columns:
            st.subheader("World Map Threat Distribution")

            country_codes = {
                'USA': 'USA', 'United States': 'USA',
                'China': 'CHN', 'Germany': 'DEU', 'France': 'FRA',
                'Russia': 'RUS', 'Japan': 'JPN', 'UK': 'GBR',
                'Canada': 'CAN', 'Australia': 'AUS', 'India': 'IND',
                'Brazil': 'BRA', 'Mexico': 'MEX', 'South Korea': 'KOR'
            }

            if 'src_country' in full_df.columns:
                src_analysis = full_df.groupby('src_country').agg({
                    'label': 'count'
                }).reset_index()
                src_analysis.columns = ['country', 'threat_count']
                src_analysis['country_code'] = src_analysis['country'].map(country_codes)

                fig_src_map = px.choropleth(
                    src_analysis,
                    locations='country_code',
                    color='threat_count',
                    hover_name='country',
                    title='Threat Origins by Source Country',
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig_src_map, use_container_width=True)

            if 'dst_country' in full_df.columns:
                dst_analysis = full_df.groupby('dst_country').agg({
                    'label': 'count'
                }).reset_index()
                dst_analysis.columns = ['country', 'threat_count']
                dst_analysis['country_code'] = dst_analysis['country'].map(country_codes)

                fig_dst_map = px.choropleth(
                    dst_analysis,
                    locations='country_code',
                    color='threat_count',
                    hover_name='country',
                    title='Threat Targets by Destination Country',
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig_dst_map, use_container_width=True)

        st.subheader("Detailed Country-Threat Analysis")
        fig_src, fig_dst = pipeline.plot_geo_distribution(full_df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_src, use_container_width=True)
        with col2:
            st.plotly_chart(fig_dst, use_container_width=True)

    # Threat Class Distribution (only if threat_label exists)
    if 'threat_label' in st.session_state.full_df.columns:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #8e44ad; text-align: center;'>📊 Threat Class Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        threat_counts = st.session_state.full_df['threat_label'].value_counts()
        fig_pie = px.pie(values=threat_counts.values, names=threat_counts.index,
                         title='Overall Threat Class Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Network Flow Analysis (only if both src_country and dst_country exist)
    if 'src_country' in st.session_state.full_df.columns and 'dst_country' in st.session_state.full_df.columns:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #8e44ad; text-align: center;'>🌐 Network Flow Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        country_flow = st.session_state.full_df.groupby(['src_country', 'dst_country']).size().reset_index(name='count')
        top_flows = country_flow.nlargest(10, 'count')

        fig_flow = px.bar(top_flows, x='count', y='src_country', 
                          orientation='h', title='Top 10 Threat Flows (Source → Destination)',
                          hover_data=['dst_country'])
        st.plotly_chart(fig_flow, use_container_width=True)

    # Export Results
    st.markdown("""
    <div class='premium-card'>
        <h2 style='color: #8e44ad; text-align: center; margin-bottom: 20px;'>💾 Export Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📥 Download Results as CSV"):
        pipeline.save_results_to_csv(results)

# Breach Detection Section
elif detect_breaches_btn:
    if st.session_state.full_df is None:
        st.error("❌ Please analyze network data first before breach detection.")
    else:
        with st.spinner('🔍 Identifying breaches and implementing protection...'):
            try:
                # Perform breach analysis
                breach_results = pipeline.identify_and_protect_breaches(
                    st.session_state.full_df,
                    sensitivity=sensitivity,
                    min_confidence=min_confidence,
                    auto_quarantine=auto_quarantine,
                    block_malicious_ips=block_malicious_ips,
                    alert_security_team=alert_security_team,
                    backup_critical_data=backup_critical_data
                )
                
                st.session_state.breach_results = breach_results
                display_breach_analysis_results(breach_results)
                
            except Exception as e:
                st.error(f"❌ Error in breach detection: {e}")

# Display previous breach results if available
elif st.session_state.breach_results is not None:
    st.markdown("""
    <div class='premium-card'>
        <h2 style='color: #e74c3c; text-align: center; margin-bottom: 20px;'>🛡️ Previous Breach Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    display_breach_analysis_results(st.session_state.breach_results)

else:
    # Welcome screen for authenticated users
    st.markdown("""
    <div class='premium-card'>
        <h2 style='color: #8e44ad; text-align: center; margin-bottom: 20px;'>Welcome to Cyber Threat Detection Framework</h2>
        <p style='color: #5d6d7e; text-align: center; line-height: 1.6;'>
            This tool leverages <strong>Hybrid GNN+XGBoost</strong> and <strong>PCA+Ensemble Classifiers</strong> 
            for advanced cyber threat detection. Upload your network traffic data to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='premium-card'>
            <h4 style='color: #8e44ad;'>🎯 How to use:</h4>
            <ol style='color: #5d6d7e;'>
                <li><strong>Upload</strong> your CSV file with network traffic data</li>
                <li>Ensure the dataset includes a <strong>'label'</strong> column</li>
                <li>Select the <strong>detection mode</strong></li>
                <li>Choose whether to apply <strong>PCA</strong></li>
                <li>Click <strong>"Analyze Threats"</strong></li>
                <li>Configure breach protection settings and click <strong>"Detect & Protect"</strong></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='premium-card'>
            <h4 style='color: #8e44ad;'>📋 Dataset Format:</h4>
            <ul style='color: #5d6d7e;'>
                <li><strong>Required:</strong> 'label' column (binary: 0=normal, 1=threat)</li>
                <li><strong>Optional:</strong> 'threat_label' column (string threat types)</li>
                <li>Features: numerical values (duration, src_bytes, etc.)</li>
                <li>protocol_type: categorical</li>
                <li>Optional: src_country, dst_country for geo analysis</li>
                <li>Optional: src_lat, src_lon, dst_lat, dst_lon for maps</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # New section explaining the two algorithms
    st.markdown("""
    <div class='premium-card'>
        <h4 style='color: #8e44ad; text-align: center;'>🧠 Two Advanced Algorithms</h4>
        <div style='display: flex; justify-content: space-between; margin-top: 20px;'>
            <div style='flex: 1; padding: 15px; background: rgba(155, 89, 182, 0.1); border-radius: 10px; margin: 5px;'>
                <h5 style='color: #8e44ad;'>🔗 Hybrid GNN+XGBoost</h5>
                <ul style='color: #5d6d7e; font-size: 0.9em;'>
                    <li>Graph Neural Networks for network structure analysis</li>
                    <li>XGBoost for feature-based classification</li>
                    <li>Combines topological and statistical patterns</li>
                    <li>Excellent for coordinated attack detection</li>
                </ul>
            </div>
            <div style='flex: 1; padding: 15px; background: rgba(52, 152, 219, 0.1); border-radius: 10px; margin: 5px;'>
                <h5 style='color: #3498db;'>📊 PCA+Ensemble Classifiers</h5>
                <ul style='color: #5d6d7e; font-size: 0.9em;'>
                    <li>PCA for dimensionality reduction</li>
                    <li>Multiple classifiers: Logistic Regression, SVM, Random Forest, KNN, XGBoost</li>
                    <li>Comprehensive performance comparison</li>
                    <li>Robust traditional ML approach</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Breach Protection Features
    st.markdown("""
    <div class='premium-card'>
        <h4 style='color: #e74c3c; text-align: center;'>🛡️ Advanced Breach Protection</h4>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;'>
            <div style='padding: 15px; background: rgba(231, 76, 60, 0.1); border-radius: 10px;'>
                <h6 style='color: #e74c3c;'>🔍 Breach Detection</h6>
                <ul style='color: #5d6d7e; font-size: 0.85em;'>
                    <li>Data exfiltration detection</li>
                    <li>Unauthorized access attempts</li>
                    <li>Malicious activity patterns</li>
                    <li>Statistical anomaly detection</li>
                </ul>
            </div>
            <div style='padding: 15px; background: rgba(46, 204, 113, 0.1); border-radius: 10px;'>
                <h6 style='color: #27ae60;'>🛡️ Protection Actions</h6>
                <ul style='color: #5d6d7e; font-size: 0.85em;'>
                    <li>Auto-quarantine suspicious nodes</li>
                    <li>Malicious IP blocking</li>
                    <li>Real-time security alerts</li>
                    <li>Critical data backup</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #e0e0e0; padding: 20px;'>
    <p style='font-size: 1.1em; margin-bottom: 10px;'>
        <strong style='color: #9b59b6;'>🛡️ Cyber Threat Detection Framework</strong>
    </p>
    <p style='font-size: 0.9em; opacity: 0.8;'>
        Hybrid GNN+XGBoost | PCA+Ensemble Classifiers | Breach Protection | Role-Based Access Control | Enterprise Security
    </p>
</div>
""", unsafe_allow_html=True)