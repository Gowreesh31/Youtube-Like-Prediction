import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time

from src.data.get_data import get_full_video_stats
from src.features.build_features import engineer_features

# Ensure must act as main module when run
st.set_page_config(
    page_title="YouTube Predictor AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom Premium CSS ----
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E2129;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00FFCA;
    }
    .metric-label {
        font-size: 1rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# ---- Load Model ----
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("models/xgb_model.joblib")
        features = joblib.load("models/xgb_model_features.joblib")
        return model, features
    except Exception as e:
        return None, None

model, feature_names = load_model_artifacts()

# ---- Sidebar ----
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png", width=150)
    st.title("Navigation")
    page = st.radio("Go to:", ["Prediction Hub", "Model Analytics Hub"])
    
    st.markdown("---")
    st.markdown("### System Status")
    if model:
        st.success("✅ Model Loaded (XGBoost)")
    else:
        st.error("❌ Model Not Found")
        
    st.markdown("### About")
    st.info("A state-of-the-art ML system to predict YouTube video engagement.")

# ---- Helper Components ----
def display_metric_card(label, value, prefix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{prefix}{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ---- Page 1: Prediction Hub ----
if page == "Prediction Hub":
    st.title("🎯 Real-Time Prediction Hub")
    st.write("Enter a YouTube Video ID to run a real-time extraction and engagement prediction.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            video_id = st.text_input("Enter Video ID (e.g., dQw4w9WgXcQ):", placeholder="dQw4w9WgXcQ")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True) # spacing
            submit_btn = st.form_submit_button("Run Analytics Engine 🚀", use_container_width=True)
            
    if submit_btn and video_id:
        if not model:
            st.error("Cannot predict: Model is missing.")
        else:
            with st.spinner("Initiating YouTube Data API extraction..."):
                try:
                    # 1. Fetch Data
                    raw_data = get_full_video_stats([video_id])
                    time.sleep(1) # Dramatic effect
                    
                    if not raw_data:
                        st.error("Error: Could not retrieve data. Is the API key valid and video ID correct?")
                    else:
                        st.toast("Data extraction successful!", icon="✅")
                        
                        df_raw = pd.DataFrame(raw_data)
                        
                        # 2. Extract specific stats for display
                        v_title = df_raw.iloc[0]['title']
                        actual_likes = df_raw.iloc[0]['like_count']
                        actual_views = df_raw.iloc[0]['view_count']
                        actual_comments = df_raw.iloc[0]['comment_count']
                        
                        st.subheader(f"Analyzing: *{v_title}*")
                        
                        # 3. Engineer Features & Predict
                        with st.spinner("Engineering features & running inference..."):
                            X_infer = engineer_features(df_raw, is_training=False)
                            # Ensure columns match training
                            for col in feature_names:
                                if col not in X_infer.columns:
                                    X_infer[col] = 0
                            X_infer = X_infer[feature_names]
                            
                            prediction = model.predict(X_infer)[0]
                            prediction = max(0, int(prediction))
                        
                        st.markdown("---")
                        
                        # Display Results
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            display_metric_card("Predicted Likes (AI)", f"{prediction:,}")
                        with res_col2:
                            display_metric_card("Actual Current Likes", f"{actual_likes:,}")
                            
                        # Context Stats
                        st.markdown("### Current Video Statistics")
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        with stat_col1:
                            display_metric_card("Total Views", f"{actual_views:,}")
                        with stat_col2:
                            display_metric_card("Total Comments", f"{actual_comments:,}")
                        with stat_col3:
                            like_ratio = (actual_likes / max(1, actual_views)) * 100
                            display_metric_card("Engagement Rate", f"{like_ratio:.2f}", prefix="% ")

                        # Show feature breakdown
                        with st.expander("View AI Interpretation (Feature Importance for this inference)"):
                            # A simple mock interaction chart (SHAP is better, but this is a lightweight sim)
                            st.write("Top 5 Drivers for this prediction based on global feature importance:")
                            importance = model.feature_importances_
                            feat_df = pd.DataFrame({'Feature': feature_names, 'Influence': importance})
                            feat_df = feat_df.sort_values(by='Influence', ascending=False).head(5)
                            fig = px.bar(feat_df, x='Influence', y='Feature', orientation='h', template='plotly_dark')
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction Pipeline Failed: {e}")

# ---- Page 2: Model Analytics Hub ----
elif page == "Model Analytics Hub":
    st.title("📊 Model Analytics & EDA")
    st.write("Explore dataset distributions, feature correlations, and model performance metrics.")
    
    st.info("Loading system metrics...")
    
    tab1, tab2 = st.tabs(["Feature Explanations & Global Importance", "Dataset Distributions"])
    
    with tab1:
        st.subheader("Global Feature Importance (XGBoost)")
        if model:
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            fig = px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h',
                         title="Top 10 Most Critical Features", template='plotly_dark',
                         color='Importance', color_continuous_scale='Tealgrn')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Insights:**
            - Features like `log_view_count` and `log_comment_count` often dominate because they are direct proxies for visibility and engagement.
            - Engineered interaction ratios (`comments_per_view`) provide the model with context relative to channel size.
            """)
        else:
            st.warning("Model not available to show importance.")
            
    with tab2:
        st.subheader("Historic Data Sample Distributions")
        
        # High-level explanation for the user
        st.markdown("""
        > [!NOTE]
        > **Data Management Notice:** To maintain a lightweight repository, the primary training dataset (`dataset/data_final.csv`) 
        > is not bundled with the source code. You can re-generate this data by running the training pipeline 
        > or by using the script: `src/data/get_data.py`.
        """)
        
        if os.path.exists("dataset/data_final.csv"):
            try:
                # Load a tiny sample of the dataset just for EDA display purposes
                df_sample = pd.read_csv("dataset/data_final.csv")
                sample_size = min(5000, len(df_sample))
                df_sample = df_sample.sample(n=sample_size, random_state=42)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.scatter(df_sample, x='viewCount', y='likeCount', log_x=True, log_y=True,
                                      title="Views vs Likes (Log Scale)", template="plotly_dark", opacity=0.5,
                                      color_discrete_sequence=['#00FFCA'])
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = px.histogram(df_sample, x='likeCount', log_y=True,
                                        title="Distribution of Likes", template="plotly_dark",
                                        color_discrete_sequence=['#00BBFF'])
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load EDA visuals: {e}")
        else:
            st.warning("Training dataset not found locally. Visualizations on this tab require `dataset/data_final.csv`.")
            st.button("Learn how to reconstruct data", on_click=lambda: st.info("Run `python src/data/get_data.py` to fetch a new batch of data from the YouTube API."))
