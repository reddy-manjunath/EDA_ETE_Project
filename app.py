import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Reorder Prediction", layout="wide")

# --- Custom CSS Styling ---

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    :root {
        --primary-color: #0b3d91;
        --secondary-color: #1f77b4;
        --success-color: #2ca02c;
        --warning-color: #ff7f0e;
    }
    
    .main {
        background-color: #FFFDD0;
        padding: 20px;
    }
    
    .stApp {
        background-color: #FFFDD0;
    }
    
    /* --- HEADER COMPONENT CHANGED HERE --- */
    .header-box {
        background: white;        /* Changed to white */
        color: #0b3d91;           /* Text changed to blue for visibility */
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    /* ------------------------------------- */
    
    .metric-card {
        background: white;
        border-left: 5px solid #0b3d91;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .visualization-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
    }
    
    .stButton > button {
        background-color: #0b3d91;
        color: white;
        border-radius: 8px;
        padding: 10px 30px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1f77b4;
        box-shadow: 0 4px 12px rgba(11, 61, 145, 0.4);
    }
    
    h1, h2, h3 {
        color: #0b3d91 !important;
        font-weight: 700;
    }
    
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #444444 !important;
    }

    [data-testid="stMetricValue"] {
        color: #0b3d91 !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Load Artifacts ---
MODEL_FILE = "best_xgb_model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
DATA_FILE = "final_dataset.csv"


def load_artifacts():
    base = Path(".")
    model_path = base / MODEL_FILE
    preproc_path = base / PREPROCESSOR_FILE
    feat_path = base / FEATURE_NAMES_FILE

    if not model_path.exists() or not preproc_path.exists() or not feat_path.exists():
        missing = [p.name for p in (model_path, preproc_path, feat_path) if not p.exists()]
        st.error(f"❌ Missing artifact files: {', '.join(missing)}")
        st.stop()

    model = joblib.load(model_path)
    preprocessor = joblib.load(preproc_path)
    feature_names = joblib.load(feat_path)
    return model, preprocessor, feature_names


model, preprocessor, feature_names = load_artifacts()

# Define the feature names expected by the pipeline
selected_features = [
    "user_id",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
    "product_id",
    "add_to_cart_order",
    "aisle_id",
    "department_id",
    "department",
]

# --- Title & Header ---
st.markdown(
    "<div class='header-box'><h1>🛒 Product Reorder Prediction System</h1><p style='font-size:16px;'>Predict whether a customer will reorder a product using ML</p></div>",
    unsafe_allow_html=True,
)

# --- Sidebar Input ---
st.sidebar.markdown("### 📋 Enter Order Details")

# Inputs with better layout
user_id = st.sidebar.number_input("👤 User ID", min_value=1, value=12345, step=1)
order_number = st.sidebar.number_input("📦 Order Number (count)", min_value=1, value=5, step=1)
order_dow = st.sidebar.selectbox("📅 Day of Week", options=list(range(7)), format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x])
order_hour = st.sidebar.slider("⏰ Hour of Day", min_value=0, max_value=23, value=14)
days_since_prior = st.sidebar.number_input("⏳ Days Since Prior Order (-1 if first)", min_value=-1, value=7, step=1)
product_id = st.sidebar.number_input("🏷️ Product ID", min_value=0, value=6789, step=1)
add_to_cart_order = st.sidebar.number_input("🛒 Add-to-cart Order", min_value=1, value=3, step=1)
aisle_id = st.sidebar.number_input("🏪 Aisle ID", min_value=0, value=10, step=1)
department_id = st.sidebar.number_input("🏬 Department ID", min_value=0, value=4, step=1)
department = st.sidebar.text_input("🏢 Department Name", value="produce")

# Build DataFrame
input_dict = {
    "user_id": int(user_id),
    "order_number": int(order_number),
    "order_dow": int(order_dow),
    "order_hour_of_day": int(order_hour),
    "days_since_prior_order": int(days_since_prior),
    "product_id": int(product_id),
    "add_to_cart_order": int(add_to_cart_order),
    "aisle_id": int(aisle_id),
    "department_id": int(department_id),
    "department": str(department),
}

df_input = pd.DataFrame([input_dict], columns=selected_features)

# --- Main Content Layout ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### 📊 Input Summary")
    st.write(df_input.T)
    
    
    predict_btn = st.button("🎯 Make Prediction", use_container_width=True)

with col2:
    result_container = st.container()

# --- Prediction Logic ---
if predict_btn:
    try:
        X_proc = preprocessor.transform(df_input)
        
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_proc)[:, 1][0])
        else:
            proba = float(model.predict(X_proc)[0])
        
        pred = int(model.predict(X_proc)[0])
        
        with result_container:
            # Prediction result
            
            st.markdown("## 🎯 PREDICTION RESULT")
            
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                status = "✅ WILL REORDER" if pred == 1 else "❌ WON'T REORDER"
                st.markdown(f"<h3 style='color:white'>{status}</h3>", unsafe_allow_html=True)
            
            with col_pred2:
                st.markdown(f"<h3 style='color:white'>Confidence: {proba*100:.1f}%</h3>", unsafe_allow_html=True)
            
            
            
            # Probability bar
            
            st.markdown("### 📈 Reorder Probability")
            
            color = "#2ca02c" if proba > 0.6 else "#ff7f0e" if proba > 0.4 else "#d62728"
            st.markdown(
                f"<div style='background:#e8e8e8;border-radius:10px;padding:8px;overflow:hidden'>"
                f"<div style='width:{proba*100}%;background:{color};padding:10px;border-radius:8px;text-align:center;color:white;font-weight:bold;transition:all 0.3s ease'>"
                f"{proba*100:.1f}%</div></div>",
                unsafe_allow_html=True
            )
            
            
            # Feature importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if len(importances) == len(feature_names):
                    imp_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": importances
                    }).sort_values("importance", ascending=False).head(10)
                    
                    
                    st.markdown("### 🔝 Top 10 Important Features")
                    
                    # Chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(
                        data=imp_df,
                        y='feature',
                        x='importance',
                        palette='viridis',
                        ax=ax
                    )
                    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

# --- Data Visualizations Section ---
st.markdown("---")
st.markdown("<h2 style='text-align:center; color:#0b3d91'>📊 Dataset Insights & Analysis</h2>", unsafe_allow_html=True)

try:
    data_path = Path(DATA_FILE)
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        # Row 1: Top Products & Top Departments
        viz_row1_col1, viz_row1_col2 = st.columns(2, gap="medium")
        
        with viz_row1_col1:
            st.markdown("### 🏆 Top 10 Products by Orders")
            top_products = df['product_name'].value_counts().head(10)
            fig_prod, ax_prod = plt.subplots(figsize=(8, 5))
            sns.barplot(
                y=top_products.index,
                x=top_products.values,
                palette='coolwarm',
                ax=ax_prod
            )
            ax_prod.set_xlabel('Order Count', fontsize=11, fontweight='bold')
            ax_prod.set_ylabel('Product Name', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_prod)
        
        with viz_row1_col2:
         
            st.markdown("### 🏬 Top 10 Departments by Orders")
            top_dept = df['department'].value_counts().head(10)
            fig_dept, ax_dept = plt.subplots(figsize=(8, 5))
            sns.barplot(
                x=top_dept.values,
                y=top_dept.index,
                palette='magma',
                ax=ax_dept
            )
            ax_dept.set_xlabel('Order Count', fontsize=11, fontweight='bold')
            ax_dept.set_ylabel('Department', fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_dept)
            
        
        # Row 2: Hourly Orders & Day of Week Orders
        viz_row2_col1, viz_row2_col2 = st.columns(2, gap="medium")
        
        with viz_row2_col1:
        
            st.markdown("### ⏰ Orders by Hour of Day")
            hourly = df.groupby('order_hour_of_day').size()
            fig_hour, ax_hour = plt.subplots(figsize=(10, 4))
            ax_hour.bar(hourly.index, hourly.values, color='steelblue', edgecolor='navy', alpha=0.8)
            ax_hour.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
            ax_hour.set_ylabel('Order Count', fontsize=11, fontweight='bold')
            ax_hour.set_title('Peak Ordering Times', fontsize=12, fontweight='bold')
            ax_hour.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_hour)
            
        
        with viz_row2_col2:
            
            st.markdown("### 📅 Orders by Day of Week")
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            daily = df.groupby('order_dow').size()
            fig_day, ax_day = plt.subplots(figsize=(8, 4))
            colors_day = plt.cm.Set3(np.linspace(0, 1, 7))
            ax_day.bar(daily.index, daily.values, color=colors_day, edgecolor='black', alpha=0.8)
            ax_day.set_xticks(range(7))
            ax_day.set_xticklabels(day_names, rotation=45)
            ax_day.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
            ax_day.set_ylabel('Order Count', fontsize=11, fontweight='bold')
            ax_day.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_day)
            
        
        # Row 3: Reorder Rates
        viz_row3_col1, viz_row3_col2 = st.columns(2, gap="medium")
        
        with viz_row3_col1:
            
            st.markdown("### 📈 Reorder Rate by Hour")
            reorder_hour = df.groupby('order_hour_of_day')['reordered'].mean()
            fig_reorder_h, ax_reorder_h = plt.subplots(figsize=(10, 4))
            ax_reorder_h.plot(reorder_hour.index, reorder_hour.values, marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
            ax_reorder_h.fill_between(reorder_hour.index, reorder_hour.values, alpha=0.3, color='#2ca02c')
            ax_reorder_h.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
            ax_reorder_h.set_ylabel('Reorder Rate', fontsize=11, fontweight='bold')
            ax_reorder_h.grid(True, alpha=0.3)
            ax_reorder_h.set_ylim([0, 1])
            plt.tight_layout()
            st.pyplot(fig_reorder_h)
            
        
        with viz_row3_col2:
            
            st.markdown("### 📈 Reorder Rate by Day of Week")
            reorder_day = df.groupby('order_dow')['reordered'].mean()
            fig_reorder_d, ax_reorder_d = plt.subplots(figsize=(8, 4))
            ax_reorder_d.plot(reorder_day.index, reorder_day.values, marker='s', linewidth=2.5, markersize=8, color='#ff7f0e')
            ax_reorder_d.fill_between(reorder_day.index, reorder_day.values, alpha=0.3, color='#ff7f0e')
            ax_reorder_d.set_xticks(range(7))
            ax_reorder_d.set_xticklabels(day_names, rotation=45)
            ax_reorder_d.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
            ax_reorder_d.set_ylabel('Reorder Rate', fontsize=11, fontweight='bold')
            ax_reorder_d.grid(True, alpha=0.3)
            ax_reorder_d.set_ylim([0, 1])
            plt.tight_layout()
            st.pyplot(fig_reorder_d)
            
        
        # Summary stats
        
        st.markdown("### 📊 Dataset Summary")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        col_stat1.metric("Total Orders", f"{len(df):,}")
        col_stat2.metric("Unique Products", f"{df['product_id'].nunique():,}")
        col_stat3.metric("Reorder Rate", f"{df['reordered'].mean()*100:.1f}%")
        col_stat4.metric("Unique Users", f"{df['user_id'].nunique():,}")
    
    else:
        st.info(f"💡 Optional dataset '{DATA_FILE}' not found. Visualizations will appear once the data file is placed in this folder.")

except Exception as e:
    st.warning(f"⚠️ Could not load visualizations: {str(e)}")

# --- Footer ---
st.markdown("---")

