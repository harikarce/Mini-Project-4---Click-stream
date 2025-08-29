import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load trained models
# ===========================
regression_model = joblib.load("regression_model.pkl")
classification_model = joblib.load("classification_model.pkl")
clustering_model = joblib.load("clustering_model.pkl")

# Define input fields
numeric_features = ['year','month','day','order','session_id','price_2','page']
categorical_features = ['country','page1_main_category','page2_clothing_model',
                        'colour','location','model_photography']

st.set_page_config(page_title="Customer Analytics App", layout="wide")
st.title("🛍️ Customer Analytics App")

# Tabs for Regression, Classification, Clustering
tab1, tab2, tab3 = st.tabs(["📈 Revenue Prediction", "🛒 Purchase Prediction", "👥 Customer Segmentation"])

# ==============================================================
# TAB 1: REGRESSION (Revenue Prediction)
# ==============================================================
with tab1:
    st.header("📈 Predict Customer Revenue")

    # CSV Upload
    uploaded_file = st.file_uploader("Upload CSV for Revenue Prediction", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        preds = regression_model.predict(df)
        df["Predicted_Revenue"] = preds
        st.dataframe(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "revenue_predictions.csv")
    else:
        inputs = {}
        for col in numeric_features + categorical_features:
            if col in numeric_features:
                inputs[col] = st.number_input(f"Enter {col}", value=0.0)
            else:
                inputs[col] = st.text_input(f"Enter {col}", value="Unknown")

        if st.button("Predict Revenue"):
            input_df = pd.DataFrame([inputs])
            revenue_pred = regression_model.predict(input_df)[0]
            st.success(f"💰 Estimated Revenue: **{revenue_pred:.2f}**")


# ==============================================================
# TAB 2: CLASSIFICATION (Purchase Prediction)
# ==============================================================
with tab2:
    st.header("🛒 Will the Customer Complete a Purchase?")

    uploaded_file = st.file_uploader("Upload CSV for Purchase Prediction", type=["csv"], key="clf_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        preds = classification_model.predict(df)
        df["Purchase_Prediction"] = preds
        st.dataframe(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "purchase_predictions.csv")
    else:
        inputs = {}
        for col in numeric_features + categorical_features:
            if col in numeric_features:
                inputs[col] = st.number_input(f"Enter {col}", value=0.0, key=f"clf_{col}")
            else:
                inputs[col] = st.text_input(f"Enter {col}", value="Unknown", key=f"clf_{col}")

        if st.button("Predict Purchase"):
            input_df = pd.DataFrame([inputs])
            purchase_pred = classification_model.predict(input_df)[0]
            result = "✅ Customer will Purchase" if purchase_pred == 1 else "❌ Customer will NOT Purchase"
            st.success(result)


# ==============================================================
# TAB 3: CLUSTERING (Customer Segmentation)
# ==============================================================
with tab3:
    st.header("👥 Customer Segmentation")

    uploaded_file = st.file_uploader("Upload CSV for Segmentation", type=["csv"], key="clust_upload")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        preds = clustering_model.predict(df)
        df["Cluster"] = preds
        st.dataframe(df)
        st.download_button("Download Clusters", df.to_csv(index=False), "customer_segments.csv")
    else:
        inputs = {}
        for col in numeric_features + categorical_features:
            if col in numeric_features:
                inputs[col] = st.number_input(f"Enter {col}", value=0.0, key=f"clust_{col}")
            else:
                inputs[col] = st.text_input(f"Enter {col}", value="Unknown", key=f"clust_{col}")

        if st.button("Find Customer Segment"):
            input_df = pd.DataFrame([inputs])
            cluster_pred = clustering_model.predict(input_df)[0]
            st.success(f"🧩 Customer belongs to **Cluster {cluster_pred}**")
