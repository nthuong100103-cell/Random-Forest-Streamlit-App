import streamlit as st
import pandas as pd
import joblib

# =====================
# Load artifacts
# =====================
MODEL_PATH = "models/RandomForest_best.pkl"
SCALER_PATH = "models/RandomForest_scaler.pkl"
ENCODER_PATH = "models/RandomForest_label_encoders.pkl"
FEATURE_PATH = "models/RandomForest_important_features.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    important_features = joblib.load(FEATURE_PATH)
    return model, scaler, label_encoders, important_features

model, scaler, label_encoders, important_features = load_artifacts()

# =====================
# Page config
# =====================
st.set_page_config(
    page_title="Dự đoán ý định mua hàng",
    layout="wide"
)

# =====================
# UI style
# =====================
st.markdown("""
<style>
.header {
    background-color: #2563eb;
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.section {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# Header
# =====================
st.markdown("""
<div class="header">
    <h2>Hệ thống dự đoán ý định mua hàng</h2>
    <p>Ứng dụng mô hình Random Forest để dự đoán khả năng mua hàng</p>
</div>
""", unsafe_allow_html=True)

# =====================
# Input section
# =====================
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("🔎 Thông tin khách truy cập")

input_data = {}

num_cols = 4
for i in range(0, len(important_features), num_cols):
    cols = st.columns(num_cols)
    for col, feature in zip(cols, important_features[i:i + num_cols]):
        with col:
            if feature in label_encoders:
                options = label_encoders[feature].classes_
                input_data[feature] = st.selectbox(
                    feature, options
                )
            else:
                input_data[feature] = st.number_input(
                    feature, min_value=0.0, value=0.0
                )

st.markdown("</div>", unsafe_allow_html=True)

# =====================
# Prediction
# =====================
if st.button("🔮 Dự đoán"):
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, encoder in label_encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.subheader("📊 Kết quả dự đoán")

    if prediction == 1:
        st.success("✅ Khách hàng **CÓ khả năng mua hàng**")
    else:
        st.warning("❌ Khách hàng **KHÔNG có khả năng mua hàng**")

    st.write("Xác suất dự đoán:")
    st.dataframe(
        pd.DataFrame({
            "Lớp": model.classes_,
            "Xác suất": probability
        })
    )
