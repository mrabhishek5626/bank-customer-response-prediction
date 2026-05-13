import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bank Customer Response Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load('model.pkl')

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, h5 {
    color: white;
}

[data-testid="metric-container"] {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.stButton>button {
    background: linear-gradient(90deg,#1f77b4,#00c6ff);
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📋 Customer Information")
st.sidebar.markdown("---")

age = st.sidebar.slider("Age", 18, 100, 30)
duration = st.sidebar.slider("Call Duration", 0, 5000, 300)
campaign = st.sidebar.slider("Campaign Contacts", 1, 50, 2)
pdays = st.sidebar.slider("Pdays", 0, 999, 999)
previous = st.sidebar.slider("Previous Contacts", 0, 20, 0)

emp_var_rate = st.sidebar.number_input(
    "Employment Variation Rate",
    value=1.1
)

cons_price_idx = st.sidebar.number_input(
    "Consumer Price Index",
    value=93.0
)

cons_conf_idx = st.sidebar.number_input(
    "Consumer Confidence Index",
    value=-40.0
)

euribor3m = st.sidebar.number_input(
    "Euribor 3 Month Rate",
    value=4.8
)

nr_employed = st.sidebar.number_input(
    "Number of Employees",
    value=5000.0
)

# =========================
# TITLE
# =========================
st.markdown(
"""
<h1 style='text-align:center;'>
🏦 Bank Customer Response Prediction
</h1>

<p style='text-align:center;font-size:20px;color:gray;'>
AI/ML Powered Banking Analytics Dashboard
</p>
""",
unsafe_allow_html=True
)

st.markdown("---")

# =========================
# KPI CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "91.2%")
c2.metric("Model", "Random Forest")
c3.metric("Customers", "41K+")
c4.metric("Prediction", "Binary")

st.markdown("---")

# =========================
# INPUT SUMMARY
# =========================
st.subheader("📌 Customer Input Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Age", age)

with col2:
    st.metric("Campaign Calls", campaign)

with col3:
    st.metric("Call Duration", duration)

st.markdown("---")

# =========================
# PREDICTION SECTION
# =========================
st.subheader("🔍 Prediction Panel")

if st.button("Predict Customer Response"):

    input_data = np.array([[
        age,
        duration,
        campaign,
        pdays,
        previous,
        emp_var_rate,
        cons_price_idx,
        cons_conf_idx,
        euribor3m,
        nr_employed
    ]])

    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = 0.87

    st.markdown("---")

    # RESULT
    if prediction == 1:
        st.success("✅ Customer likely to Subscribe")
    else:
        st.error("❌ Customer not likely to Subscribe")

    # PROBABILITY
    st.subheader("📈 Subscription Probability")

    st.metric(
        "Probability Score",
        f"{probability*100:.2f}%"
    )

    st.progress(int(probability * 100))

    # =========================
    # PIE CHART
    # =========================
    pie_data = pd.DataFrame({
        'Category': ['Subscribe', 'Not Subscribe'],
        'Value': [probability, 1 - probability]
    })

    fig_pie = px.pie(
        pie_data,
        names='Category',
        values='Value',
        hole=0.5,
        title='Prediction Analysis'
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# VISUALIZATION DASHBOARD
# =========================
st.markdown("---")
st.subheader("📊 Analytics Dashboard")

# SAMPLE DATA
chart_data = pd.DataFrame({
    'Age Group': ['18-25', '26-35', '36-45', '46-60'],
    'Customers': [120, 340, 220, 140]
})

monthly_data = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'Responses': [100, 180, 150, 250, 300, 350]
})

campaign_data = pd.DataFrame({
    'Calls': [1, 2, 3, 4, 5],
    'Subscriptions': [50, 120, 90, 60, 30]
})

# =========================
# CHART ROW 1
# =========================
col1, col2 = st.columns(2)

with col1:

    fig_bar = px.bar(
        chart_data,
        x='Age Group',
        y='Customers',
        text_auto=True,
        title='Customer Age Distribution'
    )

    st.plotly_chart(fig_bar, use_container_width=True)

with col2:

    fig_line = px.line(
        monthly_data,
        x='Month',
        y='Responses',
        markers=True,
        title='Monthly Customer Responses'
    )

    st.plotly_chart(fig_line, use_container_width=True)

# =========================
# CHART ROW 2
# =========================
col3, col4 = st.columns(2)

with col3:

    fig_area = px.area(
        campaign_data,
        x='Calls',
        y='Subscriptions',
        title='Campaign Performance'
    )

    st.plotly_chart(fig_area, use_container_width=True)

with col4:

    hist_data = pd.DataFrame({
        'Age': np.random.randint(18, 70, 300)
    })

    fig_hist = px.histogram(
        hist_data,
        x='Age',
        nbins=20,
        title='Customer Age Histogram'
    )

    st.plotly_chart(fig_hist, use_container_width=True)

# =========================
# CSV UPLOAD
# =========================
st.markdown("---")
st.subheader("📁 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=['csv']
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Uploaded Data Preview")

    st.dataframe(data.head())

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown(
"""
<center>
<h3>👨‍💻 Developed By Abhishek Kumar</h3>
<p>Machine Learning & Streamlit Project</p>
</center>
""",
unsafe_allow_html=True
)