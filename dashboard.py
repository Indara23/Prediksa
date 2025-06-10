import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np

# -----------------------
# Load model dan scaler
# -----------------------
with open("voting_soft_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("num_features.pkl", "rb") as f:
    num_features = pickle.load(f)

# -----------------------
# Load dataset untuk EDA
# -----------------------
df = pd.read_csv("heart.csv")

# -----------------------
# Kategori mapping
# -----------------------
sex_map = {"Female": 0, "Male": 1}
cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
fbs_map = {"No": 0, "Yes": 1}
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2, "Undefined": 3}

# -----------------------
# Sidebar: Logo dan Input
# -----------------------
st.sidebar.image("prediksa.png", use_container_width=True)
st.sidebar.header("📝 Input Client Health Data")

age = st.sidebar.number_input("Age", min_value=20, max_value=100, step=1)
sex = st.sidebar.selectbox("Sex", list(sex_map.keys()))
cp = st.sidebar.selectbox("Chest Pain Type", list(cp_map.keys()))
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200)
chol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL?", list(fbs_map.keys()))
restecg = st.sidebar.selectbox("Resting ECG Results", list(restecg_map.keys()))
thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
exang = st.sidebar.selectbox("Exercise Induced Angina?", list(exang_map.keys()))
oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=6.0, step=0.1)
slope = st.sidebar.selectbox("Slope of ST Segment", list(slope_map.keys()))
ca = st.sidebar.slider("Number of Major Vessels (0–3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", list(thal_map.keys()))

# -----------------------
# Main Area
# -----------------------
st.title("Heart Disease Classification for Insurance")
st.markdown('<p style="font-style: italic; color:gray">by Prediksa Company</p>', unsafe_allow_html=True)

st.markdown("""
This application helps **insurance professionals and individuals** predict the likelihood of heart disease and estimate insurance premiums using machine learning based on health indicators.

### How to Use:
👉 **Step 1:** Explore the dataset used by the model  
👉 **Step 2:** Use the sidebar form to provide client health data  
👉 **Step 3:** View the predicted risk of heart disease  
👉 **Step 4:** Get an **estimated insurance premium** based on your risk level  

This tool is designed to support early risk detection and help make informed decisions in the insurance industry.
""")

with st.expander("Tell me more about Heart Disease"):
    st.markdown("""
Heart disease is one of the leading causes of death both worldwide and in Indonesia. The heart is a vital organ responsible for pumping blood, delivering oxygen, and essential nutrients throughout the body. When the heart is impaired, it can affect other organs and overall health. Global health data shows that heart disease causes millions of deaths each year and is expected to keep increasing. In Indonesia, the rising cases of heart disease present serious challenges not only to public health but also to economic sectors like insurance.

For insurance companies, early identification of individuals at high risk of heart disease is crucial to managing risk accurately. Accurate risk prediction helps insurers set better policies, adjust premiums, and create targeted prevention programs, ultimately reducing potential financial losses.

Machine learning techniques, especially classification algorithms, have proven capable of handling complex medical data and providing fast, accurate predictions. One effective method is the voting classifier — which combines multiple models like logistic regression, k-nearest neighbors (KNN), and support vector machines (SVM) — to improve prediction accuracy by aggregating decisions through majority voting or probabilistic averaging.

In Indonesia, health datasets are generally well balanced, so oversampling methods like SMOTE can be minimized. This makes modeling simpler and more effective. Therefore, voting classifiers are a promising choice to build strong predictive models tailored to local conditions. By using this technology, insurance companies can better manage heart disease risks, leading to improved service and more informed business decisions.
""")


# -----------------------
# Feature Explanation Section
# -----------------------
st.markdown("---")
st.header("📋 Feature Explanation")
st.markdown("""
- **Age**: Patient age in years  
- **Sex**: Male or Female  
- **Chest Pain Type**: Type of chest pain  
- **Resting Blood Pressure**: Blood pressure at rest (mm Hg)  
- **Cholesterol**: Cholesterol level in mg/dl  
- **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (Yes/No)  
- **Resting ECG**: Resting electrocardiographic results  
- **Maximum Heart Rate**: Highest heart rate achieved during test  
- **Exercise-induced Angina**: Chest pain during exercise (Yes/No)  
- **Oldpeak**: ST depression induced by exercise relative to rest  
- **Slope**: The slope of the peak exercise ST segment  
- **Number of Major Vessels**: Number of major vessels colored by fluoroscopy  
- **Thalassemia**: Results of the thalassemia test  
- **Target**: Heart disease indication (Yes/No)  
""")

# -----------------------
# Exploratory Data Analysis (EDA)
# -----------------------
st.markdown("---")
st.header("🔍 Exploratory Data Analysis")

# Target Distribution
st.subheader("Target Distribution (Heart Disease)")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x="target", ax=ax1, palette=["#a6cee3", "#fb9a99"])  # pastel coolwarm tone
ax1.set_xticklabels(['No Disease', 'Disease'])
st.pyplot(fig1)
with st.expander("View description of target distribution chart"):
    st.write("""
    This chart displays the number of patients diagnosed with and without heart disease.  
    It highlights the relative balance between these two groups in the dataset, which is important for building reliable predictive models.
    """)


# Age Distribution
st.subheader("Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df["age"], bins=20, kde=True, ax=ax2)
st.pyplot(fig2)
with st.expander("View description of age distribution chart"):
    st.write("""
    The chart shows that the number of patients with heart disease generally increases with age, 
    peaking in their 50s. However, the highest number of individuals diagnosed with heart disease 
    is actually in their 40s. This suggests that factors beyond age, such as lifestyle or health 
    conditions, may play a significant role in the onset of heart disease.
    """)

# Sex vs Target Comparison
st.subheader("Sex vs Heart Disease")
fig3, ax3 = plt.subplots()
sns.countplot(data=df, x="sex", hue="target", ax=ax3, palette=["#f4c7c3", "#a3c4f3"])
ax3.set_xticklabels(["Female", "Male"])
ax3.legend(['No Disease', 'Disease'])
st.pyplot(fig3)
with st.expander("View description of sex comparison chart"):
    st.write("""
    Although the number of male patients is higher than female patients, the chart shows that both sexes have a significant proportion of heart disease cases. This highlights that while men may be more frequently diagnosed, women are also at considerable risk and should not be overlooked in risk assessments. This underlines the importance of heart disease screening for both males and females, regardless of overall prevalence.
    """)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)
with st.expander("View description of correlation heatmap"):
    st.write("""
    The heatmap displays the correlation matrix between feature variables and the target. The feature with the highest positive correlation to heart disease is thalach (maximum heart rate), with a value of 0.42, indicating that higher thalach values are associated with a greater risk of heart disease. Conversely, oldpeak (ST segment depression) has a negative correlation of -0.44 with the target, meaning that higher oldpeak values are linked to a lower risk of heart disease.
    """)

# -----------------------
# Prediction Section
# -----------------------
st.markdown("---")
st.header("🫀 Prediction Result")
st.info("ℹ️ Please fill in the required medical information in the **sidebar** to get a heart disease risk prediction.")

# Setelah prediksi selesai dan probabilitas proba sudah dihitung

if st.button("Predict"):
    input_data = np.array([[
        age,
        sex_map[sex],
        cp_map[cp],
        trestbps,
        chol,
        fbs_map[fbs],
        restecg_map[restecg],
        thalach,
        exang_map[exang],
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal]
    ]])

    # Scaling hanya fitur numerik
    input_df = {col: val for col, val in zip(
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        input_data.flatten()
    )}
    input_df = {k: [v] for k, v in input_df.items()}
    import pandas as pd
    input_df = pd.DataFrame.from_dict(input_df)

    input_df_scaled = input_df.copy()
    input_df_scaled[num_features] = scaler.transform(input_df[num_features])

    # Prediksi
    prediction = model.predict(input_df_scaled)[0]
    proba = model.predict_proba(input_df_scaled)[0][1]

    # simpan probabilitas ke session_state
    st.session_state["proba"] = proba

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease Detected! (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Disease Detected. (Probability: {proba:.2%})")


# ---------------------------------------------------
# Estimasi Premi 
# ---------------------------------------------------

st.markdown("---")
st.header("💰 Estimated Insurance Premium Calculation")

# Cek dulu apakah ada hasil prediksi (proba) di session_state
if "proba" in st.session_state:
    base_premium = st.number_input(
        "Enter base premium (По) in IDR", 
        min_value=0, value=1000000, step=10000, format="%d"
    )

    risk_level = st.session_state["proba"]
    final_premium = base_premium * (1 + risk_level)

    st.write(f"Calculated risk factor (R): {risk_level:.2%}")
    st.subheader("Premium to be paid by participant (П)")
    st.write(f"Rp {final_premium:,.0f}")

    with st.expander("Premium Calculation Formula Explanation"):
            st.markdown(r"""
            The premium paid by the participant is calculated using the formula:  
            $$
            П = По \times (1 + R)
            $$
            where:  
            - \(П\) is the premium to be paid (in rupiah)  
            - \(По\) is the base premium for participants without additional risk  
            - \(R\) is the additional risk factor (calculated from heart disease probability) expressed as a decimal (e.g., 0.2 for 20%)
            """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Please perform prediction first to get risk factor before calculating premium.")