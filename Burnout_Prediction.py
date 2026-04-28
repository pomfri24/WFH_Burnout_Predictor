import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------
# 1. Page Configuration & Custom CSS
# --------------------------
# Responsibility: Set up Streamlit page configuration and apply custom CSS for a professional look
st.set_page_config(page_title="Burnout Predictor", layout="wide")

# Custom CSS to style the Streamlit interface with a professional gradient background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #1f3c88, #6a1b9a);
        color: white;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        width: 100%;
        border-radius: 5px;
    }
    label, .stSlider, .stSelectbox {
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# 2. Data Processing & Model Training
# --------------------------
@st.cache_data
def load_and_train():
    # Responsibility: Data cleaning and formatting
    # Loading the synthetic burnout dataset
    df = pd.read_csv("work_from_home_burnout_dataset.csv")

    # Cleaning column names by stripping any leading/trailing whitespace (Defensive Programming)
    df.columns = df.columns.str.strip()

    # Responsibility: Data preprocessing and model training
    # Encoding categorical variables (Day Type and Burnout Risk) into numerical values for the model
    # Handle missing values
    df = df.dropna()  
    df = df.drop_duplicates()
    le_day = LabelEncoder()
    df['day_type_encoded'] = le_day.fit_transform(df['day_type'])

    le_target = LabelEncoder()
    df['burnout_risk_encoded'] = le_target.fit_transform(df['burnout_risk'])

    # Defining the specific behavioral features used for machine learning prediction
    features = ['day_type_encoded', 'work_hours', 'screen_time_hours', 'meetings_count',
                'breaks_taken', 'after_hours_work', 'sleep_hours', 'task_completion_rate', 'burnout_score']

    X = df[features]
    y = df['burnout_risk_encoded']

    # Balancing classes using SMOTE to ensure the model learns "High Risk" cases effectively
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Responsibility: Implement train-test split
    # Splitting data so 20% is reserved for evaluating model performance and accuracy
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Training the Random Forest Classifier with balanced class weights
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    return rf_model, le_target, X_test, y_test, df, features

# Responsibility: Execute the training function and store results in variables
model, le_target, X_test, y_test, full_df, feature_cols = load_and_train()

# --------------------------
# 3. Sidebar - Input Features (UI)
# --------------------------
# Responsibility: Create interactive sidebar for user input of daily work parameters
# These interactive fields allow users to enter their daily work parameters
st.sidebar.header("📋 Employee Work Metrics")
day_type = st.sidebar.selectbox("Day Type", ("Weekday", "Weekend"))
work_hours = st.sidebar.slider("Work Hours", 0, 16, 8)
screen_time = st.sidebar.slider("Screen Time Hours", 0, 16, 6)
meetings = st.sidebar.slider("Meetings Count", 0, 15, 3)
breaks = st.sidebar.slider("Breaks Taken", 0, 10, 2)
after_hours = st.sidebar.selectbox("After Hours Work?", ("Yes", "No"))
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
task_rate = st.sidebar.slider("Task Completion Rate (%)", 0, 100, 85)
b_score = st.sidebar.slider("Current Burnout Subjective Score", 0, 150, 45)

# Mapping user-friendly inputs back into numerical format for the model prediction
day_val = 1 if day_type == "Weekday" else 0
after_val = 1 if after_hours == "Yes" else 0

input_data = pd.DataFrame([[
    day_val, work_hours, screen_time, meetings,
    breaks, after_val, sleep, task_rate, b_score
]], columns=feature_cols)

# --------------------------
# 4. Main UI - Predictions & Performance
# --------------------------
st.title("💻 The Burnout Predictor")
st.markdown("---")

# Layout using columns to display results and metrics side-by-side
col1, col2 = st.columns([1, 1])

# Responsibility: Implement model prediction and display results 
with col1:
    st.subheader("🚀 Live Prediction")
    # Triggering the model prediction when the user clicks the button
    if st.button("Analyze Risk"):
        prediction = model.predict(input_data)
        risk_label = le_target.inverse_transform(prediction)[0]

        # Applying dynamic color coding based on the risk level
        color = "#ff4b4b" if risk_label == "High" else "#ffa500" if risk_label == "Medium" else "#28a745"
        st.markdown(f"<h2 style='text-align: center; color: {color};'>{risk_label} Risk</h2>", unsafe_allow_html=True)

        # Providing specific guidance based on the machine learning result
        if risk_label == "High":
            st.warning("Action Required: High risk detected. Consider mandatory time off and meeting reduction.")
        elif risk_label == "Medium":
            st.info("Precaution: Moderate risk. Monitor workload and encourage more frequent breaks.")
        else:
            st.success("Maintaining Balance: Continue current healthy work patterns and wellness habits.")

with col2:
    # Responsibility: Evaluate model performance
    st.subheader("📈 Model Evaluation")
    y_pred = model.predict(X_test)

    # Displaying accuracy to show the model's reliability
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Current Accuracy:** {acc:.2%}")

    # Generating and visualizing the Confusion Matrix for performance analysis
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    plt.title("Confusion Matrix")
    st.pyplot(fig_cm)

# --------------------------
# 5. Interactive Visualizations
# --------------------------
# Responsibility: Create interactive visualizations to explore the dataset and model insights
st.markdown("---")
st.subheader("📊 Interactive Data Visualizations")

# Dropdown to allow users to explore different aspects of the underlying data
vis_option = st.selectbox("Choose a visualization to explore the dataset:", [
    "Feature Importance (Key Drivers of Burnout)",
    "Work Hours vs. Sleep (Correlation)",
    "Burnout Risk Distribution"
])

if vis_option == "Feature Importance (Key Drivers of Burnout)":
    # Analyzing which features have the most predictive power in the Random Forest model
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig = px.bar(importances, orientation='h', title="Which factors impact burnout most?",
                 labels={'value': 'Importance Score', 'index': 'Factor'},
                 color=importances, color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

elif vis_option == "Work Hours vs. Sleep (Correlation)":
    # Visualizing the relationship between work intensity and sleep quality
    fig = px.scatter(full_df, x="work_hours", y="sleep_hours", color="burnout_risk",
                     title="Impact of Work Intensity on Sleep Quality",
                     labels={"work_hours": "Hours Worked", "sleep_hours": "Hours Slept"})
    st.plotly_chart(fig, use_container_width=True)

elif vis_option == "Burnout Risk Distribution":
    # Showing the breakdown of risk levels within the balanced training data
    fig = px.histogram(full_df, x="burnout_risk", color="burnout_risk",
                       title="Total Counts of Risk Levels in Training Data",
                       category_orders={"burnout_risk": ["Low", "Medium", "High"]})
    st.plotly_chart(fig, use_container_width=True)
