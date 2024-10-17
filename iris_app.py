import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Load the converted dataset
@st.cache_data
def load_data():
    df = pd.read_csv('append_BMI_converted.csv')
    return df

df = load_data()

# Define the target column for obesity prediction (assuming ob_BMI as the indicator for obesity)
df = df.dropna(subset=['bmxbmi', 'ob_BMI'])  # Replace 'ob_BMI' with actual obesity column

# Features and target for prediction
X = df[['bmxbmi']]  # Using BMI as the feature for simplicity
y = df['ob_BMI']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sidebar for classifier selection, including GBM
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Random Forest", "Naive Bayes", "Gradient Boosting Machine")
)

# Display dataset
st.write("Dataset Overview")
st.dataframe(df.head(10))  # Displaying top 10 rows

# Sidebar input slider for BMI
st.sidebar.title("Input Features")
bmi_value = st.sidebar.slider("BMI Value", float(df['bmxbmi'].min()), float(df['bmxbmi'].max()))

input_data = [[bmi_value]]

# Model selection and prediction
if classifier_name == "Random Forest":
    model = RandomForestClassifier()
elif classifier_name == "Naive Bayes":
    model = GaussianNB()
elif classifier_name == "Gradient Boosting Machine":
    model = GradientBoostingClassifier()

# Fit the selected model on the training data
model.fit(X_train, y_train)

# Calculate the accuracy on the test data
accuracy = model.score(X_test, y_test)

# Make prediction
prediction = model.predict(input_data)
predicted_obesity_status = "Obese" if prediction[0] == 1 else "Not Obese"

# Display prediction results
st.write(f"The selected classifier is: {classifier_name}")
st.write("Prediction")
st.write(f"The predicted obesity status is: {predicted_obesity_status}")

# Display the input data and prediction
result_df = pd.DataFrame(input_data, columns=['BMI'])
result_df['Predicted Obesity Status'] = predicted_obesity_status
st.write("Prediction Dataframe")
st.dataframe(result_df)

# Display accuracy
st.write(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
