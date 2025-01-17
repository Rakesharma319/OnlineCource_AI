import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('online_course_engagement_model1.pkl')

# Define the prediction function
def predict_course_completion(course_category, time_spent, videos_watched, quizzes_taken, quiz_scores, completion_rate, device_type):
  # Create a DataFrame from the input features
  input_data = pd.DataFrame([[course_category, time_spent, videos_watched, quizzes_taken, quiz_scores, completion_rate, device_type]], 
                         columns=['CourseCategory', 'TimeSpentOnCourse', 'NumberOfVideosWatched',
                                  'NumberOfQuizzesTaken', 'QuizScores', 'CompletionRate', 'DeviceType'])
  
  # Make the prediction
  prediction = model.predict(input_data)[0]
  
  return prediction

# Streamlit app
st.title("Online Course Completion Prediction")

# Input fields
course_category = st.selectbox("Course Category", ["Health", "Business", "Technology", "Art", "Other"])
time_spent = st.number_input("Time Spent on Course (hours)", min_value=0)
videos_watched = st.number_input("Number of Videos Watched", min_value=0)
quizzes_taken = st.number_input("Number of Quizzes Taken", min_value=0)
quiz_scores = st.number_input("Average Quiz Score", min_value=0, max_value=100)
completion_rate = st.number_input("Course Completion Rate (%)", min_value=0, max_value=100)

device_dict = {"Mobile/Tablet":1, "Desktop/Laptop":0}
device_type_cat = st.selectbox("Device Type", ["Mobile/Tablet", "Desktop/Laptop"])
device_type = device_dict[device_type_cat]

# device_type = st.number_input("Device Type", min_value=0, max_value=1)

# Display entered input values
st.subheader("Entered Input:")
st.write("Course Category: ", course_category)
st.write("Time Spent on Course (hours): ", time_spent)
st.write("Number of Videos Watched: ", videos_watched)
st.write("Number of Quizzes Taken: ", quizzes_taken)
st.write("Average Quiz Score: ",quiz_scores)
st.write("Course Completion Rate (%): ", completion_rate)
st.write("Device Type: ", device_type_cat)

# Display Prediction values
st.subheader("Prediction:")

# Prediction button
if st.button("Predict"):
  prediction = predict_course_completion(course_category, time_spent, videos_watched, quizzes_taken, quiz_scores, completion_rate, device_type)
  
  # Display the prediction
  if prediction == 1:
    st.success("The student is likely to complete the course.")
  else:
    st.error("The student is unlikely to complete the course.")

