import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json

# Install libraries (this won't run in Streamlit Cloud, but helps in Colab)
# Remove this section when deploying
st.write("Installing dependencies...")
!pip install opencv-python mediapipe numpy

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to process video and extract movement data
def process_video(video_file):
    movement_data = []
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            movement_data.append(landmarks)
    cap.release()
    return movement_data

# Function to calculate distance
def calculate_distance(pro_frame, user_frame):
    total_distance = 0
    num_points = 0
    for pro_landmark, user_landmark in zip(pro_frame, user_frame):
        pro_x, pro_y = pro_landmark['x'], pro_landmark['y']
        user_x, user_y = user_landmark['x'], user_landmark['y']
        distance = np.sqrt((pro_x - user_x)**2 + (pro_y - user_y)**2)
        total_distance += distance
        num_points += 1
    return total_distance / num_points if num_points > 0 else float('inf')

# Function to get feedback
def get_feedback(pro_frame, user_frame):
    feedback = []
    left_elbow_pro, left_elbow_user = pro_frame[13], user_frame[13]
    right_elbow_pro, right_elbow_user = pro_frame[14], user_frame[14]
    left_knee_pro, left_knee_user = pro_frame[25], user_frame[25]
    right_knee_pro, right_knee_user = pro_frame[26], user_frame[26]
    
    elbow_threshold = 0.1
    if abs(left_elbow_pro['y'] - left_elbow_user['y']) > elbow_threshold:
        feedback.append("Adjust your left elbow height.")
    if abs(right_elbow_pro['y'] - right_elbow_user['y']) > elbow_threshold:
        feedback.append("Adjust your right elbow height.")
    
    knee_threshold = 0.1
    if abs(left_knee_pro['y'] - left_knee_user['y']) > knee_threshold:
        feedback.append("Bend your left knee more.")
    if abs(right_knee_pro['y'] - right_knee_user['y']) > knee_threshold:
        feedback.append("Bend your right knee more.")
    
    return feedback if feedback else ["Great job! Try to match the timing better."]

# Streamlit app interface
st.title("AIMASTER: Compare Your Moves to a Pro!")
st.write("Upload your video and see how you stack up against a pro athlete.")

# Upload user video
user_video = st.file_uploader("Upload Your Video", type=["mp4"])

# Load pro data (you'll need to upload this manually for now)
pro_data_path = "pro_movement_data.json"
try:
    with open(pro_data_path, 'r') as f:
        pro_data = json.load(f)
except FileNotFoundError:
    st.error("Please upload 'pro_movement_data.json' to the app first!")
    pro_data = None

if user_video and pro_data:
    # Save the uploaded video temporarily
    with open("temp_user_video.mp4", "wb") as f:
        f.write(user_video.read())
    
    # Process the user's video
    user_data = process_video("temp_user_video.mp4")
    
    # Compare movements
    min_length = min(len(pro_data), len(user_data))
    distances = []
    feedback_list = []
    
    for i in range(min_length):
        distance = calculate_distance(pro_data[i], user_data[i])
        distances.append(distance)
        if i == min_length // 2:
            feedback_list = get_feedback(pro_data[i], user_data[i])
    
    # Calculate similarity score
    avg_distance = np.mean(distances)
    max_distance = 1.0
    similarity_score = max(0, 100 * (1 - avg_distance / max_distance))
    
    # Display results
    st.write(f"**Similarity Score:** {similarity_score:.2f}%")
    st.write(f"You matched {similarity_score:.2f}% with the pro!")
    st.write("**Coaching Tips:**")
    for tip in feedback_list:
        st.write(f"- {tip}")

# Clean up
pose.close()
