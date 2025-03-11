import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load pro spike data
with open('pro_movement_data.json', 'r') as f:
    pro_data = json.load(f)

st.title("AIMASTER: Spike Like a Pro!")
st.write("Upload your volleyball spike video to match a pro!")

uploaded_file = st.file_uploader("Choose your spike video...", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Analyzing your spike... Please wait!")
    
    # Process the user spike
    cap = cv2.VideoCapture("temp_video.mp4")
    user_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
            user_data.append(landmarks)
    
    cap.release()
    pose.close()
    
    # Compare spike movements (focus on arms and wrists)
    if user_data and pro_data:
        min_length = min(len(user_data), len(pro_data))
        total_diff = 0
        arm_diff = 0
        wrist_diff = 0  # Track wrist specifically
        for i in range(min_length):
            for j in range(len(pro_data[i])):
                pro = pro_data[i][j]
                user = user_data[i][j]
                diff = np.sqrt((pro['x'] - user['x'])**2 + (pro['y'] - user['y'])**2 + (pro['z'] - user['z'])**2)
                total_diff += diff
                if j in [11, 12, 13, 14]:  # Shoulders, elbows
                    arm_diff += diff
                if j in [15, 16]:  # Wrists
                    wrist_diff += diff
        
        avg_diff = total_diff / (min_length * len(pro_data[0]))
        arm_avg_diff = arm_diff / (min_length * 4)  # 4 arm joints
        wrist_avg_diff = wrist_diff / (min_length * 2)  # 2 wrist joints
        score = max(0, 100 - (avg_diff * 100))  # 0-100% similarity
        
        st.write(f"Your spike matches the pro by {score:.1f}%!")
        
        # Refined spike-specific feedback
        if score > 85:
            st.write("Pro-level spike! Your arm swing and wrist snap are on point.")
        elif score > 60:
            if arm_avg_diff > 0.1:
                st.write("Solid effort! Raise your arm higher for a stronger swing.")
            elif wrist_avg_diff > 0.1:
                st.write("Good try! Snap your wrist faster for a killer spike.")
            else:
                st.write("Nice! Adjust your body angle to match the pro’s power.")
        else:
            if arm_avg_diff > wrist_avg_diff:
                st.write("Keep practicing! Focus on a full arm swing to boost your spike.")
            else:
                st.write("Work on it! A quick wrist snap will sharpen your hit.")
    else:
        st.write("Oops! Couldn’t track your spike—use a clear, well-lit video.")
