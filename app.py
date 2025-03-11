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
st.write("Tip: Face the camera and use good lighting for best tracking!")

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
    
    # Compare spike movements (detailed analysis)
    if user_data and pro_data:
        min_length = min(len(user_data), len(pro_data))
        total_diff = 0
        wrist_diff = 0
        elbow_diff = 0
        knee_diff = 0
        jump_height_diff = 0
        timing_diff = 0
        
        # Track jump height (hip movement) and timing
        user_jump_peak = min([frame[23]['y'] for frame in user_data])  # Hip (left) lowest point
        pro_jump_peak = min([frame[23]['y'] for frame in pro_data])
        jump_height_diff = abs(user_jump_peak - pro_jump_peak)
        
        # Timing: Wrist snap relative to jump peak
        user_wrist_snap_frame = max(range(min_length), key=lambda i: abs(user_data[i][15]['y'] - user_data[i][13]['y']))
        pro_wrist_snap_frame = max(range(min_length), key=lambda i: abs(pro_data[i][15]['y'] - pro_data[i][13]['y']))
        timing_diff = abs(user_wrist_snap_frame - pro_wrist_snap_frame) / min_length
        
        # Compare joint differences
        for i in range(min_length):
            for j in range(len(pro_data[i])):
                pro = pro_data[i][j]
                user = user_data[i][j]
                diff = np.sqrt((pro['x'] - user['x'])**2 + (pro['y'] - user['y'])**2 + (pro['z'] - user['z'])**2)
                total_diff += diff
                if j in [15, 16]:  # Wrists
                    wrist_diff += diff
                if j in [13, 14]:  # Elbows
                    elbow_diff += diff
                if j in [25, 26]:  # Knees
                    knee_diff += diff
        
        # Calculate average differences
        avg_diff = total_diff / (min_length * len(pro_data[0]))
        wrist_avg_diff = wrist_diff / (min_length * 2) if wrist_diff else 0
        elbow_avg_diff = elbow_diff / (min_length * 2) if elbow_diff else 0
        knee_avg_diff = knee_diff / (min_length * 2) if knee_diff else 0
        
        score = max(0, 100 - (avg_diff * 100))  # Overall similarity
        
        st.write(f"Your spike matches the pro by {score:.1f}%!")
        st.success("Great job uploading—here’s your full spike breakdown!")
        
        # Always give feedback on all factors
        st.header("Your Spike Analysis")
        
        # Wrist feedback
        if wrist_avg_diff > 0.08:
            st.write("- **Wrist:** Snap it faster for a sharper hit—your wrist action needs a boost!")
        else:
            st.write("- **Wrist:** Solid wrist snap—matches the pro’s sharpness!")
        
        # Elbow feedback
        if elbow_avg_diff > 0.08:
            st.write("- **Elbows:** Extend your elbows more during the swing for better reach.")
        else:
            st.write("- **Elbows:** Great elbow extension—your arm’s in pro form!")
        
        # Knee feedback
        if knee_avg_diff > 0.08:
            st.write("- **Knees:** Bend your knees deeper for a stronger push off.")
        else:
            st.write("- **Knees:** Perfect knee bend—great power from your legs!")
        
        # Jump feedback
        if jump_height_diff > 0.1:
            st.write("- **Jumping:** Jump higher—use your legs more for extra lift!")
        else:
            st.write("- **Jumping:** Awesome jump height—matches the pro’s airtime!")
        
        # Timing feedback
        if timing_diff > 0.1:
            st.write("- **Timing:** Sync your swing better—hit at the peak of your jump.")
        else:
            st.write("- **Timing:** Spot-on timing—your swing and jump are in sync!")
        
    else:
        st.write("Oops! Couldn’t track your spike—use a clear, well-lit video.")
