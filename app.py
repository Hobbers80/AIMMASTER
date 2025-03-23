import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import json
import os

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define pro hits (YouTube-based)
pro_hits = {
    "Karch Kiraly": {"file": "karch_spike.json", "url": "https://www.youtube.com/watch?v=example_karch"},
    "Taylor Sander": {"file": "taylor_spike.json", "url": "https://www.youtube.com/watch?v=example_taylor"},
    "How-To Guide": {"file": "howto_spike.json", "url": "https://www.youtube.com/watch?v=example_howto"}
}

# App branding
st.title("SpikeMaster: Train Like a Pro")
st.write("Upload your volleyball spike video to compare with top pros and get personalized drills!")
st.write("**Tip:** Face the camera with good lighting for best results.")

# File uploader
uploaded_file = st.file_uploader("Choose your spike video...", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Analyzing your spike against all pros... Please wait!")
    
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
            landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark]
            user_data.append(landmarks)
    
    cap.release()
    pose.close()
    
    # Compare to all pros
    if user_data:
        st.success("Analysis complete! Here’s your pro-level breakdown with drills!")
        st.header("Your SpikeMaster Report")
        
        for pro_name, pro_info in pro_hits.items():
            # Load pro data
            try:
                with open(pro_info["file"], 'r') as f:
                    pro_data = json.load(f)
            except FileNotFoundError:
                st.warning(f"Missing data for {pro_name}—upload {pro_info['file']} to your repo!")
                continue
            
            # Compare movements
            min_length = min(len(user_data), len(pro_data))
            total_diff = 0
            wrist_diff = 0
            elbow_diff = 0
            knee_diff = 0
            jump_height_diff = 0
            timing_diff = 0
            
            # Jump height and timing
            user_jump_peak = min([frame[23]['y'] for frame in user_data])
            pro_jump_peak = min([frame[23]['y'] for frame in pro_data])
            jump_height_diff = abs(user_jump_peak - pro_jump_peak)
            
            user_wrist_snap_frame = max(range(min_length), key=lambda i: abs(user_data[i][15]['y'] - user_data[i][13]['y']))
            pro_wrist_snap_frame = max(range(min_length), key=lambda i: abs(pro_data[i][15]['y'] - pro_data[i][13]['y']))
            timing_diff = abs(user_wrist_snap_frame - pro_wrist_snap_frame) / min_length
            
            # Joint differences
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
            
            # Calculate scores
            avg_diff = total_diff / (min_length * len(pro_data[0]))
            wrist_avg_diff = wrist_diff / (min_length * 2) if wrist_diff else 0
            elbow_avg_diff = elbow_diff / (min_length * 2) if elbow_diff else 0
            knee_avg_diff = knee_diff / (min_length * 2) if knee_diff else 0
            score = max(0, 100 - (avg_diff * 100))
            
            # Display comparison
            st.subheader(f"Vs. {pro_name} (Score: {score:.1f}%)")
            st.write(f"Watch {pro_name} here: [{pro_info['url']}]({pro_info['url']})")
            
            # Feedback and drills
            if wrist_avg_diff > 0.08:
                st.write("- **Wrist:** Lags behind {pro_name}—needs sharper snap!")
                st.write("  - **Drill:** *Wrist Snap Practice* - Toss a ball up, snap wrist to hit it down, 20 reps.")
            else:
                st.write("- **Wrist:** Matches {pro_name}’s snap—crisp and pro-like!")
                st.write("  - **Drill:** *Wrist Strengthener* - Squeeze a stress ball, 15x per hand.")
            
            if elbow_avg_diff > 0.08:
                st.write("- **Elbows:** Too bent vs. {pro_name}—extend more!")
                st.write("  - **Drill:** *Arm Extension Drill* - Swing at an imaginary ball, full extension, 25 reps.")
            else:
                st.write("- **Elbows:** Perfect extension like {pro_name}—great swing!")
                st.write("  - **Drill:** *Arm Power Boost* - Light dumbbell swings, 3 sets of 10.")
            
            if knee_avg_diff > 0.08:
                st.write("- **Knees:** Less bend than {pro_name}—dig deeper!")
                st.write("  - **Drill:** *Knee Bend Jumps* - Squat low then jump, 3 sets of 12.")
            else:
                st.write("- **Knees:** Strong bend matches {pro_name}—powerful base!")
                st.write("  - **Drill:** *Leg Endurance* - Wall sits, 3x 30 seconds.")
            
            if jump_height_diff > 0.1:
                st.write("- **Jumping:** Below {pro_name}’s height—get more lift!")
                st.write("  - **Drill:** *Vertical Jump Drill* - Jump to touch a high mark, 15 reps.")
            else:
                st.write("- **Jumping:** Matches {pro_name}’s jump—awesome air!")
                st.write("  - **Drill:** *Jump Maintenance* - Calf raises, 3 sets of 20.")
            
            if timing_diff > 0.1:
                st.write("- **Timing:** Off from {pro_name}—sync swing with peak!")
                st.write("  - **Drill:** *Timing Sync Drill* - Partner tosses, hit at jump peak, 20 reps.")
            else:
                st.write("- **Timing:** Perfect sync with {pro_name}—spot on!")
                st.write("  - **Drill:** *Timing Sharpener* - Jump and clap at peak, 15 reps.")
            
            st.write("---")
    else:
        st.error("Couldn’t track your spike—try a clear, well-lit video.")
else:
    st.info("Upload a video to start your SpikeMaster analysis!")
