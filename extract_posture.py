import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from datasets import load_dataset, Video
from huggingface_hub import login

# Make sure to install dependencies:
# pip install huggingface_hub datasets opencv-python mediapipe pandas numpy

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe Pose Landmarker
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5)
pose = vision.PoseLandmarker.create_from_options(options)

class PoseLandmark:
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def process_video(video_path):
    """
    Process a video file and extract summary posture statistics.
    """
    cap = cv2.VideoCapture(video_path)
    
    # Store key metrics over frames
    shoulder_slopes_abs = []
    head_centering_scores = []
    hand_speeds = []
    hand_to_face_dists = []
    crossed_arms_dists = []
    
    # New metrics for leaning/slouching and shifts
    shoulder_widths = []
    nose_shoulder_dists = []
    core_positions = []
    core_speeds = []
    prev_left_wrist = None
    prev_right_wrist = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find poses
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = pose.detect(mp_image)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            # Get coordinates (normalized 0.0 to 1.0)
            nose = landmarks[PoseLandmark.NOSE]
            l_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            l_wrist = landmarks[PoseLandmark.LEFT_WRIST]
            r_wrist = landmarks[PoseLandmark.RIGHT_WRIST]
            
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
            mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            
            # Head Centering (Deviation of nose X from center of shoulders)
            head_centering_scores.append(abs(nose.x - mid_shoulder_x))
            
            # Calculate absolute shoulder slope (tilt)
            # dy / dx
            dx = r_shoulder.x - l_shoulder.x
            dy = r_shoulder.y - l_shoulder.y
            if dx != 0:
                slope = dy / dx
                shoulder_slopes_abs.append(abs(slope))
                
            # Leaning (shoulder width changes as you lean in/out)
            shoulder_widths.append(abs(r_shoulder.x - l_shoulder.x))
            
            # Slouching (vertical distance from nose to shoulder midpoint)
            nose_shoulder_dists.append(abs(nose.y - mid_shoulder_y))
            
            # Hand speed calculation
            current_l_wrist = (l_wrist.x, l_wrist.y)
            current_r_wrist = (r_wrist.x, r_wrist.y)
            if prev_left_wrist and prev_right_wrist:
                l_speed = np.sqrt((current_l_wrist[0] - prev_left_wrist[0])**2 + (current_l_wrist[1] - prev_left_wrist[1])**2)
                r_speed = np.sqrt((current_r_wrist[0] - prev_right_wrist[0])**2 + (current_r_wrist[1] - prev_right_wrist[1])**2)
                hand_speeds.append((l_speed + r_speed) / 2)
            prev_left_wrist = current_l_wrist
            prev_right_wrist = current_r_wrist
            
            # Hand to Face Distance (Face Touching)
            l_hand_face_dist = np.sqrt((l_wrist.x - nose.x)**2 + (l_wrist.y - nose.y)**2)
            r_hand_face_dist = np.sqrt((r_wrist.x - nose.x)**2 + (r_wrist.y - nose.y)**2)
            hand_to_face_dists.append(min(l_hand_face_dist, r_hand_face_dist))
            
            # Crossed Arms (Distance from wrists to opposite shoulders)
            l_wrist_r_shoulder_dist = np.sqrt((l_wrist.x - r_shoulder.x)**2 + (l_wrist.y - r_shoulder.y)**2)
            r_wrist_l_shoulder_dist = np.sqrt((r_wrist.x - l_shoulder.x)**2 + (r_wrist.y - l_shoulder.y)**2)
            crossed_arms_dists.append((l_wrist_r_shoulder_dist + r_wrist_l_shoulder_dist) / 2)
            
            # Posture Shifts (Core speed)
            current_core = (mid_shoulder_x, mid_shoulder_y)
            if core_positions:
                prev_core = core_positions[-1]
                # Euclidean distance between core position in current and previous frame
                speed = np.sqrt((current_core[0] - prev_core[0])**2 + (current_core[1] - prev_core[1])**2)
                core_speeds.append(speed)
            core_positions.append(current_core)
            
    cap.release()
    
    if len(core_positions) == 0:
        return None # No poses detected in the video
        
    # Calculate summary statistics
    metrics = {
        'head_centering_score_mean': np.mean(head_centering_scores) if head_centering_scores else 0,
        'absolute_shoulder_slope_mean': np.mean(shoulder_slopes_abs) if shoulder_slopes_abs else 0,
        'shoulder_slope_var': np.var(shoulder_slopes_abs) if shoulder_slopes_abs else 0,
        # Leaning and Slouching features
        'shoulder_width_mean': np.mean(shoulder_widths) if shoulder_widths else 0,
        'shoulder_width_var': np.var(shoulder_widths) if shoulder_widths else 0,
        'nose_shoulder_dist_mean': np.mean(nose_shoulder_dists) if nose_shoulder_dists else 0,
        'nose_shoulder_dist_var': np.var(nose_shoulder_dists) if nose_shoulder_dists else 0,
        # Hand & Arm Features
        'hand_speed_mean': np.mean(hand_speeds) if hand_speeds else 0,
        'hand_to_face_touches': sum(1 for d in hand_to_face_dists if d < 0.1) if hand_to_face_dists else 0,
        'crossed_arms_score': np.mean(crossed_arms_dists) if crossed_arms_dists else 0,
        # Posture Shift features
        'core_speed_mean': np.mean(core_speeds) if core_speeds else 0,
        'posture_shift_count': sum(1 for s in core_speeds if s > 0.02) if core_speeds else 0,
        # High level Behavioral scores
        'engagement_score': (np.mean(shoulder_widths) / (np.var(head_centering_scores) + 0.1)) if shoulder_widths and head_centering_scores else 0,
        'agitation_score': (sum(1 for s in core_speeds if s > 0.02) + sum(1 for d in hand_to_face_dists if d < 0.1)) if core_speeds and hand_to_face_dists else 0
    }
    
    return metrics


def main():
    # 1. Authenticate with HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
        print("Please set it using: export HF_TOKEN='your_hf_token'")
        return
        
    print("Logging into Hugging Face...")
    login(token=hf_token)
    
    # 2. Load the Dataset
    print("Loading dataset AI4A-lab/RecruitView...")
    try:
        # NOTE: Depending on dataset structure, this might download the videos 
        # or just provide paths if streaming is supported.
        # Assuming the videos are in a specific column, you may need to adjust 'video' column name
        dataset = load_dataset("AI4A-lab/RecruitView", split="train") 
        # CRITICAL: We must cast the 'video' column to not decode automatically.
        # Otherwise, it returns a VideoDecoder object instead of the file path.
        dataset = dataset.cast_column('video', Video(decode=False))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    all_features = []
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    print(f"Total videos to process: {len(dataset)}")
    
    # 3. Process each video
    # We will just process the first 5 for a dry run, adjust this as needed!
    for i, item in enumerate(dataset):
        # Adjust 'video_id' and 'video_path' based on actual dataset columns
        # Often HF datasets return video objects with 'path'
        
        # This is a placeholder column mapping, update it based on dataset structure!
        vid_id = item.get('id', f"video_{i}") 
        
        video_data = item.get('video', None)
        if video_data and isinstance(video_data, dict) and 'path' in video_data:
            video_path = video_data['path']
        elif isinstance(video_data, str):
            video_path = video_data
        else:
             print(f"Skipping index {i}: Could not find video path.")
             continue
            
        print(f"Processing ID: {vid_id} | Path: {video_path}")
        metrics = process_video(video_path)
        
        if metrics:
            metrics['id'] = vid_id
            all_features.append(metrics)
        else:
            print(f"Warning: No posture detected for ID: {vid_id}")
            
        # Remove the break statement to process the full dataset
        if i >= 4: 
            print("Stopping after 5 videos for dry-run testing. Remove the break statement in main() to process all.")
            break
            
    # 4. Save to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Reorder columns to put ID first
        cols = ['id'] + [c for c in df.columns if c != 'id']
        df = df[cols]
        
        output_csv = "data/posture_features.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nSuccess! Features saved to {output_csv}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
