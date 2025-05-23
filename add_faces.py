import cv2
import pickle
import numpy as np
import os
import tqdm

# Ensure 'data/' directory exists
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Maximum number of people allowed
MAX_PEOPLE = 6
current_people_count = 0

# Check existing people count
if os.path.exists('data/names.pkl'):
    with open('data/names.pkl', 'rb') as f:
        existing_names = pickle.load(f)
        current_people_count = len(set(existing_names))

if current_people_count >= MAX_PEOPLE:
    print(f"Maximum number of people ({MAX_PEOPLE}) already registered.")
    exit()

# Helper function for face alignment and quality check
def process_face(frame, face):
    (x, y, w, h) = face
    # Ensure minimum face size
    if w < 100 or h < 100:
        return None
    
    # Extract face ROI
    face_roi = frame[y:y+h, x:x+w]
    
    # Convert to grayscale for better processing
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    equalized_face = cv2.equalizeHist(gray_face)
    
    return equalized_face

# Validate NID NUMBER
while True:
    name = input("Enter your 12-digit NID number: ")
    if name.isdigit() and len(name) == 12:
        break
    else:
        print("Invalid NID number. Please try again.")

# Parameters for capturing frames
faces_data = []
frames_total = 51
capture_interval = 2
progress = tqdm.tqdm(total=frames_total, desc="Capturing faces", unit="frame")
frame_counter = 0

try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Unable to capture video frame.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Process face with quality check
            processed_face = process_face(frame, (x, y, w, h))
            
            if processed_face is not None:
                # Resize face to standard size
                aligned_face = cv2.resize(processed_face, (50, 50))

                # Capture only at intervals and append
                if len(faces_data) < frames_total and frame_counter % capture_interval == 0:
                    faces_data.append(aligned_face)
                    progress.update(1)

                # Draw scanning box with animation
                box_color = (0, 255, 0) if len(faces_data) % 2 == 0 else (0, 200, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                # Display face number and capture progress
                cv2.putText(frame, f"Face #{current_people_count + 1}", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Scanning: {len(faces_data)}/{frames_total}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_counter += 1
        cv2.imshow('Face Capture', frame)

        # Exit conditions
        if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= frames_total:
            break
finally:
    progress.close()
    video.release()
    cv2.destroyAllWindows()

# Convert collected faces to numpy array
faces_data = np.asarray(faces_data, dtype=np.uint8)
faces_data = faces_data.reshape((frames_total, -1))  # Reshape for storage

# Save Aadhar number and faces data
try:
    # Save names (Aadhar numbers)
    names_file = 'data/names.pkl'
    if not os.path.exists(names_file):
        names = [name] * frames_total
    else:
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
        names += [name] * frames_total
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

    # Save faces data
    faces_file = 'data/faces_data.pkl'
    if not os.path.exists(faces_file):
        with open(faces_file, 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
        faces = np.vstack((faces, faces_data))
        with open(faces_file, 'wb') as f:
            pickle.dump(faces, f)

    print(f"Data saved successfully. Total faces stored: {len(faces)}")
    print(f"Total registered people: {current_people_count + 1}/{MAX_PEOPLE}")

except Exception as e:
    print(f"Error saving data: {e}")
