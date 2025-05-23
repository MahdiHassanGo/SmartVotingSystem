from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
import serial
import tqdm

# Initialize serial communication with Arduino
try:
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    print("Please check the port and try again.")
    exit()

def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Initialize webcam
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the required directories exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Maximum number of people allowed
MAX_PEOPLE = 6

def get_current_people_count():
    if os.path.exists('data/names.pkl'):
        with open('data/names.pkl', 'rb') as f:
            existing_names = pickle.load(f)
            return len(set(existing_names))
    return 0

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

def register_new_face():
    current_count = get_current_people_count()
    if current_count >= MAX_PEOPLE:
        print(f"Maximum number of people ({MAX_PEOPLE}) already registered.")
        return False

    # Validate Aadhar number
    while True:
        name = input("Enter your 12-digit Aadhar number: ")
        if name.isdigit() and len(name) == 12:
            break
        else:
            print("Invalid Aadhar number. Please try again.")

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
                processed_face = process_face(frame, (x, y, w, h))
                
                if processed_face is not None:
                    aligned_face = cv2.resize(processed_face, (50, 50))

                    if len(faces_data) < frames_total and frame_counter % capture_interval == 0:
                        faces_data.append(aligned_face)
                        progress.update(1)

                    # Draw scanning box with animation
                    box_color = (0, 255, 0) if len(faces_data) % 2 == 0 else (0, 200, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    
                    # Display face number and capture progress
                    cv2.putText(frame, f"Face #{current_count + 1}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Scanning: {len(faces_data)}/{frames_total}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_counter += 1
            cv2.imshow('Face Registration', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= frames_total:
                break

    finally:
        progress.close()
        cv2.destroyAllWindows()

    if len(faces_data) == frames_total:
        # Convert collected faces to numpy array
        faces_data = np.asarray(faces_data, dtype=np.uint8)
        faces_data = faces_data.reshape((frames_total, -1))

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

            print(f"Face registration successful. Total registered people: {current_count + 1}/{MAX_PEOPLE}")
            return True

        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    return False

def initialize_voting_system():
    try:
        with open('data/names.pkl', 'rb') as f:
            LABELS = pickle.load(f)
        with open('data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)
    except FileNotFoundError:
        print("No registered faces found. Please register at least one face first.")
        return None, None

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    return knn, LABELS

# CSV file column names
COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

def check_if_exists(value):
    """Check if a voter already exists in the CSV file."""
    if not os.path.exists("Votes.csv"):
        return False
    with open("Votes.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0] == value:
                return True
    return False

def record_vote(name, vote):
    """Record a vote in the CSV file."""
    timestamp = time.time()
    date = datetime.fromtimestamp(timestamp).strftime("%d-%m-%Y")
    time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    record = [name, vote, date, time_str]

    file_exists = os.path.isfile("Votes.csv")
    with open("Votes.csv", "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(COL_NAMES)
        writer.writerow(record)

    speak("Thank you for participating in the elections.")

def main():
    while True:
        print("\n1. Register New Face")
        print("2. Start Voting")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            register_new_face()
        elif choice == '2':
            # Load and resize background image to 750x650
            try:
                imgBackground = cv2.imread("background.png")
                imgBackground = cv2.resize(imgBackground, (750, 650))
            except Exception as e:
                print(f"Error loading background image: {e}")
                continue

            knn, LABELS = initialize_voting_system()
            if knn is None:
                continue

            # Flag to track if the 7-second delay has passed
            face_capture_started = False
            last_face_time = 0
            face_detection_interval = 1.0  # Check for faces every 1 second

            while True:
                ret, frame = video.read()
                if not ret:
                    print("Error: Unable to read video frame.")
                    break

                # Resize the webcam frame to 350x250
                frame = cv2.resize(frame, (350, 250))

                # Embed the webcam feed into the background on the left side
                imgBackground[200:200 + 250, 50:50 + 350] = frame

                # Display the final window (750x650)
                cv2.imshow('Voting System', imgBackground)

                # Wait for 7 seconds before starting face capture
                if not face_capture_started:
                    time.sleep(7)
                    face_capture_started = True

                # Start face detection and voting process after the delay
                if face_capture_started:
                    current_time = time.time()
                    if current_time - last_face_time >= face_detection_interval:
                        # Convert frame to grayscale for face detection
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = facedetect.detectMultiScale(gray, 1.3, 5)

                        for (x, y, w, h) in faces:
                            processed_face = process_face(frame, (x, y, w, h))
                            
                            if processed_face is not None:
                                resized_img = cv2.resize(processed_face, (50, 50)).flatten().reshape(1, -1)
                                output = knn.predict(resized_img)[0]

                                # Draw scanning box with animation
                                box_color = (0, 255, 0) if int(time.time() * 2) % 2 == 0 else (0, 200, 0)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                                cv2.rectangle(frame, (x, y-40), (x+w, y), box_color, -1)
                                
                                # Display name and scanning status
                                cv2.putText(frame, f"Scanning: {output}", (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                                # Check if the voter has already voted
                                if check_if_exists(output):
                                    speak(f"{output}, you have already voted.")
                                else:
                                    # Prompt for voting
                                    speak(f"{output}, please cast your vote.")
                                    while True:
                                        # Read input from Arduino
                                        if arduino.in_waiting > 0:
                                            key = arduino.readline().decode('utf-8').strip()
                                            if key == '1':
                                                speak("Your vote for BAL has been recorded.")
                                                record_vote(output, "BAL")
                                                break
                                            elif key == '2':
                                                speak("Your vote for BNP has been recorded.")
                                                record_vote(output, "BNP")
                                                break
                                            elif key == '3':
                                                speak("Your vote for JAMAT has been recorded.")
                                                record_vote(output, "JAMAT")
                                                break
                                            elif key == '4':
                                                speak("Your vote has been canceled.")
                                                break
                                    break

                        last_face_time = current_time

                # Check if the window is closed
                if cv2.getWindowProperty('Voting System', cv2.WND_PROP_VISIBLE) < 1:
                    break

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

    # Release resources
    video.release()
    arduino.close()

if __name__ == "__main__":
    main()