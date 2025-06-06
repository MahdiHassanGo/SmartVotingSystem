# Face Recognition Based Voting System

![image](https://github.com/user-attachments/assets/f7b2202e-2705-4d32-8e9c-af53b3db6cf9)
![image](https://github.com/user-attachments/assets/aada28b7-31dc-47ef-9a67-85f2cd1e5e43)


Elections cannot be democratic without voting serving as the foundational practice that ensures fairness and security of the electoral process. This system leverages digital voting technologies with anti-fraud mechanisms to ensure transparent and fair elections. The key components of this system include Arduino Uno, a keypad for voting input, and a 16x2 LED display for showing the results. Anti-fraud measures, including face recognition, prevent duplicate voting and ensure voter authenticity. The system is designed for small-scale elections, offering a cost-effective and reliable solution that can be expanded for larger systems.



## Introduction

Traditional voting systems have long struggled with issues like voter fraud and human error. With the advancement of digital technologies, we have seen a rise in electronic voting systems that enhance security and transparency. This project introduces a face recognition-based electronic voting system using Arduino Uno and Python (OpenCV), which eliminates the need for external software and hardware, simplifying the implementation process.

The system is designed to be offline and can operate in resource-constrained environments, ensuring that elections are secure and protected from cyber threats. The integration of face recognition enhances the accuracy and efficiency of the voting process while maintaining a low-cost and user-friendly interface.

## Features

- **Face Recognition for Voter Authentication**: Uses OpenCV and a webcam to authenticate voters through facial recognition, ensuring that each voter is legitimate.
- **Anti-Fraud Mechanism**: Prevents duplicate voting by recognizing the face of the voter.
- **Arduino Integration**: Uses Arduino Uno for keypad input and interfacing with the voting system.
- **Offline Functionality**: Operates without an internet connection, making it ideal for rural or low-infrastructure areas.
- **CSV Voting Record**: Votes are securely recorded in a CSV file for later analysis.

## Hardware Components

- **Arduino Uno R3**: Central controller capturing input from the keypad and interfacing with Python for face recognition.
- **4x4 Keypad**: Allows voters to enter their vote.
- **16x2 LCD Display**: Displays voting results in real time.
- **Webcam**: Captures voter images for face recognition.
- **1kΩ Resistors**: Protects sensitive components by regulating current.
- **Jumper Wires and Breadboard**: Connects components together for easy prototyping.

## Software Requirements

- **Python 3.x**
- **Libraries**: OpenCV, numpy, pickle, serial, scikit-learn, tqdm
- **Arduino IDE**: For uploading the code to Arduino Uno

## Installation

### 1. Install Required Python Libraries

Use pip to install the required Python libraries:

```bash
pip install opencv-python numpy scikit-learn pyserial tqdm
