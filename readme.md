# AI-Based Transit Fare Validator 🚍🤖

## Overview
The **AI-Based Transit Fare Validator** is a computer vision–based system that automatically verifies eligibility for **child and senior citizen concession fares** in public transport systems.  
Using a live camera feed, the system detects human faces, predicts age groups using a pretrained deep learning model, and applies rule-based logic to determine fare discounts.

This project helps reduce fare fraud, ensures fair access to concessions, and streamlines the fare validation process.

---

## Features
- Real-time face detection using OpenCV DNN
- Age group prediction using a pretrained age estimation model
- Automatic validation of:
  - Child concession fares
  - Senior citizen concession fares
- Live webcam-based demonstration
- Clear visual feedback with bounding boxes and labels

---

## Project Structure
AI-Transit-Fare-Validator/
│
├── model/
│ ├── deploy.prototxt
│ ├── res10_300x300_ssd_iter_140000.caffemodel
│ ├── age_deploy.prototxt
│ └── age_net.caffemodel
│
├── requirements.txt
├── fare_validator.py
└── README.md

yaml
Copy code

---

## Model Requirements
The project uses **pretrained deep learning models**:

### Face Detection Model
- `deploy.prototxt`
- `res10_300x300_ssd_iter_140000.caffemodel`

### Age Detection Model
- `age_deploy.prototxt`
- `age_net.caffemodel`

These models are loaded using **OpenCV’s DNN module**.

---

## Installation

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd AI-Transit-Fare-Validator
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Verify Model Files
Ensure all required model files are placed inside the model/ directory.

Usage
Run the application using:

bash
Copy code
python fare_validator.py
The webcam will start automatically

The system detects faces and predicts age groups

Fare eligibility is displayed on the screen

Press Q to exit the application

Fare Validation Logic
Age Group	Fare Status
0–12	Child Discount Allowed
60+	Senior Citizen Discount Allowed
Others	No Discount

Technologies Used
Python

OpenCV (DNN Module)

NumPy

Pretrained Deep Learning Models (Caffe)

Applications
Public buses

Metro and railway stations

Smart ticketing systems

Automated fare validation kiosks

Limitations
Accuracy depends on lighting conditions and camera quality

Age prediction is approximate and may vary slightly

Masks or occlusions may affect detection

Future Enhancements
Integration with ticketing systems

Mobile application support

Face recognition for identity verification

Improved accuracy using modern deep learning models (e.g., DeepFace)

Aadhaar / ID-based verification (optional)

