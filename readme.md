# AI-Based Transit Fare Validator 🚍🤖

## Overview
The **AI-Based Transit Fare Validator** is a computer vision–based system that automatically verifies eligibility for **child and senior citizen concession fares** in public transport systems.  
Using a live camera feed, the system detects human faces, predicts age groups using a pretrained deep learning model, and applies rule-based logic to determine fare discounts.

This project helps reduce fare fraud, ensures fair access to concessions, and streamlines the fare validation process.

---

## Features
- Real-time face detection using OpenCV DNN  
- Age group prediction using a pretrained deep learning model  
- Automatic validation of:
  - Child concession fares  
  - Senior citizen concession fares  
- Live webcam-based demonstration  
- Clear visual feedback with bounding boxes and labels  

---

## Project Structure
```
AI-Transit-Fare-Validator/
│
├── model/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── age_deploy.prototxt
│   └── age_net.caffemodel
│
├── requirements.txt
├── transit-fare-validator.py
└── README.md
```

---

## Model Requirements

### Face Detection Model
- deploy.prototxt  
- res10_300x300_ssd_iter_140000.caffemodel  

### Age Detection Model
- age_deploy.prototxt  
- age_net.caffemodel  

All models are loaded using OpenCV's DNN module.

---

## Installation

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Verify Model Files
Ensure all model files are placed inside the `model/` directory.

---

## Usage
```
python transit-fare-validator.py
```

- Webcam opens automatically  
- Face and age group detected in real time  
- Fare eligibility displayed on screen  
- Press **Q** to exit  

---

## Fare Validation Logic

| Age Group | Fare Status |
|----------|-------------|
| 0–12 | Child Discount Allowed |
| 60+ | Senior Discount Allowed |
| Others | No Discount |

---

## Technologies Used
- Python  
- OpenCV (DNN Module)  
- NumPy  
- Pretrained Caffe Models  

---

## Applications
- Public buses  
- Metro stations  
- Smart ticketing systems  
- Automated fare validation kiosks  

---

## Limitations
- Performance depends on lighting and camera quality  
- Age prediction is approximate  
- Occlusions (mask, scarf) may reduce accuracy  

---

## Future Enhancements
- Integration with live ticketing systems  
- Mobile application support  
- Higher accuracy deep learning models (DeepFace / InsightFace)  
- Identity verification support  

---

