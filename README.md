# Social Media Addiction Detection using AI

## Overview

This project is an AI-powered web application that analyzes smartphone usage from uploaded videos to predict the likelihood of social media addiction. It combines Computer Vision and Machine Learning to detect mobile phone usage, extract behavioral features, and classify addiction risk through an intuitive Flask web interface.

---

## Features

* Upload videos through a web interface
* Detect smartphones using YOLO object detection
* Extract video frames automatically
* Generate behavioral features from detections
* Predict social media addiction risk using a Machine Learning model
* Display prediction results through a Flask application
* Docker support for deployment

---

## AI Pipeline

```text
Upload Video
      │
      ▼
Frame Extraction
      │
      ▼
YOLO Phone Detection
      │
      ▼
Feature Extraction
      │
      ▼
Machine Learning Prediction
      │
      ▼
Result Display
```

---

## Tech Stack

### Programming Language

* Python

### Framework

* Flask

### Machine Learning

* Scikit-learn
* Joblib

### Computer Vision

* YOLO (Ultralytics)
* OpenCV

### Data Processing

* NumPy

### Frontend

* HTML
* CSS
* Jinja2

### Deployment

* Docker

---

## Project Structure

```text
social-media-addiction/
│
├── app.py
├── model.pkl
├── requirements.txt
├── Dockerfile
├── templates/
├── static/
├── dataset/
├── videos/
├── extracted_frames/
├── runs/
├── tools/
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/social-media-addiction.git
cd social-media-addiction
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Virtual Environment

**Windows**

```bash
.\.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## Running with Docker

Build the Docker image:

```bash
docker build -t socialmedia-addiction .
```

Run the container:

```bash
docker run -p 5000:5000 socialmedia-addiction
```

---

## Algorithms Used
* Random forest algorithm
* YOLO Object Detection
* Feature Extraction
* Scikit-learn Classification Model

---

## Future Enhancements

* Real-time webcam monitoring
* User authentication
* Analytics dashboard
* AWS cloud deployment
* Mobile application
* Advanced temporal behavior analysis using Deep Learning

---

## Author

**Monica M**

B.Tech in Artificial Intelligence and Machine Learning

Sri Shakthi Institute of Engineering and Technology
