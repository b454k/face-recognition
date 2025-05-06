# METU Artificial Intelligence Society – Face Recognition Project

The project focuses on building a face analysis system in three main stages using deep learning and computer vision tools like MediaPipe and OpenCV.

## Project Roadmap

- **Sprint 1 (Completed):** Face detection, alignment, and cropping  
- **Sprint 2 (Planned):** Face recognition and identity matching  
- **Sprint 3 (Planned):** Facial attribute detection (emotion, age, gender, etc.)

## Features (Sprint 1)

- Real-time face detection using **MediaPipe**  
- Alignment of rotated faces  
- Cropped and consistently scaled face output  
- **Interactive OpenCV trackbars** for adjusting face scale and vertical shift

## Requirements

- Python 3.6+  
- OpenCV  
- MediaPipe  
- NumPy

## Installation

```bash
git clone https://github.com/b454k/face-recognition.git
cd face-recognition
pip install opencv-python mediapipe numpy
python main.py
