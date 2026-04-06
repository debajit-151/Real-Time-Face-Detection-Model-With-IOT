# Smart Face Identity Base

A complete, real-time Face Recognition System built with Python, OpenCV, and DeepFace. It boasts a modern CustomTkinter graphical interface, accurate facial embedding tracking (FaceNet), and built-in Liveness/Anti-Spoofing detection to reject fake photos and screens.

## Features

- **Real-Time Detection & Recognition**: Rapidly processes webcam streams using OpenCV Haar Cascades and matches faces using DeepFace's `Facenet` model.
- **Anti-Spoofing (Liveness Detection)**: Actively defeats spoofing attempts (e.g. photos held to the camera, phones) by running DeepFace's Anti-Spoofing `MiniFASNet` module natively using PyTorch. Spoofs are highlighted in red instantly.
- **Continuous Registry Workflow**: Easily enroll new faces directly from the webcam application via an adjustable burst-capture system for higher profiling robustness.
- **Modern Dashboard UI**: A dark mode, clean GUI built entirely with `customtkinter`.
- **Persistent Logging**: Automatically drops identity logs and timestamp sightings into `logs/recognition_logs.csv`.
- **Multi-Face Support**: Smoothly detects and processes multiple individuals in the same frame synchronously.

## Prerequisites

> [!WARNING]
> This software utilizes TensorFlow and PyTorch for advanced deep learning features. It is HIGHLY recommended to use **Python 3.10, 3.11, or 3.12**. Pre-release versions like Python 3.13+ or 3.14 often lack compiled AI networking dependencies on pip.

- Python 3.10 - 3.12 (A virtual environment is incredibly recommended)
- A connected USB Webcam

## Quickstart Installation

1. **Clone the repository** (if deploying on a new machine)
   ```bash
   git clone <repository_url>
   cd Debo_Minor_Project
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # On macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   Ensure your virtual environment is active, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The first time the system runs, DeepFace will automatically download the required `Facenet` and `MiniFASNet` weights to your home directory `~/.deepface/weights/`)*

## Usage

Start the dashboard by executing the main script:

```bash
python main.py
```

### 1. Registering a New Person
- Click **"Register New Person"** from the Tkinter GUI.
- Enter the identity's name.
- Look directly into your webcam. When the green box surrounds your face, press **`C`** on your keyboard to capture an embedding.
- You can move your head slightly and press **`C`** multiple times to add different lighting/angles to your known dataset profile!
- Press **`Q`** when you are finished. The person can now be recognized.

### 2. Live Recognition
- Click **"Launch Real-Time Recognition"**.
- The webcam will open. Wait a few seconds for the neural networks to boot into your CPU context.
- Recognized faces will populate with a **Green Box** and Name tag.
- If someone holds up a phone or portrait to trick the system, it will flag them with a **Red Box** and a "**SPOOF DETECTED!**" tag.
- The history of recognized persons will automatically be written to `logs/recognition_logs.csv` every 10 seconds.
- Press **`Q`** to exit the tracker window.

## Code Structure

- **`main.py`**: The entrypoint and CustomTkinter Graphical User Interface holding exactly how frames are captured and rendered.
- **`face_detection.py`**: Contains the OpenCV detection logic isolating bounding ROI boxes around geometry mapping targets.
- **`recognition.py`**: Handles matching logic (Cosine Distances thresholds) against our recorded user list.
- **`face_encoding.py`**: Interacts with the backend DeepFace library to generate and query `embeddings.pkl`.
- **`add_person.py`**: Contains the webcam recording flow and dataset population logic for introducing new people.
- **`logger.py`**: A low-level CSV file writer maintaining event cooldown loops.
- **`dataset/`**: Directory where raw images of registered faces are saved.
- **`data/`**: Directory tracking the `.pkl` serialization array of the FaceNet 128D mathematical embeddings.
