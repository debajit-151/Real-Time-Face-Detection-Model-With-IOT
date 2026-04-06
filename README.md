# Smart Face Identity Base

A complete, real-time Face Recognition System built with Python, OpenCV, and DeepFace. It features a modern CustomTkinter graphical interface, accurate face embedding tracking (FaceNet), built-in Liveness/Anti-Spoofing detection, and **ESP32-based servo camera panning with PIR motion sensing**.

## Features

- **Real-Time Detection & Recognition**: Rapidly processes webcam streams using OpenCV Haar Cascades and matches faces using DeepFace's `Facenet` model.
- **Anti-Spoofing (Liveness Detection)**: Actively defeats spoofing attempts (e.g. photos held to the camera, phones) via DeepFace's Anti-Spoofing `MiniFASNet` module running on PyTorch.
- **ESP32 Servo Tracking + PIR Sensor**: The camera is mounted on a servo motor controlled by an ESP32. A PIR sensor detects motion to trigger scanning, and the servo smoothly tracks recognized faces.
- **Continuous Registry Workflow**: Easily enroll new faces directly from the webcam via a burst-capture system for higher profiling robustness.
- **Modern Dashboard UI**: A dark mode, clean GUI built with `customtkinter`.
- **Persistent Logging**: Automatically logs identity sightings with timestamps into `logs/recognition_logs.csv`.
- **Multi-Face Support**: Detects and processes multiple individuals in the same frame.
- **Graceful Hardware Fallback**: If the ESP32 is not connected, the system works perfectly in software-only mode.

## Prerequisites

> **⚠️ Python Version**: This software uses TensorFlow and PyTorch. Use **Python 3.10, 3.11, or 3.12**. Python 3.13+ may not have compatible wheels.

- Python 3.10–3.12 (virtual environment recommended)
- A connected USB Webcam
- (Optional) ESP32 DevKit + SG90/MG996R Servo + HC-SR501 PIR Sensor

## Quickstart Installation

### 1. Clone the Repository

```bash
git clone <repository_url>
cd Debo_Minor_Project
```

### 2. Create and Activate a Virtual Environment

```bash
# macOS / Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: On first run, DeepFace automatically downloads FaceNet and MiniFASNet model weights to `~/.deepface/weights/`.

## Usage

```bash
python main.py
```

### Register a New Person

1. Click **"Register New Person"** in the GUI.
2. Enter the person's name.
3. Look into the webcam. When the green box surrounds your face, press **`C`** to capture.
4. Move your head slightly and press **`C`** multiple times to add different angles (recommended: 5–10 captures).
5. Press **`Q`** when finished.

### Live Recognition

1. Click **"Launch Real-Time Recognition"**.
2. Recognized faces show a **Green Box** with their name.
3. Spoofing attempts (photos/screens) show a **Red Box** with **"SPOOF DETECTED!"**.
4. Logs are auto-saved to `logs/recognition_logs.csv`.
5. Press **`Q`** to exit.

## Hardware Setup (ESP32 + Servo + PIR)

### Components Required

| Component | Model | Purpose |
|---|---|---|
| Microcontroller | ESP32 DevKit (NodeMCU) | Serial communication + control |
| Servo Motor | SG90 or MG996R | Pans the camera left/right |
| PIR Sensor | HC-SR501 | Detects motion to trigger scanning |
| USB Cable | Micro-USB or Type-C | Connects ESP32 to computer |

### Wiring Diagram

```
ESP32 Pin    →    Component
──────────────────────────
GPIO 14      →    Servo Signal (orange/white wire)
GPIO 13      →    PIR Output Pin
3.3V / 5V    →    Servo VCC (red wire) + PIR VCC
GND          →    Servo GND (brown wire) + PIR GND
```

> **⚠️ Power Note**: If using a high-torque servo (MG996R), power it from an external 5V supply (not from the ESP32's pin) to avoid brownouts. Connect the grounds together.

### Flashing the ESP32

1. Install the [Arduino IDE](https://www.arduino.cc/en/software) (or PlatformIO).
2. Install the **ESP32 board package**: Arduino IDE → Board Manager → search "esp32" → Install.
3. Install the **ESP32Servo library**: Arduino IDE → Library Manager → search "ESP32Servo" → Install.
4. Open `esp32_firmware/servo_pir_controller.ino`.
5. Select your board: **Tools → Board → ESP32 Dev Module**.
6. Select the correct COM port: **Tools → Port**.
7. Click **Upload**.

### How It Works

```
┌─────────────┐
│  PIR Sensor  │──── Motion? ──→ ESP32 starts scanning
└─────────────┘                      │
                                     ▼
                              ┌─────────────┐
                              │  ESP32       │──── Sweeps servo left↔right
                              │  (Serial)    │     (smooth 1° steps)
                              └──────┬──────┘
                                     │ USB Serial
                                     ▼
                              ┌─────────────┐
                              │  Python      │──── Detects & recognizes face
                              │  (main.py)   │     in webcam frame
                              └──────┬──────┘
                                     │ TRACK:angle
                                     ▼
                              ┌─────────────┐
                              │  Servo       │──── Locks onto and follows
                              │  Motor       │     the recognized face
                              └─────────────┘
```

**State Machine:**
- **IDLE**: No motion detected. Servo stays still.
- **SCANNING**: PIR triggered. Servo sweeps 30°–150° smoothly.
- **TRACKING**: Python detected a known face. Servo locks on and follows.
- If the face goes out of frame for >1.5 seconds, the servo resumes scanning.
- If no motion is detected for 10 seconds, the servo stops.

### Serial Protocol

| Direction | Message | Meaning |
|---|---|---|
| ESP32 → Python | `MOTION:1` | Motion detected by PIR |
| ESP32 → Python | `MOTION:0` | No motion (timeout) |
| ESP32 → Python | `ANGLE:90` | Current servo angle |
| ESP32 → Python | `STATE:SCANNING` | Current ESP32 state |
| Python → ESP32 | `TRACK:95` | Move servo to 95° |
| Python → ESP32 | `SCAN` | Resume scanning sweep |
| Python → ESP32 | `STOP` | Stop servo (go idle) |

## Code Structure

| File | Purpose |
|---|---|
| `main.py` | GUI + recognition loop + servo integration |
| `face_detection.py` | OpenCV Haar cascade face detection |
| `face_encoding.py` | DeepFace FaceNet embeddings + pickle DB |
| `recognition.py` | Cosine similarity matching logic |
| `add_person.py` | Webcam capture + registration workflow |
| `servo_controller.py` | Python ↔ ESP32 serial bridge |
| `logger.py` | CSV timestamped logging |
| `esp32_firmware/servo_pir_controller.ino` | Arduino firmware for ESP32 |

## Troubleshooting

- **"No ESP32 detected"**: This is normal if the ESP32 isn't plugged in. The system works in software-only mode.
- **Servo jittering**: Use an external 5V power supply for the servo instead of the ESP32's pin.
- **Tracking goes wrong direction**: In `servo_controller.py`, uncomment the `angle_offset = -angle_offset` line in the `face_center_to_angle` method.
- **TensorFlow install fails**: Ensure you're using Python 3.10–3.12, not 3.13+.
- **"leaked semaphore" warning**: Harmless TensorFlow/macOS quirk. Safe to ignore.
