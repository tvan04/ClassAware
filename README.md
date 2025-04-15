# ClassAware: Real-Time Engagement Monitoring in Virtual Classrooms

## 📌 Overview

ClassAware is a multimodal system for monitoring real-time student engagement in virtual classrooms. It combines face verification and hand gesture recognition to identify user participation, logs engagement events, simulates teacher-student interactions, and uses machine learning to classify engagement levels.

### ✨ Key Contributions
- Real-time participation logging using DeepFace and MediaPipe.
- Simulated student logs based on probabilistic behavioral models.
- Engagement classification using Random Forest and XGBoost.
- Modular structure supporting scalable student log generation and model evaluation.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/classaware.git
cd ClassAware
```

### 2. Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies
All Python dependencies are listed below. You can install them using:
```bash
pip install numpy pandas scikit-learn xgboost opencv-python mediapipe deepface

```
```
numpy
pandas
scikit-learn
xgboost
opencv-python
mediapipe
deepface
```

---

## 🔧 Environment Setup

- **Python Version**: 3.8+
- **Required Tools**:
  - Python with pip
  - OpenCV-compatible webcam
  - Internet connection for DeepFace model downloads

---

## 📂 Code Structure

```
├── ClassAware.py             # Real-time webcam logging and gesture recognition
├── data_generation.py       # Simulates teacher prompts and student behavior logs
├── ml_models.py             # Loads logs, extracts features, trains ML models
├── Student Logs/            # Output folder for student CSV logs (created when you run data_generation.py)
├── teacher_log.csv          # Simulated prompt times and question types (created when you run data_generation.py)
├── participation_log.csv    # Real-time gesture/face detection log (via webcam)
├── final_labeled_results.csv# ML classifier predictions and outputs (created when you run ml_models.py)
```

---

## 🧪 Datasets

### 🧾 Generated Data
- `teacher_log.csv`: Contains 50 simulated prompts spaced 3–8 minutes apart.
- `Student Logs/`: Each CSV contains timestamps and detected events for one student.
- `participation_log.csv`: Log of gesture/face match events via webcam.
- All data is generated locally via `data_generation.py` and `ClassAware.py`.

---

## ⚙️ Running the Code

### 1. Generate Simulated Logs
```bash
python data_generation.py
```

### 2. Run Classifier
```bash
python ml_models.py
```

### 3. Start Real-Time Logging via Webcam
Place a reference face image as `image.jpg` in the root directory.
```bash
python ClassAware.py
```

---

## 🧹 Preprocessing Steps

- `data_generation.py` simulates 1000 students with probabilistic delays, missed events, and noise.
- Behavior is controlled by probabilistic models depending on engagement level.
- `ml_models.py` extracts on-time, late, missed, and noise response counts per student.

---

## ⚙️ Configuration Details

You can customize simulation settings in `data_generation.py`:
- `NUM_STUDENTS` – How many student log files to generate (default: 1000)
- `NUM_TEACHER_EVENTS` – Number of prompts per session (default: 50)
- `LABEL_MAP` – Maps string engagement labels to class integers
- Response delays, noise count, and probabilities vary by engagement level

---

## 💡 Example Runs

#### Simulate and Train Models:
```bash
$ python data_generation.py
Generated teacher log: teacher_log.csv
[S1] => label=Medium, file=Student Logs/S1.csv
...

$ python ml_models.py
=== Random Forest Results ===
Accuracy: 0.8571
...
=== XGBoost Results ===
Accuracy: 0.8720
```

#### Run Webcam Tracker:
```bash
$ python ClassAware.py
[Webcam feed opens with live updates on gesture and match status]
```

---

## 📊 Output Interpretation

- `final_labeled_results.csv`: Predicted and actual engagement levels per student.
- Classification reports for both Random Forest and XGBoost.
- `participation_log.csv`: Timestamped logs for events like “Raised Hand detected” or “No match detected.”

---

## 📈 Example Use Cases

- Measure engagement trends across simulated or live classrooms.
- Benchmark real-time video detection and response logging.
- Train classifiers using behavioral features extracted from response timing.

---

## 🔍 Additional Notes

- All simulations and webcam detection are local-only and do not require cloud services.
- Face verification is based on the face_recognition library.
- Gesture detection is powered by MediaPipe's landmark-based inference.

---
