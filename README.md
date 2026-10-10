# 🤟 ASL Sign Language Recognition

A machine learning mini-project that recognizes American Sign Language (ASL) alphabet letters from A to Z using a webcam, hand landmark detection, and image-based computer vision techniques.

The system uses MediaPipe to detect hand landmarks and a trained machine learning classifier to predict the corresponding alphabet letter in real time.

## 🎯 Project Objective

The objective of this project is to develop a real-time ASL alphabet recognition system that identifies static hand signs using computer vision and machine learning.

The project explores how hand landmark features can be extracted and used to classify different hand signs.

## ✨ Features

- Real-time hand detection through a webcam
- Detection of 21 hand landmarks using MediaPipe
- Recognition of ASL alphabet letters A–Z
- Machine learning-based letter classification
- Prediction smoothing to reduce flickering between predictions
- Live display of the predicted letter and model score

## 🛠️ Technologies Used

- **Python** — core programming language
- **MediaPipe** — hand landmark detection
- **OpenCV** — webcam input and real-time visualization
- **NumPy** — numerical operations and landmark normalization
- **scikit-learn** — machine learning classifier
- **Joblib** — saving and loading the trained model

## 🧠 How It Works

The system follows these steps:

1. **Capture:** The webcam captures live video frames.
2. **Hand detection:** MediaPipe detects the hand in each frame.
3. **Landmark extraction:** The system identifies 21 points representing finger joints, fingertips, and the wrist.
4. **Feature preparation:** The landmark coordinates are normalized and converted into 63 numerical features.
5. **Classification:** A trained multilayer perceptron (MLP) classifier predicts the corresponding ASL letter.
6. **Prediction smoothing:** Recent predictions are combined to make the displayed letter more stable.
7. **Display:** The predicted letter and model score appear on the webcam feed.

## 📊 Dataset and Model Performance

The project uses the ASL Alphabet image dataset for training and evaluation.

For the landmark-based model, hand landmarks were extracted from the dataset images. The resulting landmark dataset was used to train an MLP classifier.

**Validation accuracy: 98.60%**

This result was measured on a held-out validation set of extracted hand landmarks. It should not be interpreted as guaranteed accuracy during real-world webcam use.

Actual performance may vary depending on lighting, hand position, camera quality, and similarities between hand signs.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 recommended
- A working webcam
- Git installed on your computer

### 1. Clone the repository

```bash
git clone https://github.com/Ansh443-21/asl-sign-language-recognition.git
cd asl-sign-language-recognition
```

### 2. Create a virtual environment

On Windows, run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Prepare the model files

The webcam application requires the following files inside the `models/` directory:

- `hand_landmarker.task`
- `landmark_model.joblib`
- `label_encoder.joblib`

These files are excluded from the GitHub repository. Obtain them from the project team and place them in the `models/` directory before running the application.

### 5. Run the application

```powershell
python landmark_app.py
```

Show one hand sign at a time in front of the webcam. The application will display the predicted letter and model score.

Press **Q** to close the application.

## 📁 Project Structure

```text
asl-sign-language-recognition/
├── app.py
├── extract_landmarks.py
├── landmark_app.py
├── predict.py
├── train.py
├── requirements.txt
├── .gitignore
├── README.md
└── models/
    └── Model files are obtained separately
```

### Main Files

| File | Purpose |
|---|---|
| `landmark_app.py` | Runs real-time ASL recognition using webcam input and hand landmarks |
| `extract_landmarks.py` | Extracts hand landmark features from dataset images |
| `train.py` | Contains the image-classification training workflow |
| `predict.py` | Supports predictions using the image-classification model |
| `app.py` | Additional application entry point |

## 👥 Project Team

This project was developed collaboratively by a team of three members as part of a university mini-project.

| Team Member | Contributions |
|---|---|
| **Ansh Vaghela** | Developed the machine learning model, implemented real-time ASL letter prediction, and integrated webcam-based recognition. |
| **Rishika Giri** | Worked on understanding and preparing the dataset, researching hand landmark detection, and exploring how hand features are used for letter recognition. |
| **Dhruv Luhar** | Assisted with testing the recognition system, reviewing prediction results, analyzing model performance, and preparing project documentation. |

All team members participated in project discussions, reviewed results, and contributed to the overall development of the ASL Sign Language Recognition system.

## ⚠️ Limitations

- The current application focuses on static alphabet signs rather than complete sign-language sentences.
- Similar hand shapes can lead to incorrect predictions.
- Performance depends on lighting, hand visibility, and camera quality.
- Dataset validation accuracy may differ from real-world performance.
- This educational prototype is not a replacement for a qualified sign-language interpreter.

## 🔮 Future Improvements

- Improve recognition of visually similar letters.
- Evaluate performance with different users and environments.
- Add a graphical user interface.
- Explore dynamic sign recognition and word formation.
- Provide a convenient way to download the trained model files.

## 🎓 Project Context

Developed as a university mini-project to explore computer vision, hand landmark detection, and machine learning for ASL alphabet recognition.

## 📄 License

No license has been added yet. Please contact the project team regarding reuse or redistribution.

---

**GitHub Repository:** [ASL Sign Language Recognition](https://github.com/Ansh443-21/asl-sign-language-recognition)
