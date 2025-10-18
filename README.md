# Real-Time Facial Emotion Recognition

This project uses a Convolutional Neural Network (CNN) trained on the FER2013 dataset to detect facial emotions in real-time using your webcam.

## Features

* Detects faces using OpenCV's Haar Cascade classifier.
* Classifies emotions into 7 categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
* Displays the prediction directly on the webcam feed.

## Files

* `real_time_emotion.py`: The main Python script to run the application.
* `emotion_model.h5`: The pre-trained Keras model file.
* `haarcascade_frontalface_default.xml`: OpenCV's pre-trained face detector.
* `requirements.txt`: List of required Python packages.
* `.gitignore`: Specifies files for Git to ignore.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    *(Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME`)*

2.  **Create and activate a virtual environment:**
    ```bash
    # Create (use python3 if needed)
    python -m venv venv
    # Activate
    # Windows (PowerShell): .\venv\Scripts\Activate
    # Mac/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Make sure your virtual environment is activated. Then run:

```bash
python real_time_emotion.py