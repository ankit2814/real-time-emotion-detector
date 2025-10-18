import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Load the Models ---

# Load the pre-trained face detector model (Haar Cascade)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load your trained emotion recognition model
emotion_model = load_model('emotion_model.h5')

# Define the emotion labels (make sure this matches the order from training)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Start Webcam ---
cap = cv2.VideoCapture(0) # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Processing Loop ---
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (Haar Cascade works on grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,  # How much the image size is reduced at each scale
        minNeighbors=5,   # How many neighbors each candidate rectangle should have
        minSize=(30, 30)    # Minimum possible object size
    )

    # --- Process Each Detected Face ---
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) - the face
        roi_gray = gray_frame[y:y + h, x:x + w]

        # --- Preprocess the face for the emotion model ---
        # Resize to 48x48 pixels (the size the model expects)
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Ensure it has the correct shape (add batch and channel dimensions)
        # Model expects (batch_size, height, width, channels)
        cropped_img = np.expand_dims(np.expand_dims(roi_gray_resized, -1), 0) # Shape: (1, 48, 48, 1)

        # Normalize the pixel values (already done by Rescaling layer in the model, but good practice if model didn't have it)
        # cropped_img = cropped_img / 255.0  <-- We don't need this if the model has Rescaling

        # --- Make Prediction ---
        emotion_prediction = emotion_model.predict(cropped_img, verbose=0) # verbose=0 prevents printing prediction details

        # Find the index of the emotion with the highest probability
        max_index = int(np.argmax(emotion_prediction))
        predicted_emotion = emotion_labels[max_index]

        # --- Draw on the Original Frame ---
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle

        # Put the predicted emotion text above the rectangle
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # --- Exit Condition ---
    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release() # Release the webcam
cv2.destroyAllWindows() # Close all OpenCV windows

print("Webcam closed.")