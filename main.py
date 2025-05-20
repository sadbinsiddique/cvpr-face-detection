import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2

# Load Haar cascade
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# IP Webcam URL
url = "http://192.168.0.5:8080/video"
cap = cv2.VideoCapture(url)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

# Load model
model = load_model('best_model.h5')

# Fixed input shape expected by model
input_shape = (237, 237)

# Label map
def get_className(classNo):
    if classNo == 0:
        return "jon"
    elif classNo == 1:
        return "siam"
    elif classNo == 2:
        return "tony"
    else:
        return "Unknown"

# Validate resources
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
if facedetect.empty():
    print("Error: Could not load Haar cascade.")
    exit()
if model is None:
    print("Model not loaded. Please check the model path.")
    exit()

while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        continue

    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOriginal[y:y+h, x:x+w]

        try:
            
            img = cv2.resize(crop_img, input_shape)
        except Exception as e:
            print(f"Resize error: {e}")
            continue

        # Normalize and reshape
        img = img / 255.0
        img = img.reshape(1, 237, 237, 3)

        # Predict
        prediction = model.predict(img)
        classIndex = np.argmax(prediction, axis=1)[0]
        probabilityValue = np.amax(prediction)

        # Draw results
        label = f"{get_className(classIndex)}: {round(probabilityValue * 100, 2)}%"
        cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -1)
        cv2.putText(imgOriginal, label, (x + 5, y - 10), font, 0.6, (255, 255, 255), 1)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
