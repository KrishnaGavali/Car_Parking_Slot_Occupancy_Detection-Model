import cv2  # Import OpenCV library for video capture and processing
import keras  # Import Keras library for loading the trained model
import pickle  # Import Pickle library for loading ROI and scaler
import numpy as np  # Import NumPy library for numerical operations
from config import videoPath  # Import videoPath from config file
import tensorflow as tf  # Import TensorFlow
import time  # Import time library to measure FPS

# TensorFlow GPU memory growth configuration
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Initialize lists to store vacant and parked spaces
vacant_spaces = []
parked_spaces = []
roi_images = []

# Load ROI (Region of Interest) data from pickle file
with open("ROI.pkl", "rb") as f:
    ROI_LIST = pickle.load(f)

# Load scaler object from pickle file
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load trained model from h5 file
model = keras.models.load_model("model.h5")


# Define a function to check if a parking space is vacant or occupied
def check_space(frame, roi):
    # Extract the region of interest (ROI) from the frame
    roi_img = frame[roi[1] : roi[3], roi[0] : roi[2]]
    # Convert the ROI to a NumPy array
    roi_img = np.array(roi_img)
    # Flatten the ROI image into a 1D array
    flatten_roi_image = roi_img.flatten()
    # Transform the flattened ROI using the loaded scaler
    flatten_roi_scaled = scaler.transform([flatten_roi_image])
    # Use the trained model to predict the occupancy of the parking space
    with tf.device("/device:GPU:0"):
        model_predict = model.predict(flatten_roi_scaled)

    # Determine if the parking space is vacant or occupied based on the model prediction
    argmax = np.argmax(model_predict)
    if argmax == 0:
        parked_spaces.append(roi)  # If the prediction is 0, the space is occupied
    elif argmax == 1:
        vacant_spaces.append(roi)  # If the prediction is 1, the space is vacant


# Open video capture object with the provided video path
cap = cv2.VideoCapture(videoPath)

# Loop through each frame in the video
while True:
    start_time = time.time()  # Start time for FPS calculation

    # Read a frame from the video capture object
    ret, frame = cap.read()

    # Break the loop if there are no more frames to read
    if not ret:
        break

    # Loop through each ROI and check its occupancy
    for roi in ROI_LIST:
        check_space(frame, roi)

    # Draw rectangles around vacant and occupied spaces on the frame
    for space in vacant_spaces:
        cv2.rectangle(
            frame, (space[0], space[1]), (space[2], space[3]), (0, 255, 0), 3
        )  # Green rectangle for vacant space

    for space in parked_spaces:
        cv2.rectangle(
            frame, (space[0], space[1]), (space[2], space[3]), (0, 0, 255), 2
        )  # Red rectangle for occupied space

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display FPS on the frame
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the frame with rectangles
    cv2.imshow("Frame", frame)

    # Clear the lists of vacant and occupied spaces for the next frame
    vacant_spaces = []
    parked_spaces = []

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
