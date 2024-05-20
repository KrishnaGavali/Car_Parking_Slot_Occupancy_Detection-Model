import cv2  # Import OpenCV library for video capture and processing
import keras  # Import Keras library for loading the trained model
import pickle  # Import Pickle library for loading ROI and scaler
import numpy as np  # Import NumPy library for numerical operations
import tensorflow as tf  # Import TensorFlow for GPU configuration and deep learning
import threading  # Import Threading for parallel execution of drawing rectangles
import time  # Import Time for FPS calculation
from config import videoPath  # Import videoPath from config file

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")  # List available GPUs
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu, True
        )  # Set memory growth for each GPU

# Load ROI list and scaler from pickle files
with open("ROI.pkl", "rb") as f:
    ROI_LIST = pickle.load(f)  # Load ROI list from ROI.pkl file

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # Load scaler object from scaler.pkl file

# Load the pre-trained model from the .h5 file
model = keras.models.load_model("model.h5")  # Load the pre-trained model

# Initialize the video capture object with the provided video path
cap = cv2.VideoCapture(videoPath)

# Frame skipping parameter
n = 5  # Skip prediction for every n frames
frame_counter = 0  # Initialize frame counter

# Lists to store parking space coordinates and confidence scores
vacant_spaces = []
parked_spaces = []
confidence_scores = []


def check_space(frame, roi):
    """
    Check the parking space by extracting the ROI, flattening it, scaling it,
    and reshaping it to the required input shape of the model.
    """
    roi_img = frame[roi[1] : roi[3], roi[0] : roi[2]]  # Extract ROI from frame
    roi_img = np.array(roi_img)  # Convert ROI to NumPy array
    flatten_roi_image = roi_img.flatten()  # Flatten the ROI image
    flatten_roi_scaled = scaler.transform(
        [flatten_roi_image]
    )  # Scale the flattened ROI
    # Reshape to match the model's input shape
    reshaped_roi = flatten_roi_scaled.reshape(-1, 15552)
    roi_images.append(reshaped_roi)  # Append reshaped ROI to the list


def draw_rectangles(frame, vacant_spaces, parked_spaces, confidence_scores):
    """
    Draw rectangles on the frame for visualization.
    """
    for i, roi in enumerate(vacant_spaces):
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Put confidence score in the center of the rectangle
        confidence = confidence_scores[i][0]
        cv2.putText(
            frame,
            f"{confidence:.2f}",
            ((x1 + x2) // 2, (y1 + y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    for i, roi in enumerate(parked_spaces):
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put confidence score in the center of the rectangle
        confidence = confidence_scores[i][1]
        cv2.putText(
            frame,
            f"{confidence:.2f}",
            ((x1 + x2) // 2, (y1 + y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


# Main loop to read frames and check spaces
while True:
    start_time = time.time()  # Start time for FPS calculation

    ret, frame = cap.read()  # Read a frame from the video capture object
    if not ret:  # If there are no more frames to read, reset the capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:  # If resetting fails, break the loop
            break

    frame_counter += 1  # Increment frame counter

    if frame_counter % n == 0:  # Skip prediction for every n frames
        roi_images = []  # Clear the list of ROI images for each frame
        vacant_spaces = []  # Clear the list of vacant spaces for each frame
        parked_spaces = []  # Clear the list of parked spaces for each frame
        confidence_scores = []  # Clear the list of confidence scores for each frame

        for roi in ROI_LIST:  # Loop through each ROI
            check_space(frame, roi)  # Check the parking space

        if roi_images:  # Ensure there are ROI images to predict
            # Predict occupancy of parking spaces using the model
            model_predict = model.predict(
                np.array(roi_images).reshape(len(roi_images), 15552)
            )

            for i, prediction in enumerate(
                model_predict
            ):  # Iterate through predictions
                argmax = np.argmax(prediction)  # Get index of maximum prediction
                confidence_scores.append(prediction)  # Append the confidence score

                if argmax == 0:  # If prediction indicates vacant space
                    vacant_spaces.append(
                        ROI_LIST[i]
                    )  # Append ROI to vacant spaces list
                elif argmax == 1:  # If prediction indicates parked space
                    parked_spaces.append(
                        ROI_LIST[i]
                    )  # Append ROI to parked spaces list

    # Create and start a thread for drawing rectangles
    drawing_thread = threading.Thread(
        target=draw_rectangles,
        args=(frame, vacant_spaces, parked_spaces, confidence_scores),
    )
    drawing_thread.start()
    drawing_thread.join()

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Display FPS on the frame
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    cv2.imshow("Frame", frame)  # Display the frame
    key = cv2.waitKey(1)  # Wait for key press (1 ms)
    if key == ord("q"):  # Break the loop if 'q' key is pressed
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
