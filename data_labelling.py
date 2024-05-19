# data collection, labelling and sorting

import cv2  # Import the OpenCV library for image and video processing
import pickle  # Import the pickle module for object serialization
from config import videoPath  # Import the video path from a configuration file

# Initialize counters for parked and vacant images
parked_image_counter = 0
vacant_image_counter = 0

# List of frame numbers to be processed
frame_list = [25, 75, 125, 175, 225, 275, 325, 375, 425, 475, 525, 575, 625, 675]

# Load the list of Regions of Interest (ROI) from a pickle file
with open("ROI.pkl", "rb") as f:
    ROI_LIST = pickle.load(f)
    print("Roi list : ", ROI_LIST)  # Print the loaded ROI list

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(videoPath)


def show_frames(cap, frame_list):
    """
    Function to display frames and classify ROI images as parked or vacant.
    """
    global parked_image_counter  # Use the global variable for parked images counter
    global vacant_image_counter  # Use the global variable for vacant images counter

    # Iterate through each frame number in the frame list
    for frame_num in frame_list:
        # Set the current frame position of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()  # Read the frame from the video
        if not ret:  # Check if the frame was read successfully
            print(f"Error: Unable to read frame {frame_num}")
            continue  # Skip to the next frame if reading failed

        # Iterate through each ROI in the ROI list
        for roi in ROI_LIST:
            # Extract the ROI image from the frame using the coordinates in ROI
            roi_image = frame[roi[1] : roi[3], roi[0] : roi[2]]
            cv2.imshow("roi_image", roi_image)  # Display the ROI image
            key = cv2.waitKey(0)  # Wait for a key press

            # Check if the '0' key (ASCII code 48) was pressed
            if key == 48:
                print("Parked")
                # Save the ROI image as a parked image with a unique filename
                cv2.imwrite(
                    f"Data\Raw\Parked\parkedImage{parked_image_counter}.png", roi_image
                )
                parked_image_counter += 1  # Increment the parked images counter
            # Check if the '.' key (ASCII code 46) was pressed
            elif key == 46:
                print("Vacant")
                # Save the ROI image as a vacant image with a unique filename
                cv2.imwrite(
                    f"Data\Raw\Vacant\\vacantImage{vacant_image_counter}.png", roi_image
                )
                vacant_image_counter += 1  # Increment the vacant images counter

            cv2.destroyAllWindows()  # Close the ROI window after each key press

    cap.release()  # Release the video capture object


# Call the function to display frames and process ROI images
show_frames(cap, frame_list)
