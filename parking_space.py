# Import the image path from the configuration file
from config import imgPath

# Import OpenCV library
import cv2

# Import Pickle library for Saving and Loading Regions of Interest data
import pickle

# Define constants for the width and height of the rectangle
WIDTH, HEIGHT = 108, 48


# Intialize an empty list to store the regions of interest
ROI_LIST = []


# Try to load ROI list from a pickle file{}
try:
    with open("ROI.pkl", "rb") as f:
        ROI_LIST = pickle.load(f)
        print("ROI list loaded from ROI.pkl")
        print("ROI list loaded : ", ROI_LIST)
except FileNotFoundError:
    print("No existing ROI.pkl file found. Starting with an empty ROI list.")


# Define a function to handle mouse click events
def get_mouse_click(event, x, y, flags, params):
    global img, latest_img  # Use global variables to modify the images inside the function
    if event == cv2.EVENT_LBUTTONDOWN:
        # When the left mouse button is clicked, create a new ROI and add it to the list
        temp_roi = [x, y, x + WIDTH, y + HEIGHT]
        ROI_LIST.append(temp_roi)
        # Create a copy of the original image to draw the rectangles
        latest_img = img.copy()
        # Draw all rectangles from the ROI list on the image
        for roi in ROI_LIST:
            cv2.rectangle(
                latest_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2
            )
        # Display the updated image with the rectangles
        cv2.imshow("image", latest_img)

    if event == cv2.EVENT_RBUTTONDOWN:
        # When the right mouse button is clicked, remove the last ROI from the list
        roi_len = len(ROI_LIST)
        if roi_len > 0:
            del ROI_LIST[roi_len - 1]
            # Create a copy of the original image to draw the remaining rectangles
            latest_img = img.copy()
            # Draw all remaining rectangles from the ROI list on the image
            for roi in ROI_LIST:
                cv2.rectangle(
                    latest_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2
                )
            # Display the updated image with the rectangles
            cv2.imshow("image", latest_img)


# Load the image from the specified path
img = cv2.imread(imgPath)
# Create a copy of the original image to work with
latest_img = img.copy()
for roi in ROI_LIST:
    cv2.rectangle(latest_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

while True:
    # Display the latest image with the rectangles
    cv2.imshow("image", latest_img)

    # Set the mouse callback function to handle click events
    cv2.setMouseCallback("image", get_mouse_click)

    # Wait for a key press and break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break


# Save ROI list to a pickle file
with open("ROI.pkl", "wb") as f:
    pickle.dump(ROI_LIST, f)
    print("ROI list saved to ROI.pkl")
    print("ROI list saved : ", ROI_LIST)
    # Points are Stored as [X1 , Y1 , X2 , Y2] : [X1 , Y1] Top Left Corner , [X2 , Y2] Bottom Right Corner.
    # [X1,Y1] : [x, y] , [X2,Y2] : [x+WIDTH, y+HEIGHT]


# Close all OpenCV windows
cv2.destroyAllWindows()
