"""
Take pictures every second from webcam and zip every hour
"""

from time import time
from datetime import datetime
import cv2
import os
import time as tm
import shutil

# Create a new VideoCapture object
cam = cv2.VideoCapture(0)

# Initialise variables to store current time difference as well as previous time call value
previous = time()
delta = 0

photos_in_group = 0

# Keep looping
while True:
    # Get the current time, increase delta and update the previous variable
    current = time()
    delta += current - previous
    previous = current

    # one more photo
    photos_in_group += 1

    # Check if 3 (or some other value) seconds passed
    if delta > 3:
        # Operations on image
        # Reset the time counter
        delta = 0

    # Show the image and keep streaming
    _, img = cam.read()
    cv2.imshow("Frame", img)
    cv2.imwrite("./classifier/training/images/FrontStreet_"+str(datetime.timestamp(datetime.now()))+".jpg", img)
    
    # check to wait or save files
    if photos_in_group >= 3600:
        shutil.make_archive("./classifier/training/zipped/"+str(datetime.timestamp(datetime.now())), 'zip', "./classifier/training/images")

        for filename in os.listdir("./classifier/training/images"):
            file_path = os.path.join("./classifier/training/images", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        photos_in_group = 0
    else:    
        cv2.waitKey(1)
        tm.sleep(1)