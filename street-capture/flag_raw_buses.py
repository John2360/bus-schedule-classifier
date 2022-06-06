"""
Use CV2 to detect certain amount of red in image and flag the image
"""

import cv2
import os
import shutil

# folder = "1654124218.104456"
# folder = "1654127995.838634"
# folder = "1654135568.511279"
# folder = "1654131783.18748"
folder = "1654191971.893699"
flagged_images = []

for filename in os.listdir("../classifier/data/"+folder):
    file_path = os.path.join("../classifier/data/"+folder, filename)

    img = cv2.imread(file_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2)

    # Determine if the color exists on the image
    if (mask>0).mean()*100 >= 1:
        print((mask>0).mean()*100)
        flagged_images.append(file_path)

        croped = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow("mask", mask)
        cv2.imshow("croped", img)
        
        c = cv2.waitKey(0) % 256

        if c == ord('u'):
            shutil.move(file_path, "../classifier/data/up_street/"+filename)
        elif c == ord('d'):
            shutil.move(file_path, "../classifier/data/down_street/"+filename)

    else:
        print("Bus not present")

print("Images collected: ")
print(flagged_images)