"""
Use CV2 to detect certain amount of red in image and flag the image
"""

import cv2
import os
import shutil

completed_folders = ['1654124218.104456', '1654127995.838634', '1654380298.17326', '1654474401.603433', '1654478183.898067', '1654350148.025803', '1654353909.618664', '1654387856.880012', '1654391636.736659', '1654395403.311944', '1654481955.309156']
folder = completed_folders[-1]

for filename in os.listdir("./classifier/training/images/"+folder):
    file_path = os.path.join("./classifier/training/images/"+folder, filename)

    img = cv2.imread(file_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (175,50,20), (180,255,255))
    mask = cv2.bitwise_or(mask1, mask2)

    # Determine if the color exists on the image
    if (mask>0).mean()*100 >= 1:
        print((mask>0).mean()*100)

        croped = cv2.bitwise_and(img, img, mask=mask)
        # cv2.imshow("mask", mask)
        cv2.imshow("croped", img)
        
        c = cv2.waitKey(0) % 256

        if c == ord(' '):
            shutil.move(file_path, "./classifier/training/reviewed/"+filename)
        # elif c == ord('d'):
        #     shutil.move(file_path, "../classifier/data/down_street/"+filename)
        else:
            os.remove(file_path)

    else:
        os.remove(file_path)
        print("Bus not present")