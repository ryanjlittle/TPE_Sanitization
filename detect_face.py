import sys
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


classifier = cv2.CascadeClassifier('/home/ryan/.local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')

img = cv2.imread(sys.argv[1])
dataset = sys.argv[2]
dest = sys.argv[3]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = classifier.detectMultiScale(gray, 1.3, 5)

for result in faces:

    x, y, w, h = result
    face = img[y:y+h, x:x+w]

    # This is intended to be used with the UTKFace dataset, where all images
    # are 200x200.
    resized_face = cv2.resize(face, (200, 200))

    max_sim = 0
    best_target = None
    for filename in os.listdir(dataset):
        filename = dataset + filename
        target = cv2.imread(filename)

        # Replace this line with the next to use PSNR instead of SSIM
        similarity = ssim(resized_face, target, multichannel=True)
        #similarity = psnr(resized_face, target)
        
        if similarity > max_sim:
            max_sim, best_target = similarity, target

    #cv2.imshow('source', resized_face)
    #cv2.imshow('target', best_target)
    #cv2.waitKey(0)

    resized_target = cv2.resize(best_target, (w, h))
    img[y:y+h, x:x+w] = resized_target

cv2.imshow('target', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save modified image
cv2.imwrite(dest, img)
