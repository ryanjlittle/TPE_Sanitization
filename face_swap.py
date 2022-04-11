import sys
import os
import argparse
import cv2
import imutils
import numpy as np
import time
import heapq
import random
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from index_dataset import ColorDescriptor
from image_search import Searcher


def face_search(face, dataset, method, index=None, n=20):
    if method.upper() == 'SSIM' or method.upper() == 'PSNR':
        return face_search_similarity_measure(face, dataset, n, method)
    elif method.upper() == 'HIST':
        return face_search_hist(face, dataset, index, n)
    else:
        raise Exception('Provided method is invalid')

def face_search_similarity_measure(face, dataset, n, measure):
    resized_face = cv2.resize(face, (100, 100))
    max_sim = 0
    best_target = None
    # heap to get the n closest
    heap = []

    for filename in os.listdir(dataset):
        filename = dataset + filename
        target = cv2.imread(filename)

        if measure.upper() == "SSIM":
            similarity = ssim(resized_face, target, multichannel=True)
        elif measure.upper() == "PSNR":
            similarity = psnr(resized_face, target)

        if len(heap) < n:
            heapq.heappush(heap, (similarity, target))
            continue
        
        if heap[0][0] < similarity:
            # replace smallest item in heap 
            heapq.heapreplace(heap, (similarity, target))
        
    i = random.randint(0, n-1)
    return heap[i][1]

def face_search_hist(face, dataset, index, n):
    cd = ColorDescriptor((8,12,3))
    features = cd.describe(face)
    searcher = Searcher(index)
    matches = searcher.search(features, limit=n)
    i = random.randint(0,n-1)
    return cv2.imread(dataset + '/' + matches[i][1])

def manually_approve_face(face) -> bool:
    cv2.imshow("face", face)
    cv2.waitKey(0)
    response = ''
    while response != 'y' and response != 'n':
        print("Is this a face? (y/n)")
        response = input()
    return response == 'y'

def manually_select_eyes(face, eyes):

    def highlight_and_show_eye(face, eye):
        x, y, w, h = eye
        face_copy = face.copy()
        cv2.rectangle(face_copy, (x,y) , (x+w, y+h), (0,255,0), 3)
        cv2.imshow("eye", face_copy)
        cv2.waitKey(0)

    for i in range(len(eyes)):
        highlight_and_show_eye(face, eyes[i])
        response = ''
        while response != 'y' and response != 'n':
            print("Is this the left eye? (y/n)")
            response = input()
        if response == 'y':
           left_eye_idx = i
           break

    for i in range(len(eyes)):
        if i == left_eye_idx:
            continue
        highlight_and_show_eye(face, eyes[i])
        response = ''
        while response != 'y' and response != 'n':
            print("Is this the right eye? (y/n)")
            response = input()
        if response == 'y':
           right_eye_idx = i
           break

    return eyes[left_eye_idx], eyes[right_eye_idx]

def get_face_angle(face, left_eye, right_eye):

    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))

    diff_x = right_eye_center[0] - left_eye_center[0]
    diff_y = right_eye_center[1] - left_eye_center[1]

    if diff_y == 0:
        return 0

    angle = np.degrees(np.arctan(diff_x / diff_y))
    if angle > 0:
        angle -= 90
    else:
        angle += 90
    
    cos_angle = -diff_y / np.sqrt(diff_x**2 + diff_y**2)
    sin_angle = diff_x / np.sqrt(diff_x**2 + diff_y**2)

    return angle, cos_angle, sin_angle


def get_rotated_corners(face, angle, cos_angle, sin_angle):
    x, y, w, h = face
    rotation_mat = np.matrix([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    corners = np.matrix([[w/2, w/2, -w/2, -w/2], [h/2, -h/2, -h/2, h/2]])
    rotated_corners = rotation_mat * corners
    face_center = (int(x + w/2), int(y + h/2))
    rotated_corners += np.matrix([[face_center[0]]*4, [face_center[1]]*4])
    return [(int(col[0,0]), int(col[0,1])) for col in np.transpose(rotated_corners)]


def get_bounding_box_coords(rotated_corners, dims):
    height, width = dims

    left = max(min([corner[0] for corner in rotated_corners]), 0)
    right = min(max([corner[0] for corner in rotated_corners]), width)
    bottom = max(min([corner[1] for corner in rotated_corners]), 0)
    top = min(max([corner[1] for corner in rotated_corners]), height)

    return bottom, top, left, right
    

def rotate_face(img, face, angle, cos_angle, sin_angle, rotated_corners):
    x, y, w, h = face
    b, t, l, r = get_bounding_box_coords(rotated_corners, img.shape[0:2])
    bounding_box = img[b:t, l:r]
    
    rotated = imutils.rotate_bound(bounding_box, angle)

    crop_horiz = np.floor(abs(w*cos_angle*sin_angle)).astype(int)
    crop_vert = np.floor(abs(h*cos_angle*sin_angle)).astype(int)
    return rotated[crop_vert:-crop_vert, crop_horiz:-crop_horiz]


def rotate_matched_face(img, match, angle, rotated_corners):
    b, t, l, r = get_bounding_box_coords(rotated_corners, img.shape[0:2])
    rotated_match = imutils.rotate_bound(match, -angle)
    return cv2.resize(rotated_match, (r-l, t-b))


def replace_face(img, match, rotated_corners):
    
    b, t, l, r = get_bounding_box_coords(rotated_corners, img.shape[0:2])

    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv2.fillPoly(mask, [np.array(rotated_corners)], 255)

    mask_inv = cv2.bitwise_not(mask)
    masked = cv2.bitwise_and(img, img, mask=mask_inv)

    replacement = img
    replacement[b:t, l:r] = match
    replacement = cv2.bitwise_and(img, img, mask=mask)

    return cv2.add(masked, replacement)


def detect_and_replace_faces(args):
    face_classifier = cv2.CascadeClassifier('/home/ryan/.local/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier('/home/ryan/.local/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml')

    img = cv2.imread(args.img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:

        start = time.time()
        face = img[y:y+h, x:x+w]

        gray_face = gray[y:y+h, x:x+w]
        
        if args.auto==0 and not manually_approve_face(face):
            continue

        eyes = eye_classifier.detectMultiScale(gray_face, scaleFactor=1.3, minNeighbors=5)

        # If we detect less than 2 eyes, perform the face swap without rotating.
        if len(eyes) < 2:
            match = face_search(face, args.dataset, args.method, args.index, n=20)
            match = cv2.resize(match, (h,w))
            img[y:y+h, x:x+w] = match
            continue

        # Sort detected eyes from leftmost to rightmost based on center of eye
        eyes_l_to_r = sorted(eyes, key=lambda e: e[0] + (e[2] / 2))
        
        if args.auto==0 and len(eyes) > 2:
            left_eye, right_eye = manually_select_eyes(face, eyes_l_to_r)
        else:
            left_eye, right_eye = eyes_l_to_r[0], eyes_l_to_r[-1]
            
        # Compute angle of face based on eye position
        angle, cos_angle, sin_angle = get_face_angle(face, left_eye, right_eye)

        # Rotate face so it's straightened. This aligns the face with the dataset images, which are all straightened
        rotated_corners = get_rotated_corners((x,y,w,h), angle, cos_angle, sin_angle)
        rotated_face = rotate_face(img, (x,y,w,h), angle, cos_angle, sin_angle, rotated_corners)
        
        # Find similar looking face from the dataset
        match = face_search(rotated_face, args.dataset, args.method, args.index, n=20)

        # Rotate matched face to the proper angle and overlay it on top of the original face in the image
        rotated_match = rotate_matched_face(img, match, angle, rotated_corners)
        img = replace_face(img, rotated_match, rotated_corners)
        
        stop = time.time()
        if args.auto==1:
            print("Face detected and replaced. Time: ", stop-start)

    return img
    
if __name__=="__main__":

    parser = argparse.ArgumentParser('face_swap.py', description="Detect and replace faces in an image with similar looking faces from a dataset")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-i', '--img', type=str, help='Path to image', required=True)
    required.add_argument('-d', '--dataset', type=str, help='Path to dataset', required=True)
    required.add_argument('-o', '--output', help='Path to store sanitized image', required=True)
    optional.add_argument('-m', '--method', default='PSNR', help='Either PSNR, SSIM, or HIST (default PSNR)')
    optional.add_argument('-x', '--index', help='Path to indexed dataset csv, if using histogram similarity measure')
    optional.add_argument('-a', '--auto', type=int, default=1, help='Set to 0 to manually approve detected faces')


    args = parser.parse_args()

    img = detect_and_replace_faces(args)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    cv2.imwrite(args.output, img)
