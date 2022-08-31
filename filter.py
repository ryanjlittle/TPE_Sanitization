import cv2
import numpy as np
import argparse

def filter_gaussian_blur(img):
    return cv2.GaussianBlur(img, (105,105), 0)

def filter_median_blur(img):
    return cv2.medianBlur(img, 35)

def filter_emboss(img):
    kernel = np.array([
      [-4, -2, 0],
      [-2, 1, 2],
      [0, 2, 4]
    ])
    return cv2.filter2D(img, -1, kernel)

def filter_edge_detect(img):
    kernel = np.array([
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1]
    ])
    return cv2.filter2D(img, -1, kernel)


def sanitize(args):
    img = cv2.imread(args.img)
    filtered = filter_emboss(img) # Replace this with any of the filter functions!

    cv2.imshow("filtered", filtered)
    cv2.waitKey(0)

    cv2.imwrite(args.output, filtered)

if __name__=="__main__":
    parser = argparse.ArgumentParser('filter.py', description="Filter a face")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-i', '--img', type=str, help='Path to image', required=True)
    required.add_argument('-o', '--output', help='Path to store sanitized image', required=True)

    args = parser.parse_args()
    sanitize(args)

