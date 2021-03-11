# Facial Sanitization for TPE

### Perturbation and Reverse Perturbation
The perturbation and reverse perturbation functions are in perturb.m and reverse_perturbation.m respectively. To perform the encryption and decryption, both functions call openssl from the command line. Thus, openssl must be installed and added to your path. 

The length of the padded metadata is hardcoded in the perturb function, and is currently set to 10% of the image size. This can be changed on line 93 of perturb.m (reverse_perturbation.m shouldn't need to be modified). 


Usage (in MATLAB):

```
target = imread(<target image location>);
source = imread(<source image location>);
block_width = 32;

sanitized = perturb(target, source, block_width);
imwrite(sanitized, <destination location>);
```

```
sanitized = imread(<sanitized location>);
restored = reverse_perturbation(sanitized);
```

The first example perturbs a source image to have the same thumbnail as a target image (excluding the metadata region), and saves the sanitized image with its metadata.

The second example loads a sanitized image and reverses the perturbation such that `restored` is identical to the source image.

### Automatic Target Face Selection

Automatic target face selection is done by detect_face.py and can be run from the command line.

Usage:

```
$ python detect_face.py <image location> <dataset location> <output image destination>
```

The script takes in an image, detects faces in it, and replaces them with the image in the dataset that has the highest similarity to the face based on either PSNR or SSIM. The modified image is stored in the given output image destination. This image can then be used as a target image for perturbation.

The script is written to be used with subsets of the UTKFace database, which contains 200x200 images. If using a dataset with different sized images, the script will have to be modified.

The choice of PSNR or SSIM is hardcoded, and can be changed on line 36. 
