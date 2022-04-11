# Facial Sanitization for TPE

## Sanitization and Restoration
Sanitization and restoration are done with the Matlab functions `perturb.m` and `reverse_perturbation.m` respectively. They require a source and target image as input. For face sanitization, the source image should be the unaltered image, and the target image should be a modified version of the source image with the faces swapped with dataset faces. This target image can be produced with the python script in this repo.

Example of sanitization (run in Matlab):
```
target = imread(<target image location>);
source = imread(<source image location>);
block_width = 32;

sanitized = perturb(target, source, block_width);
imwrite(sanitized, <destination location>);
```

Example of restoration (run in Matlab):
```
sanitized = imread(<sanitized location>);
restored = reverse_perturbation(sanitized);
imwrite(restored, <destination location>);

```

## Automatic Target Face Selection and Swapping

Given an image with faces, a target image can be produced with the python script `face_swap.py`. Right now the script is only built to be able to use the UTKFace dataset, which contains 200x200 images.

The overall workflow for sanitizing a face image is to generate a target image with `face_swap.py`, produce the sanitized image with metadata using the Matlab script, and then run TPE on the resulting sanitized image.

### Usage:

First install requirements
```
$ pip install -r requirements.txt
```
If you wish to use the histogram similarity measurement (not really recommended as it's very slow compared to PSNR), you'll need to index your dataset first. This can be done with `index_dataset.py`:
```
python index_dataset.py --dataset=datasets/FilteredUTKFace --index=datasets/filteredUTKFaceIndex.csv
```
Run the face swap (example usage):
```
$ python face_swap.py --img=images/friends.jpg --output=images/friends_target.jpg --dataset=datasets/FilteredUTKFace --method=PSNR
```
Add the argument `--auto=0` to run the script in manual mode. This allows you to manually approve all detected faces to get rid of false positives.

If using histogram similarity measurement, you'll need to add the argument `--index` with the path to your indexed dataset csv as produced by `index_dataset.py`.

