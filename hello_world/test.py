import numpy as xp
import os
from PIL import Image
import cv2
import predictor
import sys

if len(sys.argv) < 2:
    print('Usage: python scratch.py <image path>')
    sys.exit(0)

image_path = sys.argv[1]

orig_image = Image.open(image_path)
image = predictor.predict(orig_image)
image.save('output.png')
