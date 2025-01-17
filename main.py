# Colourizing Black & White Images using Python
import numpy as np 
import cv2

# Using Github Repo to train models
# Models: https://github.com/richzhang/colorization/tree/caffe/colorization/models
# Points: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy

# Paths
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/dummy.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'image_one.jpg'

