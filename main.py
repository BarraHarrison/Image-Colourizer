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

neural_network = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points = points.transpose().reshape(2, 313, 1, 1)
# LAB (lightness, A and B are colour values)
neural_network.getLayer(neural_network.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
neural_network.getLayer(neural_network.getLayerId("conv8_313_rn")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

