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

# Load the Black & White Image
black_white_image = cv2.imread(image_path)
normalized = black_white_image.astype("float32") / 255.0

lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
light = cv2.split(resized)[0]
light -= 50

neural_network.setInput(cv2.dnn.blobFromImage(light))
ab = neural_network.forward()[0, :, :, :].transpose((1,2,0))
ab = cv2.resize(ab, black_white_image[1], black_white_image[0])

light = cv2.split(lab)[0]
colorized = np.concatenate((light[:,:,np.newaxis], ab), axis=2)
colorized - cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized = (255.0 * colorized.astype("uint8"))

cv2.imshow("B&W Image", black_white_image)
cv2.imshow("Colourized Image", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()