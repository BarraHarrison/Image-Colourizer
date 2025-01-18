# Colourizing Black & White Images using Python
import sys
sys.path.append("/Users/barraharrison/Desktop/2025 Coding/Image-Colourizer/venv/lib/python3.9/site-packages")
from deoldify import device
from deoldify.visualize import get_image_colorizer
import os 

# Set up the device
set_fastai_device()

colorizer = get_image_colorizer(artistic=True)

# Paths
image_path = "images/image_one.jpeg"
output_path = "images/colorized_image.jpg"


points = points.transpose().reshape(2, 313, 1, 1)
# LAB (lightness, A and B are colour values)
neural_network.getLayer(neural_network.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
neural_network.getLayer(neural_network.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the Black & White Image
black_white_image = cv2.imread(image_path)
if black_white_image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit(1)


normalized = black_white_image.astype("float32") / 255.0

lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224, 224))
light = cv2.split(resized)[0]
light -= 50

neural_network.setInput(cv2.dnn.blobFromImage(light))
ab = neural_network.forward()[0, :, :, :].transpose((1,2,0))
ab = cv2.resize(ab, (black_white_image.shape[1], black_white_image.shape[0]))

light = cv2.split(lab)[0]
colorized = np.concatenate((light[:,:,np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized = (255.0 * colorized.astype("uint8"))

cv2.imshow("B&W Image", black_white_image)
cv2.imshow("Colourized Image", colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()