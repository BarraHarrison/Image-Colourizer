# Colourizing Black & White Images using Python
import sys
sys.path.append("/Users/barraharrison/Desktop/2025 Coding/Image-Colourizer/venv/lib/python3.9/site-packages")
from deoldify import device
from deoldify.visualize import get_image_colorizer
import os 

# Set up the device
set_fastai_device()

colorizer = get_image_colorizer(artistic=True)

image_path = "images/image_one.jpeg"
output_path = "images/colorized_image.jpg"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Colorizing the image...")
colorizer.plot_transformed_image(
    path=image_path,
    render_factor=35,
    figsize=(8,8),
    results_dir=os.path.dirname(output_path),
    save_path=output_path,
)

print(f"Colourized Image saved at: {output_path}")