# Python Image-Colourizer
This project transforms Black & White Images into colour using deep learning models.
Initially I used the OpenCV libraries for colourization, but due to outdated and unavailable pre-trained models, I had to switch to Deoldify.
DeOldify is a deep learning library used for image and video colourization.

# Overview

## OpenCV-Based Colourization
- The project initially began with a pre-trained Caffe model
- The pre-trained model was no longer accessable in the github repo.


## Switching to DeOldify
- DeOldify is an alternative and older library to OpenCV
- There was issues with the fastai version and the DeOldify dependencies. For some reason, they were not compatible and the code did not run properly (Oh well!)


## Tools & Libraries
- OpenCV
- DeOldify
- FastAI
- PyTorch