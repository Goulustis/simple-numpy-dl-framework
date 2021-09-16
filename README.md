# simple numpy dl framework

This is a pytorch and tensorflow inspired numpy framework. Written just to get a sense of how the layers are implemented and how the gradients are calculated in the low-level C codes in those framework.

For sample usage, consult test.py. It closely follows how sklearn formulate their functions.

Currently, only support:
- fully connected layers (Fc)
- convolutional layers (Conv2d)
- Sequential classification models

sample_img.npy are a couple images from MNIST