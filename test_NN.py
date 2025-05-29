#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 20:16:20 2025

@author: alpesh
"""

import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import cv2

# Load model
model = load_model('/Users/alpesh/Desktop/ML projects test/Test-ML-projects/mnist_model.keras')

# Load image using OpenCV to enable easier preprocessing
img = cv2.imread('/Users/alpesh/Desktop/ML projects test/Test-ML-projects/test_imag_handwritten/digit5.png', cv2.IMREAD_GRAYSCALE)

# Invert colors (MNIST is white digit on black background)
img = 255 - img

# Resize to 28x28 (while keeping aspect ratio and centering)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

# Normalize to [0,1]
img = img.astype('float32') / 255.0

# Expand dimensions to match model input: (1, 28, 28, 1)
input_tensor = np.expand_dims(img, axis=(0, -1))

# Predict
preds = model.predict(input_tensor)
predicted_digit = np.argmax(preds, axis=1)[0]
print("Predicted digit:", predicted_digit)
