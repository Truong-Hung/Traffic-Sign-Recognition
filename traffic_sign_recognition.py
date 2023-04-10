# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:32:04 2023

@author: Hung
"""

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
NUM_OUT = 3 #print out the three outputs with highest probability

# Check command-line arguments
if len(sys.argv) != 3:
    sys.exit("Usage: python recognition_traffic_sign.py image.ppm model.h5")
model = tf.keras.models.load_model(sys.argv[2])

data_dir = os.path.join("prediction_images")

entries = os.listdir(data_dir)

predictions = [None]*NUM_CATEGORIES

for i, files in enumerate(entries):
    
    image = cv2.imread(os.path.join(data_dir,str(i)+'.png')) 
    #image = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
    predictions[i] = image

image = cv2.imread(sys.argv[1]) 
image = cv2.resize(image,(IMG_WIDTH,IMG_HEIGHT))

img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# Now, for small images like yours or any similar ones we use for example purpose to understand image processing operations or computer graphics
# Using opencv's cv2.imshow()
# Or google.colab.patches.cv2_imshow() [in case we are on Google Colab]
# Would not be of much use as the output would be very small to visualize
# Instead using matplotlib.pyplot.imshow() would give a decent visualization

classification = model.predict(
    [np.array(image).reshape(1, IMG_WIDTH,IMG_HEIGHT, 3)]
)

outputs = []

for i in range(0,NUM_OUT):
    maxval = (classification.max(),classification.argmax())
    classification = np.delete(classification, classification.argmax())
    outputs.append(maxval)



# plt.subplot(1, 2, 1) # row 1, col 2 index 1
# plt.imshow(img)
# plt.title("Input!")
# plt.xlabel('X-axis ')
# plt.ylabel('Y-axis ')

# plt.subplot(1, 2, 2) # index 2
# plt.imshow(predictions[outputs[0][1]])
# plt.title("My prediction!")
# plt.xlabel('X-axis ')
# plt.ylabel('Y-axis ')

# plt.show()

fig, axs = plt.subplots(3, 2)
axs[0, 0].axis('off')
axs[1, 0].imshow(img)
axs[1, 0].set_title("Input!")
axs[1, 0].axis('off')
axs[2, 0].axis('off')
axs[0, 1].imshow(predictions[outputs[0][1]])
axs[0, 1].set_title("My prediction with a confidence of \n" + str(round(outputs[0][0]*100,2)) + '%')
axs[0, 1].axis('off')
axs[1, 1].imshow(predictions[outputs[1][1]])
axs[1, 1].set_title(str(round(outputs[1][0]*100,2)) + '%')
axs[1, 1].axis('off')
axs[2, 1].imshow(predictions[outputs[2][1]])
axs[2, 1].set_title(str(round(outputs[2][0]*100,2)) + '%')
axs[2, 1].axis('off')
fig.tight_layout()


print(outputs[0][1])

# for i in range(43):
#     plt.figure(i+1)
#     plt.imshow(predictions[i])