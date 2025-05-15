🍓 Fruit Image Classifier using Deep Learning

This repository contains a Convolutional Neural Network (CNN) model trained to classify fruit images. The model is built and trained using Keras and TensorFlow, and it can be used to categorize images of fruits into various types.

👨‍💻 Made by:

Sanchit (2K23CSUN01304)
Jai Gupta (2K23CSUN01294)
Drish Bhalla (2K23CSUN01291)

🧠 Model Overview

Model Name: FruitClassifier_model.h5
Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input: Image of fruit

Output: Fruit category prediction

📁 Files Included
File Name	Description
FruitClassifier_model.h5	Trained Keras CNN model for fruit image classification
app.py (optional)	Streamlit app file for running the prediction UI

🚀 How to Use the Model
Clone the repository
git clone https://github.com/Kavay2005/ULNN_project.git
cd your-repo-name

2. Install the dependencies

pip install tensorflow numpy pillow streamlit

3. Load the model in your Python code

from tensorflow.keras.models import load_model
model = load_model('fruit_classifier_model.h5')

4. Make predictions on fruit images

import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('test_fruit.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
print("Predicted class:", prediction)
python
Copy
Edit
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('test_fruit.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
print("Predicted class:", prediction)
