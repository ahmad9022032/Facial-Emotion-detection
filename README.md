# Facial-Emotion-detection
A Jupyter Notebook for building, training, and evaluating a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras.

## Table of Contents
### Installation
Usage
Project Structure
Model Architecture
Training
Evaluation
Results
Contributing

## Installation
To run this project, you'll need to have Python 3.x installed along with the following libraries:

TensorFlow
Keras
numpy
matplotlib
You can install these libraries using pip.

## Model Architecture
The model implemented in this project is a Convolutional Neural Network (CNN) with the following components:

. Convolutional layers to extract features from images.
. Max Pooling layers to reduce the dimensionality of the feature maps.
. Dropout layers to prevent overfitting.
. Dense (Fully Connected) layers to perform the final classification.
### Training
To train the model, ensure that your training data is organized in the Data/train/ directory and validation data in the Data/test/ directory. The notebook contains code to preprocess the data, define the model, and train it.

## Evaluation
The notebook includes cells to evaluate the model on the test dataset. This includes calculating metrics such as accuracy and loss, as well as generating visualizations to help understand the model's performance.

## Results
Training Accuracy and Loss
![training performance](https://github.com/ahmad9022032/Facial-Emotion-detection/assets/94382791/fd60efed-4fb2-439c-95fa-3a3d954d2674)

## Confusion Matrix
![Confusion matrix](https://github.com/ahmad9022032/Facial-Emotion-detection/assets/94382791/2d3948ad-4351-4f73-853d-f422394ba70f)

Sample Predictions

Contributing
Contributions to this project are welcome. If you have any improvements, bug fixes, or new features, please open an issue or submit a pull request.

License
