🛍️ **Fashion MNIST Classifier using ANN**

📌 **Overview**

This project implements an Artificial Neural Network (ANN) to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images of 10 different fashion categories, including T-shirts, shoes, bags, and more. The goal of this project is to build an accurate deep learning model that can distinguish between these categories effectively.

📂**Dataset**

The Fashion MNIST dataset is a well-known dataset created by Zalando, consisting of:

- 60,000 training images

- 10,000 test images

- Each image is 28x28 pixels in grayscale

- 10 classes:

   - T-shirt/top

   - Trouser

   - Pullover

   - Dress

   - Coat

    - Sandal

   - Shirt

   - Sneaker

   - Bag

   - Ankle boot

🚀 **Model Architecture**

The Artificial Neural Network (ANN) model follows a structured architecture:
- Input Layer: 784 neurons (flattened 28x28 images)

- Hidden Layers:

  - Dense layer with 512 neurons (LeakyReLU activation)

  -  Dense layer with 256 neurons (LeakyReLU activation)
  -  Dense layer with 128 neurons (LeakyReLU activation)

- Output Layer: 10 neurons (Softmax activation for classification)

🛠️ **Technologies Used**

* Python 🐍

* TensorFlow & Keras 🤖

* NumPy & Pandas 📊

* Matplotlib & Seaborn 📈

* Jupyter Notebook 📓

📊 **Model Training**

The model is trained using the following configuration:

- Loss Function: Categorical Crossentropy

- Optimizer: Adam

- Batch Size: 32 

- Epochs: 50

**Evaluation Metric** :  Accuracy  and loss

📌 **Results**

 - Achieved high accuracy on both training and test datasets.

 - Properly classified most fashion items with minimal misclassification.

 - Visualized model performance using loss and accuracy plots.

 📌 **Future Improvements**

 - Experimenting with CNNs (Convolutional Neural Networks) for better accuracy.

- Implementing data augmentation techniques to improve generalization.

- Deploying the model as a web application for real-time classification.