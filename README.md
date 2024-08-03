# Pneumonia-recognition-using-chest-xray
This repository contains a project for detecting pneumonia from chest X-ray images using deep learning techniques. The model is built using TensorFlow/Keras and leverages convolutional neural networks (CNNs) to classify images.

<br/>

Project Overview

Pneumonia is a potentially life-threatening lung infection. Early detection is crucial for effective treatment. This project aims to automate the detection of pneumonia from chest X-ray images using a convolutional neural network (CNN) model.

<br/>

Dataset

The dataset used in this project consists of chest X-ray images labeled as 'Pneumonia' or 'Normal'. The images are preprocessed and augmented to improve the model's performance.

<br/>

Data Augmentation

To prevent overfitting and improve generalization, data augmentation techniques such as rotation, zoom, and horizontal flipping are applied.

<br/>

Model Architecture

The model is a convolutional neural network (CNN) built using TensorFlow/Keras. It consists of multiple convolutional layers followed by max-pooling layers, and fully connected layers.

<br/>

Layers

Convolutional Layers: Extract features from the input images.

MaxPooling Layers: Reduce the spatial dimensions of the feature maps.

Fully Connected Layers: Perform classification based on the extracted features.

Dropout: Prevents overfitting by randomly setting a fraction of input units to 0 at each update during training.

<br/>

Compilation

The model is compiled using the Adam optimizer, with sparse softmax cross-entropy as the loss function. Early stopping and learning rate reduction are used to optimize the training process.

<br/>

Training the Model

The model is trained on the augmented dataset using a defined number of epochs and batch size. Early stopping is used to halt training when the model's performance on the validation set stops improving.

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)

<br/>

Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1-score. Confusion matrix and ROC curve are also generated to visualize the model's effectiveness.

<br/>

Results

The model achieves a high level of accuracy in detecting pneumonia from chest X-rays, demonstrating the effectiveness of deep learning in medical image classification tasks.
