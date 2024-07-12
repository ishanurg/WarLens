WarLens: Transfer Learning For Event Classification In Conflict Zones
WarLens is an innovative project utilizing transfer learning techniques to classify events in conflict zones by analyzing multimedia data such as images and videos. By leveraging pre-trained deep learning models and fine-tuning them on data specific to conflict zones, WarLens can accurately identify and classify various events such as protests, military activities, humanitarian crises, and infrastructure damage.

Scenario 1: In a conflict zone, a non-governmental organization (NGO) is monitoring social media and news outlets for reports of violence and human rights abuses. WarLens provides them with a tool to automatically analyze multimedia content and classify events, enabling rapid response and intervention to assist affected populations and document human rights violations.

Scenario 2: A government agency is responsible for monitoring security and stability in a conflict-affected region. By integrating WarLens into their surveillance systems, they can analyze live feeds from security cameras and drones to detect and classify relevant events in real time, enabling the timely deployment of resources and response teams to mitigate threats and maintain order.

Scenario 3: A media organization is reporting on developments in a conflict zone and needs to verify the authenticity of user-generated content shared on social media platforms. WarLens helps them analyze images and videos to confirm the occurrence of events and provide context for their reporting, enhancing the accuracy and credibility of their news coverage.

Technical Architecture:

<img width="1232" alt="Screenshot 2024-07-12 at 8 01 45 PM" src="https://github.com/user-attachments/assets/02fc1e6e-afb5-4d54-9131-8730665b68f1">

PRE-REQUISITES

To complete this project, you must require the following software, concepts, and packages

Anaconda navigator and PyCharm / Spyder:
Refer to the link below to download Anaconda Navigator
Link (PyCharm) : https://youtu.be/1ra4zH2G4o0
Link (Spyder): https://youtu.be/5mDYijMfSzs
Python packages:
Open anaconda prompt as administrator
Type “pip install numpy” and click enter.
Type “pip install pandas” and click enter..
Type “pip install tensorflow==2.3.2” and click enter.
Type “pip install keras==2.3.1” and click enter.
Type “pip install Flask” and click enter.

PROJECT OBJECTIVES

By the end of this project, you’ll understand the following:
      Preprocessing the images.
      Applying Transfer learning algorithms on the dataset.
      How deep neural networks detect fake images.
      You will be able to know how to find the accuracy of the model.
      You will be able to Build web applications using the Flask framework.


PROJECT FLOW

The user interacts with the UI (User Interface) to choose the image.
The chosen image is analyzed by the model which is integrated with the flask application.
The Model analyzes the image, then the prediction is showcased on the Flask UI.

To accomplish this, we have to complete all the activities and tasks listed below
Data Collection.
Create a Train and Test path.
Data Pre-processing.
Import the required library
Configure ImageDataGenerator class
ApplyImageDataGenerator functionality to Trainset and Testset
Model Building
Pre-trained CNN model as a Feature Extractor
Adding Dense Layer
Configure the Learning Process
Train the model
Save the Model
Test the model
Application Building
Create an HTML file
Build Python Code

DATA COLLECTION 
War events classification | Kaggle...

https://www.kaggle.com/datasets/saailna/war-events-classification

CREATE TRAINING AND TESTING DATASET

To build a DL model we have six classes in our dataset. But in the project dataset folder training and testing data are needed. So, in this case, we just have to assign a variable and pass the folder path to it.

Three different transfer learning models are used in our project and the best model is selected.

The image input size of the model is 224,224.

IMAGE PRE-PROCESSING

Now it's time to Build input and output layers for the VGG16 model
Hidden layers freeze because they have trained sequences, so changing the input and output layers.

IMPORTING THE LIBRARIES
Import the necessary libraries and modules as shown in the image
<img width="1071" alt="Screenshot 2024-07-12 at 8 14 33 PM" src="https://github.com/user-attachments/assets/23f3570f-5683-4c2b-bb45-8bf9aad2b639">


Configure ImageDataGenerator Class

ImageDataGenerator class is instantiated and the configuration for the types of data augmentation
There are five main types of data augmentation techniques for image data; specifically:
Image shifts via the width_shift_range and height_shift_range arguments.
The image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image shear via the shear_range argument.
Image zoom via the zoom_range argument.
An instance of the ImageDataGenerator class can be constructed for train and test. 

<img width="1081" alt="Screenshot 2024-07-12 at 8 16 28 PM" src="https://github.com/user-attachments/assets/7dd154b5-a7fe-4190-988a-8622e2b7afb7">


Apply ImageDataGenerator Functionality To Training_data And Validation_data

Let us apply ImageDataGenerator functionality to the Training_data and Validation_data by using the following code. For Training set keras. preprocessing.image_dataset_from_directory function.
This function will return batches of images from the subdirectories
Arguments:
directory: Directory where the data is located. If labels are "inferred", it should contain subdirectories, each containing images for a class. Otherwise, the directory structure is ignored.
batch_size: The size of the batches of data which is 32.
target_size Size to resize images after they are read from disk.
class_mode:
-  ‘int': means that the labels are encoded as integers (e.g. for sparse_crossentropy loss). 
- 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss). 
- 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy).

<img width="1089" alt="Screenshot 2024-07-12 at 8 21 05 PM" src="https://github.com/user-attachments/assets/859868c1-f654-45da-921b-fc45dfd5cafc">

EfficientNet B0 As A Feature Extractor
EfficientNet B0 is a convolutional neural network architecture designed to achieve state-of-the-art performance while maintaining efficiency in terms of model size and computational resources. It was introduced by Google's research team in 2019. The "B0" variant represents the baseline model in the EfficientNet series.

EfficientNet B0 achieves its efficiency by using a compound scaling method that balances network depth, width, and resolution. This method scales up all dimensions (depth, width, and resolution) of the network in a principled manner, resulting in improved performance without significantly increasing computational cost. It incorporates techniques such as depth-wise separable convolutions and channel attention mechanisms to further optimize performance.

EfficientNet B0 is particularly well-suited for tasks such as image classification and object detection, where achieving high accuracy with limited computational resources is crucial. It has become a popular choice for various computer vision applications due to its excellent trade-off between model size, speed, and accuracy.

1. Model Definition:
   `Sequential`: This defines a linear stack of layers.
   `efficientnet_model`: This represents a pre-trained EfficientNet model, which serves as the feature extractor.
   `GlobalAveragePooling2D () `: This layer pools the spatial dimensions of the feature maps from the EfficientNet model to produce a fixed-size vector.
   `Dense (128, activation='relu') `: This fully connected layer with 128 neurons and ReLU activation function helps in learning non-linear patterns in the extracted features.
   `Dropout (0.5) `: Dropout is a regularization technique that randomly sets a fraction of input units to zero during training to prevent overfitting.
   `Dense (5, activation='softmax') `: The final fully connected layer with 5 neurons (assuming it's a multi-class classification problem) and softmax activation function, which outputs probabilities for each class.

2. Model Compilation:
   `model4.compile`: This compiles the model, specifying the optimizer (Adam), loss function (categorical cross-entropy), and evaluation metrics (accuracy).

3. Model Training:
   `model4.fit`: This trains the model on the training data (`train1`) for a specified number of epochs (10 in this case), with steps per epoch and validation steps specified by the lengths of the training and test datasets, respectively.

<img width="1232" alt="Screenshot 2024-07-12 at 8 23 35 PM" src="https://github.com/user-attachments/assets/e39d4905-2cc0-43b6-98c1-db7d9803cfe8">

<img width="1234" alt="Screenshot 2024-07-12 at 8 24 13 PM" src="https://github.com/user-attachments/assets/08d8dca7-1e61-471f-8e98-9c6bef9da27c">











