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

<img width="1253" alt="Screenshot 2024-07-12 at 8 26 36 PM" src="https://github.com/user-attachments/assets/d4150a53-9d4a-4d9f-a684-af614713a89d">

<img width="1259" alt="Screenshot 2024-07-12 at 8 27 26 PM" src="https://github.com/user-attachments/assets/dbcde45a-0d08-4e49-8637-a50bb4d3deab">





InceptionV3 As A Feature Extractor
InceptionV3 is a convolutional neural network designed by Google for image classification tasks. It features Inception modules, which efficiently capture features at different scales using parallel convolutional layers. Factorization techniques reduce the computational cost, and auxiliary classifiers aid in training. Pre-trained models are available for quick deployment in various computer vision applications.

<img width="1252" alt="Screenshot 2024-07-12 at 8 30 16 PM" src="https://github.com/user-attachments/assets/ec48884e-9d74-48ed-b34a-49e8143dfead">

<img width="1244" alt="Screenshot 2024-07-12 at 8 30 51 PM" src="https://github.com/user-attachments/assets/d91dfb30-e9c2-49fa-84e1-7af366b01b2b">




Using here the pre-trained model, InceptionV3 named as inception_model.

<img width="1247" alt="Screenshot 2024-07-12 at 8 31 34 PM" src="https://github.com/user-attachments/assets/5b3a8b35-a08a-492e-bb91-a783791e6d77">

<img width="1261" alt="Screenshot 2024-07-12 at 8 31 53 PM" src="https://github.com/user-attachments/assets/8f752082-2034-4721-943e-31f862b2d0d5">


The accuracy still is very low with InceptionV3, even if it's the same for all epochs.





ResNet50 As A Feature Extractor

ResNet50 is a convolutional neural network architecture that belongs to the ResNet (Residual Network) family. It consists of 50 layers, including convolutional, pooling, and fully connected layers. ResNet50 is renowned for its deep structure, utilizing residual connections to address the vanishing gradient problem during training. These connections allow for easier training of deep networks by facilitating the flow of gradients, enabling better performance in tasks such as image classification and object detection. ResNet50 has been widely adopted in computer vision tasks due to its effectiveness and efficiency.

<img width="1237" alt="Screenshot 2024-07-12 at 8 34 15 PM" src="https://github.com/user-attachments/assets/15363265-93c9-4965-ab90-aab34c4af753">

<img width="1242" alt="Screenshot 2024-07-12 at 8 34 43 PM" src="https://github.com/user-attachments/assets/ac36448d-fd14-4e05-a333-b04025947370">



The provided code initializes a pre-trained ResNet50 model with weights from ImageNet, freezes its layers to prevent further training, and adds additional layers for classification. It then compiles and trains the model on the provided training and test datasets for 15 epochs. This approach leverages the powerful feature extraction capabilities of ResNet50 for a specific classification task while customizing the output layers for the desired number of classes.

<img width="1246" alt="Screenshot 2024-07-12 at 8 35 14 PM" src="https://github.com/user-attachments/assets/667cf7b5-f363-4084-8281-8b8528a4e6c1">

<img width="1243" alt="Screenshot 2024-07-12 at 8 36 02 PM" src="https://github.com/user-attachments/assets/fa95de5a-116f-460d-9d4b-2c52704f408e">



Here, the accuracy increased, but still, the loss is high so we will go for other models.

Xception As A Feature Extractor
Xception is a convolutional neural network architecture introduced by Google, known for its depthwise separable convolutions. It aims to improve upon traditional convolutional architectures by decoupling the spatial and channel-wise convolutions, thereby reducing computational complexity while maintaining or even enhancing performance. This design allows Xception to achieve state-of-the-art results on various computer vision tasks, including image classification and object detection, with a more efficient use of parameters and computational resources.
<img width="1234" alt="Screenshot 2024-07-12 at 8 38 31 PM" src="https://github.com/user-attachments/assets/e953fc81-0b28-407e-a1f7-4c8bd2900e4d">

<img width="1241" alt="Screenshot 2024-07-12 at 8 39 03 PM" src="https://github.com/user-attachments/assets/2c8d0270-ac51-4985-84f2-b9ffa423debf">


The output for this is
<img width="1248" alt="Screenshot 2024-07-12 at 8 39 24 PM" src="https://github.com/user-attachments/assets/d7fb5bf5-33b4-4e65-8128-535b30dc4fd9">

<img width="1237" alt="Screenshot 2024-07-12 at 8 40 08 PM" src="https://github.com/user-attachments/assets/694b4301-53ac-40f4-85f6-40e3165ed0b9">



So, here we are getting a decent accuracy of 79.9%. Let's check for other models if it is getting increased. 

DenseNet121 As A Feature Extractor
DenseNet-121 is a convolutional neural network architecture designed for image classification tasks. It belongs to the Dense Convolutional Network (DenseNet) family, characterized by densely connected layers where each layer is connected to every other layer in a feed-forward fashion. This architecture facilitates feature reuse and encourages feature propagation throughout the network, leading to more efficient parameter usage and better gradient flow during training.
<img width="1236" alt="Screenshot 2024-07-12 at 8 42 44 PM" src="https://github.com/user-attachments/assets/97541743-d306-4bbc-a4dd-cd9ce5690add">

<img width="1230" alt="Screenshot 2024-07-12 at 8 43 11 PM" src="https://github.com/user-attachments/assets/ba15e1a0-7df3-4b26-84cf-8432cef11163">



Using DenseNet121, we are trying to build a model for better accuracy.

<img width="1252" alt="Screenshot 2024-07-12 at 8 43 54 PM" src="https://github.com/user-attachments/assets/315ddcdc-fafd-418f-bfa8-c49477b143ad">


<img width="1240" alt="Screenshot 2024-07-12 at 8 44 27 PM" src="https://github.com/user-attachments/assets/9007af61-78d2-43eb-a1f9-f1c7651b8892">



So, the accuracy increased by 87% which is the more predictable model as compared to before.

MobileNetV2 As A Feature Extractor
MobileNetV2 is a lightweight convolutional neural network architecture optimized for mobile and embedded devices with limited computational resources. Developed by Google, it builds upon the original MobileNet architecture, introducing inverted residual blocks and linear bottlenecks to improve efficiency and performance.

1. Inverted Residual Blocks:

   MobileNetV2 introduces inverted residual blocks, which use lightweight depthwise separable convolutions with shortcut connections.
   These blocks allow for efficient feature extraction by reducing the number of parameters while maintaining representational capacity.

2. Linear Bottlenecks:

    Linear bottlenecks are employed to prevent information loss during feature extraction.
    This design choice enables MobileNetV2 to achieve higher accuracy compared to its predecessor while keeping computational costs low.

3. Efficiency and Versatility:

    MobileNetV2 is highly efficient, making it suitable for deployment on resource-constrained devices like smartphones and IoT devices.
   It has been widely used for various computer vision tasks such as image classification, object detection, and semantic segmentation, balancing performance and resource requirements effectively.
<img width="1235" alt="Screenshot 2024-07-12 at 8 47 11 PM" src="https://github.com/user-attachments/assets/de72335d-1744-456b-adab-2e42bdcf53b6">

<img width="1236" alt="Screenshot 2024-07-12 at 8 47 34 PM" src="https://github.com/user-attachments/assets/2c2840cd-27d9-446a-b9dd-501b83815a9c">



Overall, MobileNetV2 offers an excellent trade-off between model size, speed, and accuracy, making it a popular choice for real-time applications and scenarios where computational resources are limited.

<img width="1250" alt="Screenshot 2024-07-12 at 8 48 05 PM" src="https://github.com/user-attachments/assets/176ebbe1-0840-4490-97cc-7a6ba3132ec4">

<img width="1236" alt="Screenshot 2024-07-12 at 8 48 43 PM" src="https://github.com/user-attachments/assets/3151007d-bde2-4011-866a-fde57a58eee8">


Used MobileNetV2 to get the accuracy above 90% and even the loss is less as compared to other models.

Testing The Model
Model Testing is the process of evaluating the performance of a deep learning model on a dataset that it has not seen before. It is a crucial step in the development of any machine learning model, as it helps to determine how well the model can generalize to new data.
<img width="835" alt="Screenshot 2024-07-12 at 8 53 49 PM" src="https://github.com/user-attachments/assets/3b9c4331-bd22-4c69-8f29-7285b7e727a9">

<img width="811" alt="Screenshot 2024-07-12 at 8 55 56 PM" src="https://github.com/user-attachments/assets/f39117e1-f1d4-4c94-a471-f839fe49986e">

<img width="1091" alt="Screenshot 2024-07-12 at 8 58 38 PM" src="https://github.com/user-attachments/assets/df21ab18-a703-4dd4-86cc-3af045f5d4a5">




In the above code, we have tested the model with 5 classes 'combat', 'destroyed_building', 'fire', 'humanitarian', and 'vehicles'.
Opening and Preprocessing: First, it opens an image ('/content/war_events/Combat/10.jpeg') and resizes it to a standard size (224x224 pixels) suitable for the machine learning model being used.
Converting to Model Input: Then, it converts the image from a format understandable by humans (like a jpeg) to a numerical format (array) that the machine learning model can process. It also scales the pixel values between 0 and 1 for better model performance. Finally, it adds an extra dimension to the array to match the model's input requirements.
Prediction Time: With the preprocessed image, the code feeds it to the model_new1 (presumably trained to recognize war events) to get predictions.
Decoding the Prediction: The model outputs a set of probabilities for different categories (e.g., combat, destroyed building). The code identifies the category with the highest probability and retrieves the corresponding label from a predefined list (class_labels). This gives you the predicted class, which is the most likely war event depicted in the image.
Visualization: Finally, the code displays the original image and overlays the predicted class label as a title, allowing you to see the image and the model's interpretation side-by-side. It also hides the axis labels for a cleaner presentation.


Application Building

In this section, we will be building a web application that is integrated into the model we built. A UI is provided for the uses where he has to enter the values for predictions. The enter values are given to the saved model and prediction is showcased on the UI.
This section has the following tasks
Building HTML Pages
Building server-side script

<img width="1440" alt="Screenshot 2024-07-20 at 12 36 32 AM" src="https://github.com/user-attachments/assets/78915fbd-5ba8-469b-a7b4-688ad1eec5ef">

<img width="1440" alt="Screenshot 2024-07-20 at 12 57 07 AM" src="https://github.com/user-attachments/assets/9d2d42dd-b62e-4616-b273-b5d39c4b61a5">

<img width="1436" alt="Screenshot 2024-07-20 at 12 57 42 AM" src="https://github.com/user-attachments/assets/5e5a41c7-ab3e-4bba-bff5-9a39ccd9efb3">

<img width="898" alt="Screenshot 2024-07-20 at 12 25 09 AM" src="https://github.com/user-attachments/assets/ec267155-b50d-4045-bf10-10dddda1aba4">

<img width="1187" alt="Screenshot 2024-07-20 at 12 26 12 AM" src="https://github.com/user-attachments/assets/18cb695d-fbd3-4511-ba2b-5a5fde86fc85">







