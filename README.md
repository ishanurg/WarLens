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






