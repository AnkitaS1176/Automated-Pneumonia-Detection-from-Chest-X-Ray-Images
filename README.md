# Automated-Pneumonia-Detection-from-Chest-X-Ray-Images
Abstract
Pneumonia presents immense health care challenges to the health care system in the world, especially among the children whereby it leads to death rate among children below the age of five reaching up to fifteen percent. Conventional methods of diagnosis based on the manual analysis of chest radiographs have such constraints as inter-observer variability, time, and resource limitations in remote healthcare. This study examines the use of deep learning techniques to detect pneumonia using automated detection and compares three different convolutional neural network architectures. Based on a dataset of 5,863 pediatric chest X-ray, we compared custom CNN architecture with transfer learning models, VGG16 and ResNet50. Findings show that ResNet50 was better with 93.59 percent accuracy, 99.23 percent recall and 0.9765 score on AUC-ROC. All the models reached the accuracy of more than 89 percent, which confirms the effectiveness of deep learning in classifying medical images.

# Dataset Source : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
 This is a 75-year-old Caucasian female who recently experienced a case of pneumonia.
 The dataset used to train the model is the publicly available Chest X-Ray Images (Pneumonia), which can be found in Kaggle. It has 5,863 JPEG images that will be divided into Pneumonia and Normal cases. The data is divided into training, validation and testing sets. Preprocessing: To process the data in the model all the images were normalized and resized to 224x224. We used data augmentation methods, including random rotations, shifts, zooms and horizontal flips, on the training set. This artificially increases the data and assists the model to better generalize and avoids the overfitting risk.

# Methods: 
We have compared a custom Convolutional Neural Network (CNN) to two existing models based on transfer learning: VGG16 and ResNet50. Custom CNN: This is a baseline model that is developed using a number of convolutional and pooling layers. Transfer Learning: We used models that were pre-trained on a very large dataset ImageNet. Through this mechanism, by simply freezing the weights of the underlying convolutional layers and only training the newly added classification head, we could modify their strong feature-extraction properties to our relevant medical imaging task when using a far smaller dataset. This method proves useful as the elements trained on natural pictures (edges, textures, shapes) tend to be transferred to medical pictures. The residual connections of ResNet50, especially, are useful in the training of deeper networks, which makes it an effective option. This app works with a generative model, which is prompted to perform like a professional in this field with the help of its multimodal features to examine the given image.
# Steps to Run the Code
Drag-and-drop as an X-ray image of a chest in JPEG or PNG. When you are done, click on the "Analyze Image" button to send the image to be analyzed. The AI model also takes the picture and makes a prediction (PNEUMONIA or NORMAL) and a confidence score and probability distribution. To test a new X-ray, click on the Analzye Another Image to clear the interface and restart over again.

# Results Summary
<img width="719" height="575" alt="image" src="https://github.com/user-attachments/assets/9b28f34d-f22f-4801-b9d7-af7ce43d7fd9" />

Training Set: 5,216 images (26.1% Normal, 73.9% Pneumonia)
Validation Set: 16 images (50% Normal, 50% Pneumonia) 
Test Set 624 images (37.5% Normal, 62.5% Pneumonia)


<img width="850" height="792" alt="image" src="https://github.com/user-attachments/assets/e16b00d5-f41c-45a7-bf62-aceb3b12faa8" />

Precision: 93.59% 
Parameters 24.11M (0.52M trainable only)
Inference Time: 22ms


[<img width="1062" height="360" alt="image" src="https://github.com/user-attachments/assets/dffcc0b8-a9ee-4bfe-a18e-1789a293feeb" />]

The image of a ResNet50 Confusion Matrix.
<img width="519" height="582" alt="image" src="https://github.com/user-attachments/assets/4ba04589-9693-4ca6-86f0-b26da29755c8" />

Model comparison
<img width="607" height="549" alt="image" src="https://github.com/user-attachments/assets/e26ab4df-cbb6-461e-9421-a208272a3859" />




# Conclusion
The project manages to prove that deep learning, especially transfer learning using architectures such as ResNet50, are capable of reaching high accuracy levels when it comes to the detection of pneumonia based on the chest X-rays. One of the lessons is that high recall in the medical diagnostic tools is important to ensure that the risk of false negatives is low, one of the metrics in which all the models have excelled. Such a tool can be a useful helper to radiologists and can enhance the efficiency and speed of the diagnostic processes. 

# References
Kermany, D.S., Goldbaum, M., Cai, W. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172, 1122–1131.e9.
Mooney, P. (2018). Chest X-Ray Images (Pneumonia). Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
