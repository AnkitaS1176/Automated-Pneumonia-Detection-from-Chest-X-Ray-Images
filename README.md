# Automated-Pneumonia-Detection-from-Chest-X-Ray-Images
Abstract
Pneumonia continues to pose substantial challenges to global healthcare systems, particularly affecting pediatric populations with mortality rates reaching fifteen percent among children under five years. Traditional diagnostic workflows relying on manual interpretation of chest radiographs face limitations including inter-observer variability, time constraints, and resource availability in remote healthcare settings. This research investigates automated pneumonia detection through deep learning methodologies, implementing and comparing three distinct convolutional neural network architectures. Utilizing a dataset of 5,863 pediatric chest X-ray images, we evaluated custom CNN architecture alongside VGG16 and ResNet50 transfer learning models. Results demonstrate that ResNet50 achieved superior performance with 93.59% accuracy, 99.23% recall, and 0.9765 AUC-ROC score. All models exceeded 89% accuracy, validating the efficacy of deep learning for medical image classification tasks. 

# Dataset Source : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
The model was trained on the "Chest X-Ray Images (Pneumonia)" dataset, publicly available on Kaggle. It contains 5,863 JPEG images, categorized into Pneumonia and Normal cases. The data is organized into training, validation, and testing sets.
Preprocessing: To prepare the data for the model, all images were resized to 224x224 pixels and normalized. We applied data augmentation techniques—such as random rotations, shifts, zooms, and horizontal flips—to the training set. This artificially expands the dataset, helping the model generalize better and reducing the risk of overfitting.

# Methods: 
Our approach involved comparing a custom Convolutional Neural Network (CNN) against two established architectures using transfer learning: VGG16 and ResNet50.
Custom CNN: A baseline model built from scratch with several convolutional and pooling layers.
Transfer Learning: We leveraged models pre-trained on the vast ImageNet dataset. By freezing the weights of the base convolutional layers and training only the newly added classification head, we could adapt their powerful feature-extraction capabilities to our specific medical imaging task with a much smaller dataset.
This approach is effective because features learned from natural images (edges, textures, shapes) are often transferable to medical images. ResNet50's residual connections, in particular, help in training deeper networks, making it a powerful choice. This app uses a generative model prompted to act as an expert in this domain, leveraging its multimodal capabilities to analyze the image provided.

# Steps to Run the Code
Upload a chest X-ray image in JPEG or PNG format using the drag-and-drop area or file browser.
Click the "Analyze Image" button to submit the image for analysis.
The AI model processes the image and provides a prediction ('PNEUMONIA' or 'NORMAL') along with a confidence score and probability breakdown.
Click "Analyze Another Image" to reset the interface and test a new X-ray.

# Results Summary
<img width="719" height="575" alt="image" src="https://github.com/user-attachments/assets/9b28f34d-f22f-4801-b9d7-af7ce43d7fd9" />

Training Set:   5,216 images (26.1% Normal, 73.9% Pneumonia)
Validation Set: 16 images (50% Normal, 50% Pneumonia)
Test Set:       624 images (37.5% Normal, 62.5% Pneumonia)


<img width="850" height="792" alt="image" src="https://github.com/user-attachments/assets/e16b00d5-f41c-45a7-bf62-aceb3b12faa8" />

Accuracy: 93.59%
Parameters: 24.11M (only 0.52M trainable)
Inference Time: 22ms

[<img width="1062" height="360" alt="image" src="https://github.com/user-attachments/assets/dffcc0b8-a9ee-4bfe-a18e-1789a293feeb" />]

ResNet50 Confusion Matrix
<img width="519" height="582" alt="image" src="https://github.com/user-attachments/assets/4ba04589-9693-4ca6-86f0-b26da29755c8" />

Model comparison
<img width="607" height="549" alt="image" src="https://github.com/user-attachments/assets/e26ab4df-cbb6-461e-9421-a208272a3859" />




# Conclusion
The project successfully demonstrates that deep learning, particularly transfer learning with architectures like ResNet50, can achieve high accuracy in detecting pneumonia from chest X-rays. A key learning is the importance of high recall in medical diagnostic tools to minimize the chance of false negatives, a metric where all models performed well. This tool has the potential to act as a valuable assistant to radiologists, improving the speed and efficiency of diagnostic workflows.
References
Kermany, D.S., Goldbaum, M., Cai, W. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172, 1122–1131.e9.
Mooney, P. (2018). Chest X-Ray Images (Pneumonia). Kaggle. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
