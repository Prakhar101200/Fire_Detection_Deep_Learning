# Fire_Detection_Deep_Learning
Overview
This project aims to develop a deep learning-based fire detection system using convolutional neural networks (CNNs). The system is designed to analyze images and classify them into two categories: "fire" and "no_fire". The model is trained on a dataset of images containing examples of both fire and non-fire scenarios.

Dataset
The dataset used for training and evaluation consists of images sourced from eduonix learning center. Each image in the dataset was not labeled. But, it was copied to copuple of sub-folders names as 'With_Fire' and 'Without_fire'. This was then further labeled with its corresponding class: "fire" or "no_fire" while importing the images in jupyter Notebook. The dataset is divided into training and test sets to assess the performance of the model.

Model Architecture
The CNN model architecture used for fire detection consists of multiple convolutional and pooling layers followed by fully connected layers. The architecture is designed to extract relevant features from input images and learn discriminative patterns to differentiate between fire and non-fire scenarios.

Training
The model is trained using the training dataset with the objective of minimizing the categorical cross-entropy loss function. During training, data augmentation techniques such as random rotation, shifting, flipping, etc, are applied to increase the robustness of the model and prevent overfitting. The training process involves iterating over multiple epochs while monitoring performance metrics such as accuracy and loss on training set.

Evaluation
After training, the model's performance is evaluated using the test dataset to assess its ability to generalize to unseen data. Performance metrics such as accuracy were computed to measure the effectiveness of the model in detecting fire incidents.

Results
The trained model achieves a relatively high accuracy score on the test dataset, indicating its effectiveness in accurately classifying images as "fire" or "no_fire". The model's performance is further validated through visual inspection of sample predictions and analysis of classification metrics.
Hardships Faced
Low Accuracy: Initially, the model faced challenges with achieving high accuracy, which led to frustration and required several iterations to improve the model's performance.

Confusion with Parameters: Adjusting model parameters and hyperparameters was a daunting task, and finding the optimal configuration was a trial-and-error process.

Challenges with Data Augmentation: Understanding and implementing data augmentation techniques posed difficulties, particularly due to the nature of the problem being addressed. However, with perseverance and online resources, this obstacle was highly overcome.

Strategy
Data Preparation: Images were manually categorized into subfolders representing fire and no_fire classes. A custom function was developed to import all images along with their respective labels.

Model Development: A CNN architecture was constructed, and various iterations were made by adding layers and adjusting parameters based on experimentation and research.

Trial and Error: Multiple runs of the model were conducted, and changes were made iteratively in the code and parameters to optimize performance.

Cancellation of Callbacks: Initially, callbacks were used to monitor training progress, but eventually, they were canceled as they did not significantly contribute to improving results.

Personal Reflection.
Persistence and Learning: Despite the challenges, the project provided an opportunity to learn and grow as a coder. The perseverance to overcome obstacles ultimately led to successful results.

Room for Improvement: Acknowledging that there is always room for improvement, future iterations of the project can focus on refining the code and exploring additional techniques for further enhancing accuracy.

Usage
To use the fire detection system:

Install the necessary dependencies and libraries.
Preprocess the input images (if required) and ensure they are in the correct format.
Load the trained model weights.
Use the model to make predictions on new images or video streams.

Future Work
Future enhancements and improvements to the fire detection system may include:

Fine-tuning the model architecture and hyperparameters for better performance.
Exploring advanced techniques such as transfer learning and ensemble learning.
Integrating the model into real-time fire detection systems for deployment in various environments.
Contributors
Prakhar Raj Gupta

Acknowledgements
I would like to acknowledge the numerous sources available over the internet that helped me throughout this project hardships. Along with this, I would like to acknowledge "Miss Suchisita Mondal" for teaching me this concept in the class. 
I would also like to appreciate 'Eduonix' for creating this Data Science Certification Module that enabled me to explore the world of Data Science and make myself more knowledgable in the concepts. 
Lastly, I would like to appreciate myself for not giving up and fighting all obstacles on daily basis, whether its in coding or any other theoritical concepts or in life. 
