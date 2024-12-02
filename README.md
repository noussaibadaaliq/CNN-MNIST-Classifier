Handwritten Digit Recognition with Convolutional Neural Networks (CNN)

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to recognize handwritten digits using the MNIST dataset. The model is trained to classify images of digits (0-9) and evaluate its performance through accuracy and loss metrics.

Project Structure: 

Dataset: The project uses the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
Libraries: Utilizes popular Python libraries for deep learning (TensorFlow, Keras) and data visualization (Matplotlib).
Model Architecture:
Convolutional layers with ReLU activation.
MaxPooling and Dropout layers for improved regularization.
Fully connected dense layers for final classification.
Evaluation: The model is evaluated on the test set and plots accuracy and loss over training epochs.
How to Use This Notebook
Install Required Libraries: Ensure you have Python and the following libraries installed:

bash
Copier le code
pip install numpy matplotlib tensorflow keras
Run the Notebook: Open the notebook in Jupyter and execute the cells to train and test the CNN model.

Results:

The final accuracy and loss are printed at the end.
Graphs of training accuracy and loss are displayed to visualize model performance.
Code Explanation
Data Preparation
The MNIST dataset is loaded and preprocessed (reshaped and normalized).
The labels are one-hot encoded for categorical classification.
Model Architecture
The CNN model is built using Sequential API with:
Convolutional layers for feature extraction.
MaxPooling layers to reduce spatial dimensions.
Dropout layers to prevent overfitting.
Fully connected layers for classification.
Training and Evaluation
The model is compiled using categorical_crossentropy loss and Adam optimizer.
The model is trained for 100 epochs with a batch size of 32.
Training accuracy and loss are plotted for analysis.
Predictions
The model's predictions on the test data are displayed, including a visualization of the first test image and its predicted digit.
Acknowledgements
Dataset: The MNIST dataset used in this project is publicly available and widely used for digit recognition tasks.
Libraries: This project leverages the capabilities of TensorFlow and Keras for building deep learning models and Matplotlib for data visualization.
