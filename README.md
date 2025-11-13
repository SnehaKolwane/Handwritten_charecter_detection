This project is a Machine Learning-based terminal application that recognizes handwritten characters (A–Z, a–z) and digits (0–9) from images using a Convolutional Neural Network (CNN) trained on the EMNIST dataset (Extended MNIST).

The model reads grayscale images of handwritten characters and predicts the correct alphabet or digit without any GUI or web interface — everything runs directly from the terminal.

🎯 Features

 Recognizes both digits (0–9) and letters (A–Z, a–z)
 Uses TensorFlow and EMNIST dataset
 Works entirely offline in the terminal
 Saves the trained model (model_unified.h5) for future use
 Simple and lightweight (no GUI or Streamlit needed)

 Technologies Used

Python 3.8+

TensorFlow / Keras → Deep Learning

TensorFlow Datasets (TFDS) → Automatic EMNIST download

NumPy, Pillow (PIL) → Image handling

CNN (Convolutional Neural Network) → Character recognition model

# Installation & Setup
1 Clone or Download the Repository
git clone https://github.com/yourusername/Handwritten_Recognition.git
cd Handwritten_Recognition

2. Install Dependencies
pip install -r requirements.txt


#requirements.txt

tensorflow==2.14.0
tensorflow-datasets==4.9.4
numpy==1.26.4
pillow==10.0.0

#Training the Model

Run this once to train and save the model:

python train_model.py


# Output example:

 Downloading EMNIST dataset (byclass) using TensorFlow Datasets...
 Dataset loaded with 62 classes
 Training model...
 Model accuracy: 89.32%
 Model saved successfully as model_unified.h5


After completion, the trained model file model_unified.h5 will appear in your project folder.

# Predicting Handwritten Characters

To recognize any handwritten character or digit:

python predict_image.py


Then enter the image path when prompted, for example:

Enter image path (e.g., sample.png): test_A.png

# Example Output:

 Model loaded successfully (Digits + Letters).
 Predicted Character: A
 Confidence: 96.87%

# Preparing Test Images

For best results:

Use white background and black writing

Save one character or digit per image

Center the character (no big margins)

Use .png or .jpg images

Clear print-style handwriting (avoid cursive)

# Model Details
Parameter	Description
Dataset	EMNIST (ByClass)
Classes	62 (0–9, A–Z, a–z)
Input Shape	28×28 grayscale
Model Type	CNN
Accuracy	~88–90%
🧠 How It Works

Loads the EMNIST dataset (62 classes).

Preprocesses images to 28×28 grayscale format.

Trains a Convolutional Neural Network (CNN) to classify characters.

Saves the trained model for reuse.

During prediction, the app:

Loads your image

Preprocesses (resize, invert, normalize)

Predicts the closest character or digit

Displays confidence score



# Example Output Screenshot
 Model loaded successfully (Digits + Letters).
 Predicted Character: G
 Confidence: 97.54%
