# Tea Disease Detection Model 🍃

An advanced deep learning model for detecting diseases in tea plants using computer vision and convolutional neural networks (CNN). This project aims to help tea farmers and agronomists identify plant diseases early and efficiently.

## 🌟 Features

- Detects multiple tea plant diseases from images
- High accuracy (Training accuracy: 99.42%, Validation accuracy: 98.58%)
- Real-time disease prediction
- User-friendly interface via Jupyter notebooks
- Robust CNN architecture with multiple convolutional layers

## 🔧 Model Architecture

The model uses a deep CNN architecture with:
- Multiple convolutional layers (32 -> 64 -> 128 -> 256 -> 512 filters)
- ReLU activation functions
- MaxPooling layers for dimension reduction
- Dropout layers to prevent overfitting
- Dense layers for final classification
- Softmax activation for 8-class classification

## 📊 Performance

- Training Accuracy: 99.42%
- Validation Accuracy: 98.58%
- Loss: Categorical Cross-entropy
- Optimizer: Adam (learning rate: 0.0001)

## 🛠️ Technical Stack

- Python 3.9
- TensorFlow/Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## 📁 Project Structure

```
tea sickness dataset/
├── train/                  # Training dataset
├── test/                   # Testing dataset
├── valid/                  # Validation dataset
├── Train_tea_disease.ipynb # Main training notebook
├── Test_Tea_Disease.ipynb  # Testing and prediction notebook
├── trained_model.keras     # Latest trained model
└── training_hist.json      # Training history
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Tea-diseases-Detection-model.git
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `Train_tea_disease.ipynb` to train the model or `Test_Tea_Disease.ipynb` to test it.

## 📈 Model Training

The model is trained on a comprehensive dataset of tea plant images, including:
- Healthy leaves
- Various disease conditions
- Different lighting conditions
- Multiple angles and perspectives

Training parameters:
- Input image size: 128x128x3
- Batch size: 32
- Learning rate: 0.0001
- Dropout rate: 0.25-0.4 for regularization

## 🔍 Usage

To use the trained model for prediction:

```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('trained_model.keras')

# Prepare your image (ensure it's 128x128x3)
img = prepare_image('your_image.jpg')

# Get prediction
prediction = model.predict(img)
```


## 🙏 Acknowledgments

This project is part of a final year project focusing on agricultural technology and machine learning applications in tea Plantation in Sri Lanka.
