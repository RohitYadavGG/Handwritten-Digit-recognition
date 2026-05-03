# 🧠 Handwritten Digit Recognition Web App

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-Backend-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/Accuracy-99.36%25-brightgreen.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/Status-Production_Ready-success.svg" alt="Status">
</div>

<br>

A professional, cinematic web application that uses a custom **Convolutional Neural Network (CNN)** to recognize handwritten digits. Built with a sleek, responsive UI, it allows users to either draw a digit directly on a canvas or upload a photo of a handwritten number from their phone or computer!

---

## ✨ Features

- 🎨 **Interactive Drawing Canvas**: Draw numbers directly in your browser with real-time prediction.
- 📸 **Smart Image Upload**: Upload any photo from your phone or computer. The custom image processing pipeline automatically crops, centers, and processes real-world photos.
- ⚡ **High Accuracy CNN**: Custom-trained model on the MNIST dataset achieving **99.36%** accuracy.
- 📊 **Cinematic UI/UX**: Premium, responsive interface with beautiful scroll animations, dynamic charts, and detailed confidence metrics.
- 🐳 **Docker Ready**: Easy to deploy anywhere (including free Hugging Face Spaces) with the included Dockerfile.

---

## 🛠️ Technical Stack

- **Machine Learning**: TensorFlow, Keras, NumPy, OpenCV
- **Backend Server**: Python, Flask
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Visualizations**: ApexCharts

---

## 🚀 Step-by-Step Setup Guide

Follow these steps to run the application locally on your machine.

### Prerequisites
Make sure you have [Python 3.9+](https://www.python.org/downloads/) installed.

### 1. Clone the repository
```bash
git clone https://github.com/RohitYadavGG/Handwritten-Digit-recognition.git
cd Handwritten-Digit-recognition
```

### 2. Install Dependencies
It's recommended to create a virtual environment first, but you can directly install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run the App!
Start the Flask server:
```bash
python app.py
```
*The app will be running locally at `http://localhost:7860`*

---

## ☁️ How to Deploy Online for FREE (Hugging Face)

This project is fully configured to be deployed online for free via [Hugging Face Spaces](https://huggingface.co/spaces).

1. Create a free account on Hugging Face.
2. Click on your profile and select **+ New Space**.
3. Name your space, set the License to `MIT`, and choose the **Docker** -> **Blank** template.
4. Upload the following files to your space:
   - `app.py`
   - `image_processor.py`
   - `model_cnn.keras`
   - `requirements.txt`
   - `Dockerfile`
   - `Templates/` folder
   - `Static/` folder
5. Hugging Face will automatically build your app and give you a public, permanent URL!

---

## 🔬 How the Magic Works

### 1. The CNN Model
The model is a Convolutional Neural Network with **475,434** parameters. It has been rigorously tested against 10,000 unseen test images, achieving:
- **Overall Accuracy:** 99.36%
- **Precision:** 99.36%
- **Recall:** 99.36%
- **F1-Score:** 99.36%

### 2. The Image Processing Pipeline (`image_processor.py`)
To handle messy real-world photos uploaded from phones, the backend runs a sophisticated computer vision pipeline:
1. Grayscale conversion & inversion checking
2. Adaptive noise reduction and thresholding
3. Contour detection to find the exact digit boundary
4. Smart bounding-box cropping
5. Aspect-ratio-preserving resize to 28x28
6. Center-of-mass alignment
7. Float normalization (0-1) for the model.

---

## 👨‍💻 Author

**Created by:** Rohit  
**Date:** From April 02,2026 To May 03,2026

Feel free to fork this project, submit pull requests, or use it as a learning tool for combining Machine Learning with beautiful web interfaces!

---