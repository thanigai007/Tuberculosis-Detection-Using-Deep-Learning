import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import random

# Load your best saved model (from your earlier code)
model = load_model("C:/Users/hp/saved_models/VGG16_tb_model.h5")   # replace with actual path

DATASET_PATH = "D:/Project/Guvi_Project/Tuberculosis Detection Using Deep Learning/Dataset/Dataset of Tuberculosis Chest X-rays Images"  # change path accordingly

def load_dataset_info():
    data = []
    for label in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, label)
        for img_file in os.listdir(folder):
            img_path = os.path.join(folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
            if img is not None:
                h, w = img.shape
                brightness = img.mean()
                data.append([label, h, w, brightness])
    return pd.DataFrame(data, columns=["Class", "Height", "Width", "Brightness"])

# Load dataset info
df = load_dataset_info()
# ---------------- Streamlit Sidebar ---------------- #
st.sidebar.title("Menu")
menu = st.sidebar.radio("Go to", ["Introduction", "EDA", "Prediction"])

# ---------------- Introduction Page ---------------- #
if menu == "Introduction":
    st.markdown("""
# 🩺 Tuberculosis Detection Using Deep Learning

## 📌 Problem Statement  
Tuberculosis (TB) remains a major global health issue, particularly in developing countries.  
Manual diagnosis from chest X-rays is often time-consuming and prone to human error.  
The objective of this project is to develop a **deep learning-based system** that can automatically  
detect TB from chest X-ray images, assisting radiologists and improving early diagnosis.

---

## ✅ Solution  
We propose a **transfer learning approach** where pre-trained deep learning models  
are fine-tuned on the TB X-ray dataset. The system is deployed as an **interactive  
Streamlit application**, enabling users to upload X-ray images and receive predictions in real time.  

---

## 🧠 Models Used  
- **ResNet50**  
- **VGG16**  
- **EfficientNetB0**  

After experimentation, the **best performing model** was selected for deployment.

---

## 📊 Evaluation Metrics  
To measure model performance, we used the following metrics:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **ROC-AUC**  

---

## 🛠️ Tools & Technologies  
- **Python** (Deep Learning & Data Processing)  
- **TensorFlow / Keras** (Model Training & Transfer Learning)  
- **NumPy & Pandas** (Data Handling)  
- **Matplotlib & Seaborn** (EDA & Visualization)  
- **Streamlit** (Web App Deployment)  

""")


# ---------------- EDA Page ---------------- #
elif menu == "EDA":
    st.title("Tuberculosis Detection - EDA Section")

    # 1. How many images in each class?
    st.subheader("1. Number of images per class")
    st.bar_chart(df["Class"].value_counts())

    # 2. Distribution of image sizes
    st.subheader("2. Distribution of image sizes")
    fig, ax = plt.subplots()
    sns.scatterplot(x=df["Width"], y=df["Height"], hue=df["Class"], ax=ax)
    st.pyplot(fig)

    # 3. Random sample images
    st.subheader("3. Random Sample Images")
    cols = st.columns(5)
    for i, label in enumerate(df["Class"].unique()):
        sample = df[df["Class"] == label].sample(1).iloc[0]
        img_path = os.path.join(DATASET_PATH, label, random.choice(os.listdir(os.path.join(DATASET_PATH, label))))
        img = cv2.imread(img_path)
        cols[i].image(img, caption=f"{label}", use_container_width=True)

    # 4. Class imbalance pie chart
    st.subheader("4. Class Imbalance (Pie Chart)")
    fig, ax = plt.subplots()
    df["Class"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
    st.pyplot(fig)

    # 5. Average brightness per class
    st.subheader("5. Average Brightness per Class")
    fig, ax = plt.subplots()
    sns.barplot(x="Class", y="Brightness", data=df, ax=ax)
    st.pyplot(fig)

# ---------------- Prediction Page ---------------- #
elif menu == "Prediction":
    st.title("X-ray Prediction")

    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Preprocess image for model
        image_resized = cv2.resize(image, (224, 224))  # size depends on your model
        image_array = img_to_array(image_resized) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        prediction = model.predict(image_array)[0][0]

        # Classification using threshold 0.5
        label = "Tuberculosis Detected" if prediction > 0.5 else "Normal"
        st.subheader(f"Prediction: **{label}**")
        



