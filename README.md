# Enchanted-Wings-Marvels-of-Butterfly-Species-using-AI-ML
Here is a dynamic and professional `README.md` file tailored for your project on GitHub:
**"Enchanted Wings: Marvels of Butterfly Species using AI/ML"**

---

```markdown
# 🦋 Enchanted Wings: Marvels of Butterfly Species using AI/ML

Welcome to the official repository of **Enchanted Wings**, a deep learning-powered butterfly species classification project that leverages Transfer Learning to support biodiversity, conservation, education, and ecological research.

![Butterfly Banner](https://upload.wikimedia.org/wikipedia/commons/2/25/Monarch_In_Motion.jpg)

---

## 📌 Overview

This project uses image classification techniques, particularly **Transfer Learning with MobileNetV2**, to identify butterfly species from photos. The model is trained on a dataset comprising **75 butterfly species** with **6,499 images** sourced from [Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification). A **Streamlit web application** is built for user interaction, allowing real-time predictions with species information.

---

## 🛠 Features

- ✅ Upload butterfly image and get instant classification
- ✅ Real-time species name and description lookup
- ✅ Interactive and user-friendly Streamlit UI
- ✅ Uses Transfer Learning (MobileNetV2)
- ✅ Trained on diverse butterfly species
- ✅ Educational and conservation value

---

## 📁 Project Structure

```

Enchanted-Wings/
│
├── butterfly\_model.h5           # Trained AI model
├── butterfly\_info.json          # Species name & description
├── app.py                       # Streamlit web application
├── requirements.txt             # Required libraries
├── README.md                    # Project documentation
└── dataset/                     # Butterfly image dataset (train/test/val)

````

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/DamaGanesh/Enchanted-Wings-Marvels-of-Butterfly-Species-using-AI-ML.git
cd Enchanted-Wings-Marvels-of-Butterfly-Species-using-AI-ML
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Then go to [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Model Information

* Model: `MobileNetV2`
* Input Size: `224x224`
* Accuracy: `90%+` on validation set
* Trained using: `TensorFlow/Keras`
* Dataset: [Butterfly Image Classification on Kaggle](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)

---

## 📊 Use Cases

### ✅ **Biodiversity Monitoring**

Helps researchers quickly classify butterflies in the field using mobile devices.

### ✅ **Ecological Research**

Tracks migration and behavior of butterfly species for scientific research.

### ✅ **Citizen Science & Education**

Allows students and nature enthusiasts to learn about butterflies interactively.

---

## 📦 Requirements

* Python 3.8+
* TensorFlow
* Pillow
* NumPy
* Streamlit

You can install them using:

```bash
pip install streamlit tensorflow pillow numpy
```

Or use:

```bash
pip install -r requirements.txt
```

## 🙋‍♂️ Author

👤 **Dama Ganesh**
🔗 [GitHub](https://github.com/DamaGanesh)
Mail ID:damaganesh4@gmail.com
---

## 🌱 Future Enhancements

* 🔄 Real-time mobile app integration
* 🌍 GPS tagging and map visualization
* 🗣️ Multilingual descriptions
* ☁️ Online deployment via Streamlit Cloud



