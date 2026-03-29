# 🚀 AI Cyber Threat Detection System

This project is an **AI-based Network Intrusion Detection System (NIDS)** built using Python and scikit-learn.
It uses the **NSL-KDD dataset** to classify network traffic as **Normal** or **Attack**.

---

## 📌 Project Overview

* Detects cyber threats using Machine Learning
* Built with a clean **ML pipeline (preprocessing + model)**
* Supports real-time prediction via Streamlit UI
* Designed for **academic + resume + portfolio use**

---

## ⚙️ Features

* ✅ Uses **NSL-KDD benchmark dataset**
* ✅ Handles categorical features using `OneHotEncoder`
* ✅ Uses **RandomForestClassifier** with class balancing
* ✅ Applies **5-Fold Stratified Cross-Validation**
* ✅ Automatically selects best threshold using F1-score
* ✅ Provides:

  * Accuracy
  * Precision / Recall / F1-score
  * Confusion Matrix
  * Classification Report
* ✅ Interactive UI using **Streamlit**

---

## 📂 Project Structure

```
AI-Cyber-Threat-Detection-System/
│
├── main.py                # Model training & evaluation
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── .gitignore             # Ignored files
```

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Cyber-Threat-Detection-System.git
cd AI-Cyber-Threat-Detection-System
```

---

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

### 3. Download Dataset

Download:

* `KDDTrain+.txt`
* `KDDTest+.txt`

Place them in the project folder.

---

## ▶️ Run the Project

### 🔹 Train Model

```bash
python main.py
```

---

### 🔹 Run Web App

```bash
streamlit run app.py
```

---

## 📊 Sample Results

* Accuracy: ~80%
* Precision (Attack): ~0.97
* Recall (Attack): ~0.66
* F1-score: ~0.79

---

## 🧠 Model Details

* Algorithm: **Random Forest**
* Preprocessing: **ColumnTransformer + OneHotEncoder**
* Evaluation: **Stratified K-Fold Cross Validation**
* Metric Optimization: **F1-score based threshold tuning**

---

## 💡 Future Improvements

* 🔹 Deep Learning (LSTM / Autoencoders)
* 🔹 Real-time network packet capture
* 🔹 Deployment using Docker / Cloud
* 🔹 API integration using Flask

---

## 📌 Resume Title

**AI Cyber Threat Detection System using NSL-KDD Dataset**

---

## 👩‍💻 Author

**Yazhini Muthusamy**

---

## ⚠️ Note

* Dataset files are not included in this repository
* Model file (`.joblib`) will be generated after running `main.py`
