# 🌦️ Weather Data Preprocessing and Classification using Gaussian Naive Bayes in Python

## 📌 Overview
This project demonstrates an end-to-end machine learning pipeline on an **Australian weather dataset**, involving:
- 🔍 Data cleaning & missing value treatment
- 🧹 Feature reduction to drop sparse columns
- 🤖 Building a **Gaussian Naive Bayes classifier** for predictive analysis

The workflow helps showcase how to handle real-world datasets that often contain missing values and unnecessary features.

---

## 🚀 Features
✅ Load and inspect weather data  
✅ Handle missing values with statistical imputation  
✅ Drop columns with excessive missing data  
✅ Build & train a **Gaussian Naive Bayes** classification model  
✅ Easy to extend for regression or other classifiers

---

## 🛠 Tech Stack
- **Python** (pandas, numpy, scikit-learn)
- Jupyter Notebook / Google Colab for analysis & visualization

---

## 📂 Dataset
- **File:** `weatherAUS.csv`
- Contains daily weather observations from numerous Australian weather stations.
- Includes variables like `MinTemp`, `MaxTemp`, `Rainfall`, `Humidity`, `WindSpeed`, etc.

---

## ⚙️ Installation

Install the required Python packages using:

```bash
pip install pandas numpy scikit-learn
````

---

## 💻 Usage

### 1️⃣ Load and explore the dataset

```python
import pandas as pd
import numpy as np

data = pd.read_csv('weatherAUS.csv')
print(data.head())
print(data.info())
```

### 2️⃣ Handle missing values

* Fill missing `MinTemp` values with mean.
* Drop columns with >15,000 missing entries.

```python
data['MinTemp'].fillna(data['MinTemp'].mean(), inplace=True)
data = data.dropna(axis=1, thresh=len(data)-15000)
```

---

## 🔍 Model Building

### 3️⃣ Gaussian Naive Bayes Classification

```python
from sklearn.naive_bayes import GaussianNB

# Assuming you have already prepared x_train, y_train
model = GaussianNB()
model.fit(x_train, y_train)
```

---

## 📈 Possible Next Steps

* Tune hyperparameters or try alternative classifiers (e.g., Decision Trees, Random Forest).
* Add evaluation metrics like accuracy, precision, recall.
* Visualize confusion matrix for model diagnostics.

---

## 📝 Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

---

## 🌟 Acknowledgements

* Thanks to the creators of the dataset and the Python open-source community for tools like pandas, numpy, and scikit-learn.

---

## 🚀 Author

👋 Created by **[Balachandharsriram M](https://github.com/balachandharsriram)**.
