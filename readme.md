# 🍷 Wine Quality Prediction - Group 1  

## 📌 Project Overview  

Wine quality prediction is an essential task in the wine industry, influencing pricing, branding, and consumer satisfaction. This project leverages **Machine Learning (ML)** to predict wine quality based on physicochemical properties using the **UCI Wine Quality Dataset**.  

We employed multiple ML models, conducted statistical analyses, and applied feature engineering techniques to build an accurate and robust classification model.  

---
## EXECUTIVE BRIEF SUMMARY
In our project, we aimed to predict wine quality based on chemical properties using multiple machine learning models. by analying the UCI Wine Quality dataset, we trained and evaluated models such as RANDOM Forest, KNN, Decision Tree, Logistic Regression, SVM, and Naive Bayes to determind the best perfoming algorithm. The goal is to identify the key factors influencing wine quality and develp an accurate classification model.
---
## 📊 Dataset Information  

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)  

This dataset contains chemical properties of **red and white wine** samples, with **quality ratings (0-10)** assigned by wine tasters.  

### **Features Used:**  
- **Fixed Acidity**  
- **Volatile Acidity**  
- **Citric Acid**  
- **Residual Sugar**  
- **Chlorides**  
- **Free Sulfur Dioxide**  
- **Total Sulfur Dioxide**  
- **Density**  
- **pH**  
- **Sulphates**  
- **Alcohol**  
- **Quality (Target Variable: 0-10 scale)**  

### **Objective:**  
✅ Predict wine quality (**Low, Medium, High**) based on chemical properties.  
✅ Determine which features impact wine quality the most.  

---
### External Libraries and thier Purpose:
- pandas → Handles data (loading, processing, and analyzing).
- matplotlib.pyplot → Used for data visualization.
- numpy → Handles numerical operations.
- train_test_split → Splits dataset into training and testing sets.
- StandardScaler → Standardizes numerical data for better model performance.
- balanced_accuracy_score, accuracy_score → Used to evaluate model performance.
- SVC (Support Vector Classifier) → Implements the SVM model.
- statsmodels, scipy.stats → Used for statistical analysis and hypothesis testing.
- seaborn → Enhances visualization.

---
Tools and Technologies Employed 
- git 
- github
- Powerpoint
- Jupytor Notebook
- Slack
---
## 🔬 Data Preprocessing & Exploration  

### **📌 Steps Taken:**  

#### **1️⃣ Data Cleaning**  
✅ Removed unnecessary columns  
✅ Standardized column names for consistency  
✅ Handled missing values  

#### **2️⃣ Exploratory Data Analysis (EDA)**  
📊 Visualized correlations between chemical properties & wine quality   
📊 Calculated **Pearson Correlation** between **Density & Residual Sugar**  

#### **3️⃣ Feature Engineering**  
✅ Categorized wine quality into three classes:  
   - **Low Quality** (3-5)  
   - **Medium Quality** (6-7)  
   - **High Quality** (8-9)  

✅ Created a **custom function** to process Red & White wines separately  
✅ Applied **One-Hot Encoding** for categorical variables  

#### **4️⃣ Handling Class Imbalance**  
✅ Used **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset  

---

## 🤖 Model Training & Evaluation  

We implemented multiple machine learning models, fine-tuned hyperparameters, and compared their performance.  

### **Model Performance Summary:**  

| Model                   | Accuracy (%) |
|-------------------------|-------------|
| Logistic Regression     | 69.6%       |
| Decision Tree          | 66.0%       |
| K-Nearest Neighbors    | 66.7%       |
| Naive Bayes            | 74.5%       |
| **Random Forest**      | **🔥 80.6% ✅** |

---

## 🏆 Best Performing Model: **Random Forest**  

### **Why Random Forest?**  
✔️ Handles **Noisy & Complex Data** Better  
✔️ Reduces **Overfitting** by Combining Multiple Decision Trees  
✔️ Automatically **Ranks Feature Importance**  
✔️ Performs Well on **Small & Large Datasets**  
✔️ Maintains **High Accuracy Without Scaling**  

### **🔹 Key Feature Importances Identified by Random Forest:**  
🍷 **Alcohol content** is the **strongest predictor** of wine quality  
⚠️ **Higher Total Sulfur Dioxide** reduces wine quality  
📈 **Residual Sugar & Density** are highly correlated in white wines  

---

## 📈 Key Insights & Findings  

🔹 **Alcohol** has a **strong positive correlation** with wine quality  
🔹 **Sulfur dioxide levels** impact taste & preservation, affecting overall quality  
🔹 **Decision Trees & Random Forest models** effectively identify feature importance  
🔹 **Feature engineering & data cleaning** significantly improved model accuracy  

---

## 🚀 Future Work  

🔹 Explore **Deep Learning Models (Neural Networks)** for better classification  
🔹 Use **additional datasets** for better generalization  
🔹 Optimize hyperparameters using **Bayesian Optimization**  
🔹 Deploy the best model as a **Web API** for real-world testing  

---

## 👨‍💻 Contributors  

This project was developed by **Group 1**:  

👑 **Frank "Iron Fist" Tsibu**  
💥 **David "The Juggernaut" Clark**  
⚰️ **Laurie "The Reaper" Webb**  
🎭 **Michael "Just Mike" Garner**  

---

## 📜 License  

This project is licensed under the **MIT License**. See the `LICENSE` file for more details.  

---

## 📢 Final Thoughts  

This project showcases how **machine learning** can **revolutionize wine quality assessment**.  
By leveraging **Random Forest**, **feature engineering**, and **statistical analysis**, we achieved an impressive **80.6% accuracy!**  

With further optimizations and deployment, this model could be used in the **wine industry** to assess quality **efficiently and accurately**.  

🍷 **Cheers to Data Science!** 🎉  

---

## 🔗 References  

🔹 [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)  
🔹 Machine Learning Algorithms - **Scikit-learn**  
🔹 Shrikant Temburwar - **Wine Quality Dataset**  
