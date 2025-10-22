# 📧 Spam Mail Classifier using Logistic Regression

🚀 A simple yet powerful ML project that detects whether an email is **Spam (🚫)** or **Ham (✅)** using **Logistic Regression** and **TF-IDF Vectorizer**.

## ⚙️ Tech Used
- 🐍 Python  
- 📦 Scikit-learn  
- 🧠 Logistic Regression  
- 💬 TF-IDF Vectorization  
- 📊 Pandas, NumPy  

## 📈 Accuracy
✅ Training: 98.68%  
✅ Testing: 97.75%

## 🧩 Files
- `spam_mail_model.pkl` → Trained ML model  
- `vectorizer.pkl` → TF-IDF feature extractor  
- `spam_mail_py.ipynb` → Main code notebook  
- `spam_ham_dataset.csv` → Dataset used

 ## 🔮 Future Plans

  🌐 Web App (Flask / Streamlit)

  ☁️ Deploy on Render or Hugging Face

  🤖 Add deep learning models


##  👨‍💻 Author

Gaurav Yadav
C.S.E | ML Enthusiast | E-commerce Developer 🛍️
#MachineLearning #Python  #SpamDetection #AIProjects


## 💡 How to Use
```python
import pickle

model = pickle.load(open("spam_mail_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

mail = ["You have won $1000! Click here to claim!"]
input_data = vectorizer.transform(mail)
prediction = model.predict(input_data)

print("✅ Ham Mail" if prediction[0]==1 else "🚫 Spam Mail")



