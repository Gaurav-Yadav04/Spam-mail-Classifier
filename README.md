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

## 🎯 Live Demo
🚀 **Try it Yourself!**  
Enter any email text and instantly find out whether it’s **Spam** or **Not Spam**.  
🧠 The model predicts results in real-time.

🔗 **Live Demo:** [ https://huggingface.co/spaces/gaurav5005/spam-mail ]


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

## Wep
<img width="1505" height="316" alt="Screenshot 2025-10-23 030944" src="https://github.com/user-attachments/assets/e1dd348e-5cda-4f72-b26d-bf8b90b3dcb4" />
<img width="1544" height="319" alt="Screenshot 2025-10-23 030656" src="https://github.com/user-attachments/assets/a0a16a00-d0a0-4950-bdd0-3c2e983b2200" />

## 💡 How to Use
```python
import pickle

model = pickle.load(open("spam_mail_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

mail = ["You have won $1000! Click here to claim!"]
input_data = vectorizer.transform(mail)
prediction = model.predict(input_data)

print("✅ Ham Mail" if prediction[0]==1 else "🚫 Spam Mail")






