# 🎓 Student Placement Predictor

```
Student-Placement-Predictor/
│
├── app.py                        # Main Streamlit application
├── placement_model.pkl          # Trained Random Forest model
├── scaler.pkl                   # StandardScaler for feature scaling
├── encoders.pkl                 # LabelEncoders for categorical features
│
├── Student_Placement_Prediction.ipynb   # Model training notebook
├── preprocessed_dataset.csv             # Preprocessed training data
│
├── ai-placement-predictor.html          # Standalone HTML/CSS/JS frontend
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vikas-m-sharma/Student-Placement-Predictor.git
cd Student-Placement-Predictor
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

👉 The application will open at: [http://localhost:8501](https://code-with-soham.github.io/Student-Placement-Predictor/)

### 5️⃣ Alternative: Run HTML Version
Simply open `https://code-with-soham.github.io/Student-Placement-Predictor/` in any modern web browser.

---

## 📊 Model Performance

| Metric                | Value |
|----------------------|------|
| Accuracy             | 91.6% |
| Precision (Placed)   | 98% |
| Recall (Placed)      | 51% |
| F1-Score (Placed)    | 67% |

### 📌 Feature Importance
- Communication Skills – 26.7%  
- CGPA – 20.6%  
- Previous Semester Result – 16.6%  
- Projects Completed – 15.1%  
- College ID – 9.1%  
- Extra Curricular Score – 5.5%  
- Academic Performance – 4.9%  
- Internship Experience – 1.5%  

---

## 🎯 How It Works

### 🧾 Step 1: Enter Student Profile
- CGPA (4–10 scale)  
- Technical skills (Python, ML, Java, DSA, etc.)  
- Number of projects completed  
- Internship experience  
- Communication skills (1–10)  
- Backlogs  

### 🤖 Step 2: AI Analysis
- Uses Random Forest with 8+ features  
- Compares with 10,000+ historical records  
- Learns patterns from real placement data  

### 📈 Step 3: Get Results
- Placement probability + confidence score  
- Skill radar visualization  
- Personalized suggestions  
- Learning path recommendations  

### 💬 Step 4: Chatbot Support
- Resume tips  
- Interview preparation  
- Company insights  

---

## 🔧 API Configuration

To use the chatbot:
- Get API key from xAI  
- Enter key in chatbot panel  

⚠️ Note: Prediction works **without API key**.

---

## 📈 Dataset Information

| Feature                    | Description |
|--------------------------|------------|
| College ID               | Encoded college identifier |
| Previous Semester Result | Last semester CGPA |
| CGPA                     | Overall CGPA |
| Academic Performance     | Rating (1–10) |
| Internship Experience    | Yes/No |
| Extra Curricular Score   | Activity score |
| Communication Skills     | Self-rating |
| Projects Completed       | 0–5 |
| Placement                | Target (Placed/Not Placed) |

- Dataset Size: 10,000 records  
- Class Distribution:  
  - 83.4% Not Placed  
  - 16.6% Placed  

---

## 🚀 Future Improvements

- Add models (XGBoost, SVM, Neural Networks)  
- User authentication system  
- LinkedIn integration  
- Mock interviews (voice-based)  
- Mobile app version  
- Resume parser  
- Weekly progress tracking  
- Real-time placement updates  

---

## 🤝 Contributing

```bash
git checkout -b feature/AmazingFeature
git commit -m "Add some AmazingFeature"
git push origin feature/AmazingFeature
```

Then open a Pull Request.

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

- Dataset contributors  
- xAI (Grok API)  
- Streamlit  
- Scikit-learn  
- FontAwesome & Google Fonts  

---

## 📧 Contact

**Soham Kundu**

- GitHub: (https://github.com/code-with-soham) 
- LinkedIn: https://www.linkedin.com/in/soham-kundu-b5a9a0250/ 
- Email: sohamkundu84@gmail.com  

---

## 🌟 Show Your Support

If helpful, ⭐ the repo and share!

---

## 📸 Screenshots

- Prediction Page  
- Dashboard Analytics  
- AI Chatbot  

---

## 📖 Citation

```bibtex
@misc{placement-predictor-2026,
  author = {Soham Kundu},
  title = {Student Placement Predictor: AI-Powered Placement Prediction System},
  year = {2026},
  publisher = {GitHub},
  url = https://code-with-soham.github.io/Student-Placement-Predictor/
}
```

---

## 📊 Model Training Details

### Preprocessing
- Label Encoding  
- Standard Scaling  
- Class imbalance handling  

### Hyperparameters
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)
```

### Results
- Training Accuracy: 98.7%  
- Test Accuracy: 91.6%  
- Cross-validation: 89.3% ± 2.1%  

---

## 🛠️ Troubleshooting

**Q: Chatbot not working?**  
→ Check API key  

**Q: Predictions inaccurate?**  
→ Use as guidance, not absolute  

**Q: Streamlit not running?**  
→ Install dependencies, Python 3.8+  

**Q: HTML version not predicting?**  
→ It’s only a frontend demo  

---

## ⚡ Performance Tips

- Use HTML version for fast UI  
- Deploy on Streamlit Cloud/VPS  
- Cache model for speed  

---

💡 Built By Soham ❤️   
