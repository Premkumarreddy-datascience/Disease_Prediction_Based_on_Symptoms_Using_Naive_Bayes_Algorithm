
# Disease Prediction Based on Symptoms using Naive Bayes

A machine learning project that predicts diseases based on input symptoms using the Naive Bayes algorithm. This tool aids in early diagnosis and medical assistance by analyzing symptoms and providing a probable condition.

---

## Table of Contents

- [About the Project](#-about-the-project)
- [Scope](#-scope)
- [Algorithm Used](#-algorithm-used)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Installation & Setup](#️-installation--setup)
- [Execution Steps](#-execution-steps)
- [Sample Output](#-sample-output)
- [Testing](#-testing)
- [Results](#-results)
- [Future Enhancements](#-future-enhancements)
- [Conclusion](#-conclusion)

---

## About the Project

This project leverages the **Naive Bayes classification algorithm** to predict potential diseases based on a user's symptoms. It helps healthcare professionals and patients by offering a fast and intelligent diagnosis tool built with Python and machine learning libraries.

---

## Scope

- Early diagnosis and preventive care.
- Assistance for healthcare providers.
- Reduced healthcare costs through timely predictions.
- Potential for public health monitoring and response.

---

## Algorithm Used

The system uses:
- ✅ **Naive Bayes Classifier** – for accurate probabilistic disease prediction.
- ✅ Compared with SVM and Random Forest classifiers for performance evaluation.

---

## Project Architecture

1. **Data Collection** – From Kaggle dataset.
2. **Preprocessing** – Handling null values, normalization.
3. **Training** – Naive Bayes algorithm on symptoms-disease mapping.
4. **Prediction** – Based on input symptoms from the user.
5. **Evaluation** – Using confusion matrix, accuracy score.

---

## Project Structure

```
Disease_Prediction_Project/
├── dataset/
│   └── Training.csv
├── models/
│   ├── NaiveBayesModel.pkl
│   ├── SVMClassifier.pkl
│   └── RandomForestClassifier.pkl
├── utils/
│   ├── preprocessing.py
│   └── evaluate.py
├── notebooks/
│   └── Disease_Prediction.ipynb
├── outputs/
│   └── confusion_matrix.png
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10.2
- Jupyter Notebook
- Pip packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn

### Installation

```bash
# Clone this repository
git clone https://github.com/your-username/Disease-Prediction-NaiveBayes.git
cd Disease-Prediction-NaiveBayes

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## Execution Steps

1. Launch Jupyter Notebook.
2. Open `Disease_Prediction.ipynb`.
3. Upload `Training.csv` in the correct folder.
4. Run all cells to train and test the model.
5. Input symptoms to receive disease predictions.

---

## Sample Output

```
Input: ['fever', 'fatigue', 'headache']
Predicted Disease: Typhoid
Confidence: 87.5%
```


---

## Testing

- Unit Testing – Model accuracy and input validation
- Integration Testing – Pipeline and algorithm connectivity
- System Testing – End-to-end checks with real data
- Evaluation – Accuracy, Sensitivity, Specificity

---

## Results

- Achieved **~90% accuracy** with Naive Bayes
- Naive Bayes outperformed Random Forest in speed and simplicity
- SVM performed well but required more tuning

---

## Future Enhancements

- Web-based front-end for live symptom input.
- Integration with genetic data for improved accuracy.
- Better visualization of prediction confidence and data analysis.
- Expand to deep learning models for rare disease prediction.

## Conclusion

This project successfully demonstrates how machine learning—specifically Naive Bayes—can aid in fast and accurate disease prediction. While promising, it should complement and not replace expert clinical diagnosis.

---
