
---

## 📌 Problem Statement

Despite having a large dataset, dropping missing values disproportionately affected SSD records, leading to biased model performance. As a result, **two separate modeling pipelines** were developed:
- One for **HDDs**, with more complete data
- One for **SSDs**, using mode/missing value imputation and simpler models

---

## 🔄 Preprocessing Steps

- Features with **more than 50% missing values** were dropped.
- For binary/categorical flags (e.g., `PREDICTIVE_FAILURE_ANALYSIS`, `DEVICE_NOT_READY`), **mode imputation** was applied.
- The dataset was split into SSD and HDD groups to prevent overfitting and ensure model fairness.

---

## 🔬 Feature Selection

- Pearson correlation was computed between numeric features and the `WILL_FAIL` target.
- Top 10 correlated features were visualized for both HDD and SSD drives.
- RFE (Recursive Feature Elimination) was used to select the most predictive features.

---

## 🤖 Models Trained

| Model                   | Drive Type | Accuracy | Precision | Recall | F1-Score |
|------------------------|------------|----------|-----------|--------|----------|
| Logistic Regression     | HDD        | 0.88     | 0.60      | 0.89   | 0.71     |
| SVM                     | SSD        | 0.82     | 0.43      | 1.00   | 0.60     |
| Random Forest           | HDD        | 0.91     | 0.65      | 0.95   | 0.77     |
| XGBoost                 | HDD        | 0.91     | 0.67      | 0.86   | 0.76     |
| Tuned XGBoost           | SSD        | 1.00     | 1.00      | 1.00   | 1.00     |
| Tuned Random Forest     | SSD        | 1.00     | 1.00      | 1.00   | 1.00     |

> **Note**: While tuned models show perfect scores on SSD data, this may be due to **overfitting**, as SSD data is much smaller. Therefore, simpler models like Logistic Regression and SVM are preferred for SSDs to ensure generalization.

---

## Key Findings

- **HDDs** show strong predictive patterns, and Random Forest/XGBoost are effective.
- **SSDs** are more sparse and may suffer from overfitting; hence, **simpler models** generalize better.
- Splitting the dataset by drive type significantly improved the reliability of predictions.

---

## Future Improvements

- Collect more SSD failure examples to balance datasets.
- Use time series features (e.g., trend in temperature) for dynamic prediction.
- Explore anomaly detection techniques for rare failure types.

---

## Technologies Used

- Python (Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- Jupyter Notebook
- Recursive Feature Elimination (RFE)
- SMOTE for class balancing

---

## Authors

- Mahsa Dorri (Data Science Certificate, University of Toronto)
- (https://github.com/MahsaDorri)



