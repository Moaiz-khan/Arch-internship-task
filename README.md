
🛳️ Titanic Survival Prediction - Logistic Regression
Predict survival outcomes on the Titanic using machine learning (Logistic Regression). This is a beginner-friendly project that walks through the full data science workflow: EDA, data cleaning, model training, and evaluation — with visualizations and explanations.

📁 Dataset
Source: Titanic dataset via Seaborn (sns.load_dataset('titanic'))

Includes features like: pclass, sex, age, fare, and survival outcome (survived).

🧠 Objective
To build a classification model that predicts whether a passenger survived the Titanic disaster based on selected features.

🔍 Project Workflow
1. Data Loading
Loaded Titanic dataset using Seaborn.

2. Exploratory Data Analysis (EDA)
Visualized survival counts.

Analyzed survival by gender.

Identified and visualized missing data using heatmaps.

3. Data Cleaning & Preprocessing
Selected key features: pclass, sex, age, fare.

Filled missing age values using median.

Converted categorical column sex into numeric.

4. Model Building
Logistic Regression used for binary classification.

Dataset split into training and testing sets (80/20).

5. Model Evaluation
Accuracy: 80.4%

Evaluation metrics:

Confusion Matrix

Precision, Recall, F1-score

6. Visualizations
Heatmaps before and after cleaning

Survival counts

Gender vs Survival

Age distribution

Confusion matrix for model performance

🧪 Results
Metric	Score
Accuracy	80.4%
Precision (Survived)	78%
Recall (Survived)	73%
F1-Score (Survived)	76%

🛠️ Tools & Libraries
Python

Pandas, Seaborn, Matplotlib

Scikit-learn (LogisticRegression, train_test_split, metrics)

🚀 Future Improvements
Add more features: sibsp, parch, embarked

Try models like Decision Tree, Random Forest, or XGBoost

Use cross-validation

Apply class imbalance techniques (e.g., SMOTE)

📸 Sample Visualizations
Heatmap of missing values

Countplot of survival

Histogram of age

Confusion matrix for model evaluation

📂 Repository Structure
bash
Copy
Edit
titanic-survival-logistic/
│
├── titanic_logistic.ipynb        # Main notebook
├── README.md                     # Project overview
└── requirements.txt              # (Optional) Python dependencies
🙌 Credits
Dataset via Seaborn Titanic dataset

Inspired by Kaggle’s Titanic competition

