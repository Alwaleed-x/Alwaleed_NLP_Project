# News Topic Classification using NLP

This project implements a Natural Language Processing (NLP) app that classifies short news texts into categories like World, Sports, Business, and Science & Tech using a Logistic Regression model.

---

## 1. Requirements

Before running the app, make sure you have Python installed, then install the following dependencies:

```bash
pip install gradio pandas scikit-learn nltk
2. How to Run
To launch the application locally:

bash
نسخ
تحرير
python app.py
This will open an interactive Gradio interface in your browser to test the model in real time.

3. Files Included
app.py: Gradio-based application code with real-time prediction interface.

train.csv: Dataset used to train the model (includes title, description, and label).

Alwaleed_NLP_Project.ipynb: Colab notebook for full data preprocessing, model training, and evaluation.

ScreenShot1.png, Screenshot2.png: Preview images of the app interface.

4. Tools Used
Language: Python

Libraries: Scikit-learn, Pandas, NLTK, Gradio

Platforms: Google Colab (for model development), VS Code (for deployment)

Model: Logistic Regression

Preprocessing: Text cleaning, stopword removal, TF-IDF vectorization

5. Model Evaluation
The model was trained on clean labeled data and evaluated using a test split (80/20). Performance highlights:

Accuracy: ~91.5%

Precision, Recall, F1-score: Between 89%–97% across categories

Evaluation tools used: classification_report, confusion_matrix, and visualization via seaborn

Invalid or mislabeled rows (e.g. label = 0) were excluded during preprocessing for cleaner generalization.

6. Generalization
The model demonstrates strong generalization capability, achieving high performance on unseen test data. It avoids overfitting and works well across categories with balanced F1-scores.

7. Interface Preview
You can find preview images of the UI interface below:



8. Summary
This NLP application allows users to classify news text based on title and description using classical machine learning. It demonstrates:

End-to-end text processing

Machine learning pipeline

Model evaluation and visualization

Clean and simple UI with Gradio
