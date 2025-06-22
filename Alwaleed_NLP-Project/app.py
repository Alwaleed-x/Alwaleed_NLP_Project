import pandas as pd
import re
import nltk
import gradio as gr
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df = pd.read_csv('train.csv', header=None)
df.columns = ['label', 'title', 'description']
df['text'] = df['title'] + ' ' + df['description']
df = df[['text', 'label']]
df = df[df['label'] != 0]
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

label_html = {
    1: '<div style="display:flex;align-items:center;"><img src="https://cdn-icons-png.flaticon.com/512/219/219983.png" width="25" style="margin-right:8px;">World</div>',
    2: '<div style="display:flex;align-items:center;"><img src="https://cdn-icons-png.flaticon.com/512/905/905568.png" width="25" style="margin-right:8px;">Sports</div>',
    3: '<div style="display:flex;align-items:center;"><img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="25" style="margin-right:8px;">Business</div>',
    4: '<div style="display:flex;align-items:center;"><img src="https://cdn-icons-png.flaticon.com/512/2965/2965567.png" width="25" style="margin-right:8px;">Science & Tech</div>'
}

def predict_topic_display(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    print(f"Input: {text}")
    print(f"Predicted label: {pred}")
    return label_html.get(int(pred), '<div style="color:red;">❌ Unknown Category</div>')

gr.Interface(
    fn=predict_topic_display,
    inputs=gr.Textbox(lines=4, placeholder="Enter a news headline or paragraph...", label="News Text"),
    outputs=gr.HTML(label="Predicted Topic"),
    title="News Topic Classifier",
    description="Enter a short news article or title to identify its category.",
    theme="default"
).launch()