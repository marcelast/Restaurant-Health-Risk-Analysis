import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


st.title('Analiza Riscurilor Sănătății Publice în Restaurante pe Baza Recenziilor')

# Încărcarea setului de date
file_path = "restaurantsReviews.csv"  # Încărcați fișierul corect
data = pd.read_csv(file_path)

# Maparea locationId cu numele restaurantelor
location_mapping = {
    3643106: "La Sarkis",
    9593555: "Zaxi Fun & Finest",
    12794044: "Saperavi",
    23890224: "Little Napoli",
    1089531: "La Placinte",
    9729405: "OSHO bar&kitchen",
    2261775: "Pegas Restaurant & Terrace",
    21041974: "Fuior",
    23474553: "Divus Restaurant",
    25270571: "Charmat Prosecco Bar"
}

data['restaurant_name'] = data['locationId'].map(location_mapping)

# Prezentarea datelor
st.subheader('Vizualizare Date - Primele 5 Linii')
st.write(data.head())

# Statistici
st.subheader('Date Statistice')
st.write(f"Număr total de recenzii: {data.shape[0]}")
st.write("Valori lipsă pe coloană:")
st.write(data.isnull().sum())
st.write("Statistici descriptive pentru coloanele numerice:")
st.write(data.describe())

# Vizualizarea distribuției rating-urilor
st.subheader('Distribuția Rating-urilor')
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=data, x='rating', palette='viridis')
ax.set_title("Distribuția Rating-urilor")
ax.set_xlabel("Rating")
ax.set_ylabel("Număr de Recenzii")
st.pyplot(fig)

# preprocesarea textului
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['processed_text'] = data['text'].apply(preprocess_text)

# Funcția pentru analiza sentimentului
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

data['sentiment'] = data['processed_text'].apply(get_sentiment)

# Crearea etichetei de risc pe baza sentimentului
def assign_risk(sentiment):
    if sentiment > 0.2:
        return 'low'
    elif sentiment < -0.2:
        return 'high'
    else:
        return 'medium'

data['risk'] = data['sentiment'].apply(assign_risk)

# Antrenarea modelului
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['processed_text'])
y = data['risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluarea modelului
st.subheader('Evaluarea Modelului')
st.write(classification_report(y_test, y_pred))

# Vizualizarea distribuției riscurilor prezise
st.subheader('Distribuția Riscurilor Prezise')
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=data, x='risk', palette='viridis', hue='risk', dodge=False)
ax.set_title("Distribuția Riscurilor Prezise")
ax.set_xlabel("Nivel de Risc")
ax.set_ylabel("Număr de Recenzii")
st.pyplot(fig)

# Crearea Word Cloud pentru recenziile cu risc înalt și scăzut
high_risk_reviews = data[data['risk'] == 'high']['text']
low_risk_reviews = data[data['risk'] == 'low']['text']

high_risk_text = ' '.join(high_risk_reviews)
low_risk_text = ' '.join(low_risk_reviews)

high_risk_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(high_risk_text)
low_risk_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(low_risk_text)

st.subheader('Word Clouds - Risc Înalt vs Risc Scăzut')
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Word cloud pentru risc înalt
ax[0].imshow(high_risk_wordcloud, interpolation='bilinear')
ax[0].set_title("Risc Înalt")
ax[0].axis('off')

# Word cloud pentru risc scăzut
ax[1].imshow(low_risk_wordcloud, interpolation='bilinear')
ax[1].set_title("Risc Scăzut")
ax[1].axis('off')

st.pyplot(fig)

# Creăm un grafic barplot pentru a vizualiza restaurantele și nivelul de risc
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='restaurant_name', hue='risk', palette='coolwarm')

# Setăm titlul și etichetele axelor
plt.title('Nivelul de risc al restaurantelor', fontsize=16)
plt.xlabel('Restaurant', fontsize=14)
plt.ylabel('Număr de recenzii', fontsize=14)

# Rotim etichetele de pe axa X pentru a fi mai lizibile
plt.xticks(rotation=90)

# Afișăm graficul în Streamlit
st.pyplot(plt)
