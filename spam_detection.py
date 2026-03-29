#spam_detection system
import numpy as np
import pandas as pd
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

print("Path to dataset files:", path)

data = pd.read_csv(os.path.join(path,os.listdir(path)[0]),encoding="latin-1")

data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

print(data.head())

data["v1"] = data["v1"].map({"ham":0,"spam":1})

data["v2"] = [text.lower() for text in data["v2"]]

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

data["v2"] = [remove_punctuation(text) for text in data["v2"]]

def remove_digits(text):
  return re.sub(r'\d+', '', text)

data["v2"] = [remove_digits(text) for text in data["v2"]]

data["v2"] = [text.split() for text in data["v2"]]

def remove_stopwords(text):
  stop_words = list(stopwords.words('english'))
  scanned = []
  for word in text:
    if word not in stop_words:
      scanned.append(word)
  return scanned

data["v2"] = [remove_stopwords(text) for text in data["v2"]]

def lematization(text):
  lemmatizer = WordNetLemmatizer()
  lemmatized = []
  for word in text:
    lemmatized.append(lemmatizer.lemmatize(word))
  return lemmatized

data["v2"] = [lematization(text) for text in data["v2"]]

data["v2"] = [str(" ".join(text)) for text in data["v2"]]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(data["v2"])
y = data["v1"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = MultinomialNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy*100}%")
