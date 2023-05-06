import streamlit as st
import pickle
import nltk
import string
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from flask import Flask
app=Flask(__name__)

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear();

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms=st.text_input("Enter the message")

#preprocess
transformed_sms=transform_text(input_sms)
#vectorise
vector_input=tfidf.transform([transformed_sms])
#predict
result=model.predict(vector_input)[0]
#dispaly
if result==1:
  st.header("Spam")
else:
  st.header("ham")


if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
