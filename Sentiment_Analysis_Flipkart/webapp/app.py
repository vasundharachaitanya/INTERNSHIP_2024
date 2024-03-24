# from flask import Flask, request, render_template
# #from streamlit.report_thread import stop_thread 

# import numpy as np
# import pickle
# import streamlit as st
# from PIL import Image

# # loading the saved model
# loaded_model = pickle.load(open(r'C:\Users\sss\Desktop\ai-elite-batch-10\Sentiment_Analysis_Flipkart\webapp\lr_model.pkl', 'rb'))
# #loaded_vectorizer = pickle.load(open(r'C:\Users\sss\Desktop\ai-elite-batch-10\Sentiment_Analysis_Flipkart\webapp\BOW_MODEL.pkl', 'rb')) 


# def text_review(input_data):
#     new_X = loaded_vectorizer.transform(input_data)
#     prediction = loaded_model.predict(new_X)
#     return prediction


# app = Flask(__name__) 


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     input_data = [request.form['reviewtext']]
#     score = text_review(input_data)
#     return render_template('result.html', score=score)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



from flask import Flask, request, render_template
import numpy as np
#import pickle
import streamlit as st
from PIL import Image
from joblib import load
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# loading the saved model
loaded_model = load("lr_model.pkl")

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def text_preprocess(raw_text, flag):
    # Removing special characters and digits
    sentence = re.sub("[^a-zA-Z]", " ", raw_text)

    # Change sentence to lowercase
    sentence = sentence.lower()

    # Tokenize into words
    tokens = sentence.split()

    # Remove stop words
    clean_tokens = [t for t in tokens if not t in stopwords.words("english")]

    # Stemming/Lemmatization
    if flag == 'stem':
        clean_tokens = [stemmer.stem(word) for word in clean_tokens]
    else:
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]

    return " ".join(clean_tokens)

def text_review(input_data):
    # Preprocess the input text
    preprocessed_data = text_preprocess(input_data, flag='lem')  # Assuming lemmatization is used
    new_X = [preprocessed_data]
    prediction = loaded_model.predict(new_X)
    return prediction


app = Flask(__name__) 


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['reviewtext']
    score = text_review(input_data)
    return render_template('result.html', score=score)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
