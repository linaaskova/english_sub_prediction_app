import streamlit as st
from io import StringIO
import pandas as pd
import pysrt
import spacy
import re
from spacy.tokenizer import Tokenizer 
from collections import defaultdict
import pickle
from PIL import Image

st.title(':blue[English level prediction app]') 

st.subheader('Данное приложение определяет уровень английского языка на основании субтитров фильма') 

st.subheader('Вам необходимо загрузить субтитры ниже в формате srt') 

image = Image.open('D:/english_sub/lg_6343b9933faa6327b9116d05a24cd8b7-lgX2.jpg')
st.image(image)

def load_model():
    with open('D:/english_sub/main.pcl','rb') as fid:
        return pickle.load(fid)
    


file = st.file_uploader('Загрузка файла с субтитрами в формате srt', type = 'srt')
    
if st.button('Уровень английского'):
    subs = []
    encodings = ['', 'UTF-8-SIG', 'ISO-8859-1', 'utf-8', 'Windows-1252', 'ascii']
    encoding_number = 0

    while not subs:
        try:
            string = StringIO(file.getvalue().decode(encodings[encoding_number]))
            subs = string.getvalue()
        except LookupError:
            encoding_number += 1

    results = defaultdict(list)
    results['subtitles'].append(subs)
    data = pd.DataFrame(results)
    
    HTML = r'<.*?>' # html тэги меняем на пробел
    TAG = r'{.*?}' # тэги меняем на пробел
    COMMENTS = r'[\(\[][A-Za-z ]+[\)\]]' # комменты в скобках меняем на пробел
    UPPER = r'[[A-Za-z ]+[\:\]]' # указания на того кто говорит (BOBBY:)
    LETTERS = r'[^a-zA-Z\'.,!? ]' # все что не буквы меняем на пробел 
    SPACES = r'([ ])\1+' # повторяющиеся пробелы меняем на один пробел
    DOTS = r'[\.]+' # многоточие меняем на точку
    SYMB = r"[^\w\d'\s]" # знаки препинания кроме апострофа

    nlp = spacy.load('en_core_web_sm',exclude=["tok2vec", "parser", "ner", "attrbute_ruler"])
    stopwords = nlp.Defaults.stop_words
    tokenizer = Tokenizer(nlp.vocab)

    def clean_subs(subs):
        subs = subs[1:] # удаляем первый рекламный субтитр
        txt = re.sub(HTML, ' ', subs) # html тэги меняем на пробел
        txt = re.sub(COMMENTS, ' ', txt) # комменты в скобках меняем на пробел
        txt = re.sub(UPPER, ' ', txt) # указания на того кто говорит (BOBBY:)
        txt = re.sub(LETTERS, ' ', txt) # все что не буквы меняем на пробел
        txt = re.sub(DOTS, r'.', txt) # многоточие меняем на точку
        txt = re.sub(SPACES, r'\1', txt) # повторяющиеся пробелы меняем на один пробел
        txt = re.sub(SYMB, '', txt) # знаки препинания кроме апострофа на пустую строку
        txt = re.sub('www', '', txt) # кое-где остаётся www, то же меняем на пустую строку
        txt = txt.lstrip() # обрезка пробелов слева
        txt = txt.encode('ascii', 'ignore').decode() # удаляем все что не ascii символы   
        txt = txt.lower() # текст в нижний регистр
        return txt

    def token_drop_stop(subs):
            
        new_tokens = tokenizer(subs)
        drop_stops = [w for w in new_tokens if w not in stopwords]  
        res_text = ''.join([token.lemma_ for token in nlp(str(drop_stops))])
        return res_text
    
    data['subtitles'] = data['subtitles'].apply(lambda row: clean_subs(row)).apply(lambda row: token_drop_stop(row))
    model = load_model()
    y = model.predict(data['subtitles'])

    st.markdown(f'Уровень английского языка на основании данного фильма: {y[0]}')
    
