from django.shortcuts import render
from django.http import HttpResponse,Http404

from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

import json
import pickle
import numpy as np
import re
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from rasa_nlu.model import Interpreter

import nltk 
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import parsing
from gensim.parsing.preprocessing import split_alphanum
from spellchecker import SpellChecker
from chatbot_project import preprocessing

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
filename = os.path.abspath('chatbot_project/model.sav')
file = os.path.abspath('chatbot_project/tokenizer.sav')

@api_view(["POST"])
def prediction_sentiment(text):
    try:
        msg=json.loads(text.body)
        ch=msg["message"]
        ch = ch.lower()
        ch  = preprocessing.replace_word(ch)
        ch  = preprocessing.normaliser_word(ch)
        ch= re.sub(r'[^\x00-\x7f]',r' ',ch)
        stops={'at', 'only', 'your', 'yourself', 'a', 'i', 'during', 'off', 'myself', 'so', 'o', 'after', 'under', 
           'there', 'against', 'over', 'ourselves', 'they', 'me', 'its', 'then', 'above', 'theirs', 'this', 'into', 
           'from', 'very', 'on', 'yours', 'yourselves', 'herself', 'themselves', 'between', 'if', 'below', 'own', 
           'and', 'you', 'itself', 'him', 'while', 's', 'who', 'we', 'what', 'by', 'ma', 'further', 'such', 'until',
           'through', 'too', 'until', 'through', 't', 'too', 'where', 'up', 'my', 'm', 'out', 'down', 're', 'to', 
           'she', 'd', 'those', 'when', 'it', 'because', 'he', 'in', 'other','each', 'both', 'her', 'but', 'as', 'all', 
           'his', 'again', 'with', 'once', 'am', 'just', 'should', 'why', 'than', 'any', 'should', 'why', 'than',
           'more', 'most', 'that', 've', 'will', 'ours', 'our', 'll', 'the', 'y', 'which', 'whom', 'hers', 'an', 'here',
           'how', 'before', 'about', 'for', 'them', 'these', 'their', 'for', 'them', 'these', 'their', 'or', 'must', 
           'shall', 'would', 'could' , 'need', 'might'}
        filtered_words = [word for word in ch.split() if word not in stops]
        filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
        ch = " ".join(filtered_words)
        ch = gensim.parsing.preprocessing.strip_punctuation2(ch)
        spell = SpellChecker()
        misspelled = ch.split()
        wordnet_lemmatizer = WordNetLemmatizer()
        for i in range(len(misspelled)):
             word = spell.correction(misspelled[i])
             misspelled[i]=word
             misspelled[i] = wordnet_lemmatizer.lemmatize(misspelled[i], pos="v")
             misspelled[i] = wordnet_lemmatizer.lemmatize(misspelled[i], pos="n")
        ch = " ".join(misspelled)
        filtered_words = [word for word in ch.split() if word not in stops]
        ch = " ".join(filtered_words)
        ch = gensim.corpora.textcorpus.strip_multiple_whitespaces(ch)

        interpreter = Interpreter.load(os.path.abspath('chatbot_project/current'))
        intent=interpreter.parse(ch)
        loaded_model = pickle.load(open(filename, 'rb'))
        token = pickle.load(open(file, 'rb'))
        x_input = np.array([ch])
        seq= token.texts_to_sequences(x_input)
        seqs = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        probability = loaded_model.predict(seqs)
        class_pred = loaded_model.predict_classes(seqs)
        K.clear_session()
        prob=str(probability[0][class_pred[0]])
        if class_pred[0]==0 :
           classe='negative' 
        if class_pred[0]==1 :
           classe='neutre'
        if class_pred[0]==2 :
           classe='positive'
        return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":classe,"probability":prob,"intent":intent['intent']['name']}), content_type='application/json')
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

