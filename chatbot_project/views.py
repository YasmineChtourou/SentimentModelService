from django.http import HttpResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status 
import json
import pickle
import numpy as np
import pandas as pd
import os

from keras.preprocessing.sequence import pad_sequences
from rasa_nlu.model import Interpreter

import psycopg2
import subprocess

from chatbot_project import preprocessing
from chatbot_project import config

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000

rasamodel_file = 'rasa_nlu_model'
train_file = os.path.abspath('train_sentimentmodel.py')
DIR_DATA = os.path.abspath('Dataset/Pretrained_Data.csv')

@api_view(["POST"])
def prediction_sentiment(request):
    try:
        message=json.loads(request.body)
        ch = preprocessing.transformText(message["message"])

        interpreter = Interpreter.load(config.repo + rasamodel_file)
        intent=interpreter.parse(message["message"])

        model = pickle.load(open(config.repo + 'sentiment_classification_model/' + config.model_file, 'rb'))
        token = pickle.load(open(config.repo + 'sentiment_classification_model/' + config.token_file, 'rb'))
        x_input = np.array([ch])
        seq= token.texts_to_sequences(x_input)
        seqs = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        probability = model.predict(seqs)
        class_pred = model.predict_classes(seqs)
        if class_pred[0]==0 :
           classe='negative' 
        if class_pred[0]==1 :
           classe='neutre'
        if class_pred[0]==2 :
           classe='positive'

        #store in database 
        connection = psycopg2.connect("dbname='sentiment_context' user='postgres' password='Equipe1pfe' host='localhost' port='5433'")
        mark = connection.cursor()
        statement_prediction = 'INSERT INTO ' + 'prediction_sentiment_store' + ' (id,text,label,probability) VALUES (%s,%s,%s,%s)'
        statement_feedback = 'INSERT INTO ' + 'feedback_sentiment_store' + ' (id,text,label) VALUES (%s,%s,%s)'
        mark.execute(statement_prediction,(message["id"],message["message"],classe,str(probability[0][class_pred[0]])))
        mark.execute(statement_feedback,(message["id"],message["message"],classe))
        connection.commit()
        connection.close()
        mark.close()
      
        return HttpResponse(json.dumps({"i":config.i,"id":message["id"],"message":message["message"],"label":classe,"probability":str(probability[0][class_pred[0]]),"intent":intent['intent']['name']}), content_type='application/json')
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
 

@api_view(["POST"])
def store(request):
    try:
        body = json.loads(request.body) #feedback
        id = body["id"]
        text = body["text"]
        label = body["label"]
        
        #update
        connection = psycopg2.connect("dbname='sentiment_context' user='postgres' password='Equipe1pfe' host='localhost' port='5433'")
        mark = connection.cursor()
        statement_feedback = 'SELECT * FROM ' + 'feedback_sentiment_store' + ' WHERE id = %s'
        mark.execute(statement_feedback,[id])
        ligne = mark.fetchone()
        if ligne[2] != label :
             statement_feedback = 'Update ' + 'feedback_sentiment_store' + ' SET (label,validation) = (%s,%s) WHERE id = %s'
             mark.execute(statement_feedback,(label,0,id))
        connection.commit()
        connection.close()
        mark.close()

        return HttpResponse(json.dumps({"i":config.i}), content_type='application/json')
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["GET"])
def load_model(request):
    try:
        connection = psycopg2.connect("dbname='sentiment_context' user='postgres' password='Equipe1pfe' host='localhost' port='5433'")
        mark = connection.cursor()
        statement_count = 'SELECT count(*) FROM feedback_sentiment_store'
        mark.execute(statement_count)
        count = mark.fetchone()
        if (count[0] % 10) == 0:
             df = pd.read_csv(DIR_DATA, delimiter=';')
             row = 'SELECT text,label FROM feedback_sentiment_store ORDER BY id DESC LIMIT 10'
             mark.execute(row)
             row = mark.fetchone()
             while row is not None:
                  df.loc[len(df)]=[row[1],preprocessing.transformText(row[0])]
                  row = mark.fetchone()
             df.to_csv(DIR_DATA, sep=';', index=False)
             config.i = config.i + 1
             subprocess.run(['python ',train_file,str(config.i)]) #executer le script train
        os.remove(os.path.abspath('')+'/sentiment_classification_model/model_'+str(config.i-1)+'.sav')
        os.remove(os.path.abspath('')+'/sentiment_classification_model/tokenizer_'+str(config.i-1)+'.sav')
        config.files = os.listdir(config.path)
        config.model_file = config.files[0]
        config.token_file = config.files[1]
        return HttpResponse(json.dumps({"i":config.i}), content_type='application/json')
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)
 


 
