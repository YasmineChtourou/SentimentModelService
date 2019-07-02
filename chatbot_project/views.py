from django.http import HttpResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status 
import json
import pickle
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from rasa_nlu.model import Interpreter

from chatbot_project import preprocessing


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
model_file = os.path.abspath('sentimentmodelservice/model.sav')
token_file = os.path.abspath('sentimentmodelservice/tokenizer.sav')

@api_view(["POST"])
def prediction_sentiment(request):
    try:
        message=json.loads(request.body)
        ch = preprocessing.transformText(message["message"])

        interpreter = Interpreter.load(os.path.abspath('sentimentmodelservice/rasa_nlu_model'))
        intent=interpreter.parse(message["message"])

        model = pickle.load(open(model_file, 'rb'))
        token = pickle.load(open(token_file, 'rb'))
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

        return HttpResponse(json.dumps({"id":message["id"],"message":message["message"],"label":classe,"probability":str(probability[0][class_pred[0]]),"intent":intent['intent']['name']}), content_type='application/json')
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

