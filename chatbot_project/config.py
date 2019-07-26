global repo
global i
global model_file
global token_file
import os

path = os.path.abspath('') + '/sentiment_classification_model'
files = os.listdir(path)
model_file = files[0]
token_file = files[1]
i = int(model_file[6])
repo = ''

