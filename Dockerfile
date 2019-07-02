FROM python:3.6-stretch

# The enviroment variable ensures that the python output is set straight to the terminal with out buffering it first

ENV PYTHONUNBUFFERED 1

ADD . /sentimentmodelservice

RUN apt-get update && apt-get update && apt-get install -y gcc build-essential autoconf cmake libtool git python-dev apt-utils python-pip libpython-dev

RUN pip install awscli

RUN pip install --upgrade pip

RUN pip install -r sentimentmodelservice/requirements.txt

RUN [ "python", "-c", "import nltk; nltk.download('wordnet')" ]

RUN python -m spacy download en

EXPOSE 80

CMD ["python", "sentimentmodelservice/manage.py", "migrate", "0.0.0.0:80"]

CMD ["python", "sentimentmodelservice/manage.py", "runserver", "0.0.0.0:80"]