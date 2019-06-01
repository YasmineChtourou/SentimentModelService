# SentimentModelService

This is a web service created with django Framework to consume a sentiment model classification and a rasa nlu model
   * The sentiment model classification classify short texts into positive, negative and neutral sentiment
   * The rasa nlu model classify short texts into many intents
   
   
## Install dependencies

To get a development environment running you should :

   * Install virtualenv  :

``` pip install virtualenv ```

   * Create a new virtual environment and easily install all libraries by running the following command :

``` conda create  --name venv_name  --file requirements.txt ```

In the file requirements.txt you find all necessary dependencies for this project.

   * To activate the new environment:

``` source activate  venv_name ```


## Running the tests
 
To run this project use this command:
```
python manage.py runserver
```