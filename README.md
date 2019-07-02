# SentimentModelService

This is a web service created with django Framework to consume a sentiment model classification and a rasa nlu model
   * The sentiment model classification classify short texts into positive, negative and neutral sentiment
   * The rasa nlu model classify short texts into many intents
   
   
## Install dependencies


   * Create a new virtual environment and easily install all libraries by running the following command :

``` virtualenv venv ```

``` source venv/bin/activate (Under windows run $ venv/Scripts/activate.bat) ```

``` pip install -r requirements.txt ```

In the file requirements.txt you find all necessary dependencies for this project.

   * To activate the new environment:

``` source activate  venv_name ```


## Running the tests
 
To run this project use this command:
```
python manage.py runserver
```


## Building Docker image and running a container

To build an image from docker file:
```
docker build --tag=image_name .
```
To run a container from docker image:
```
docker run -p 80:80 image_name
```