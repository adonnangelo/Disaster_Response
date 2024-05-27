# Disaster Response

## Project Description
This project explores a dataset consisting of prelabelled messages and tweets from disaster events. These messages have been labelled with 36 categories that highlight their message type and summarize thier intent. The goal of this project is to use these labelled messages to train a NLP model which can then segment messages from future events into respecitive buckets. A model such as this can help search and rescue and aid personell get help to where its needed most. 

## Installation
The code in this project was written with Python in VS Code. The following libraries will be required to run the scripts:
Pandas, Numpy, sys, os, sqlalchemy, nltk (with punkt), sklearn, pickle, json, plotly, flask, and joblib. 

## File Descriptions
There are several files located in this repository:
The data folder holds the main dataset for the project

-categories.csv

-messages.csv

The above csvs are merged, cleaned, and loaded into the DisasterRespose.db database. The is accomplished with the process_data.py file.

The app folder houses templates for the web app and run.py to create the flask app.

The models folder houses the train_classifier.py NLP script and its exported pkl file. 

## Executing The Program
To run the data processing script execute the following command:
- python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db

To run the ML pipeline script execute the following command:
- python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

To run the web app script execute the following command:
- python app/run.py 


## Licensing, Authors, Acknowledgements
The data used in this project originated from Figure-8 (Appen) and Udacity.
