# Disaster Response Project
Machine learning pipeline to categorize messages for disaster response.

### Table of Contents

1. [Project Motivation](#motivation)
2. [Web Application Screenshots](#screenshots)
3. [File Descriptions](#files)
4. [Usage](#usage)
5. [Acknowledgements](#acknowledgements)

## Project Motivation <a name="motivation"></a>

Following a disaster, typically you get millions of communications right at the time when disaster response organizations have the least capacity to filter and pull out the messages that are the most important for an appropiate response. Different organizations take care of different parts of the problem.
This project involves a web app where you can input a new message and get a classification result in several categories, such as "food", "shelter" and "medical aid". Therefore it serves as a way to classify the information, in order to refer it to the corresponding response organization. The web app also displays a visualization of the data. 

## Web application screenshots <a name="screenshots"></a>



## File Descriptions <a name="files"></a>

- Data
  - process_data.py: Python script to read in data from the csv files, clean it and store it in a SQL database.
  - disaster_messages.csv: Csv file with "id", "message", "original" and "genre" as columns.
  - disaster_categories.csv: Csv file with "id" and "categories" as columns.
  - DisasterResponse.db: Database created by process_data.py with cleaned data.
  -  
- Models
  - train_classifier.py: Python script to load data from the SQL database, train a classification model and save it as a pickle file.

- App
  - run.py: Python script to run the Flask app.
  - templates: HTML templates for the web app.


## Usage <a name="usage"></a>
 
In a terminal navigate to the disaster-response-project and run the following commands:
- python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
- python train_classifier.py DisasterResponse.db classifier.pkl
- python run.py

!!! [Falta how to run the app]

## Acknowledgements <a name="acknowledgements"></a>

The dataset was provided by [Udacity](https://www.udacity.com/), partnered with [Appen](https://appen.com/), as part of its Data Scientist Nanodegree Program.

