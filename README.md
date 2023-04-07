# Dissaster Response Multilabel Model on WebApp
Multi label classification web app for dissaster messages <br>
Date: 07-Apr-2023

## Problem
The social media has showed us that it can be a powerful source of communication, in case of a disaster a challenge is how to properly use this source to classify each message to improve the response time to each case.

## Objective
Create a preprocessing and machine learning pipelines for disaster messages to create a multi model and deploy it as a web app.

### Process:
1. Create a database with a table for the preprocessed the CSV files with the texts and categories.
2. Create a machine learning pipeline for the multilabel classification.
3. Deploy the model as a Web App.

### Main Libraries used
Flask==2.2.3 <br>
nltk==3.8.1 <br> 
numpy==1.24.2 <br>
pandas==2.0.0 <br>
plotly==5.14.0 <br>
scikit-learn==1.2.2 <br>
SQLAlchemy==2.0.8 <br>

The complete set of libraries can be found in the requirements.txt file

### Files Description
</ul>
<b>app/</b>
<ul>
  <li><b>run.py:</b> Python file to execute the Web App</li>
  <li><b>templates/go.html:</b> HTML file to display the text input and classification result</li>
  <li><b>templates/master.html:</b> HTML file for the main page</li>
</ul>
<b>data/</b>
<ul>
  <li><b>DisasterResponse.db:</b> Database of the merged CSV files preprocessed</li>
  <li><b>process_data.py:</b> Python file to create the database from the CSV files</li>
  <li><b>disaster_categories.csv:</b> CSV Dataset with the categories of each text</li>
  <li><b>disaster_messages.csv:</b> Raw text messages to train the model</li>
</ul>
<b>models/</b>
<ul>
  <li><b>classifier.pkl:</b> Classification model as a pickle file</li>
  <li><b>train_classifier.py:</b> Python file to create the classification model</li>
</ul>
<b>notebooks/</b>
<ul>
  <li><b>ETL Pipeline Preparation.ipynb:</b> Jupyter Notebook for the creation of the process_data.py</li>
  <li><b>ML Pipeline Preparation.ipynb:</b> Jupyter Notebook for the creation of the train_classifier.py</li>
</ul>

### Results Summary
The current multilabel model has room for improvement in the NLP by adding additional feature extractions, and the ML model can also be improved by trying different models and increasing the range of the hyperparameter tuning which was limited due to time and process capabilities.

### Acknowledgements
Appen dataset shared by [Udacity](http://udacity.com/)
