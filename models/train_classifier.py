import sys
import re
import pandas as pd
#import numpy as np
import sqlalchemy as db
import pickle

import nltk
nltk.download(['stopwords', 'punkt', 'wordnet'])

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath):
    '''
    Load dataframe with messages and categories from database and separate
    for classification model

    INPUT
    database_filepath: file path to the database as a string with extension *.db

    OUTPUT
    X and Y vectors for model and labels for classification
    '''
    engine = db.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    
    #Select only text values for classification model
    X = df.message.values
    
    #Drop the additional columns to obtain a matrix with target values
    df_y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    y = df_y.values
    labels = list(df_y.columns)
    
    return X, y, labels


def tokenize(text):
    '''
    Create clean tokens without stop words from text strings for model
    
    INPUT
    text: text string for tokenization

    OUTPUT
    list of clean tokens of text string
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    
    stop_words = stopwords.words("english")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
   
    return clean_tokens


def build_model():
    '''
    Create pipeline model from tokenization to model classifier

    OUTPUT
    Pipeline model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range= (1, 1), tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #Parameter to search for best model performance based of f1 score
    params = {
        'vect__max_df': [0.75, 1.0],
        'clf__estimator__criterion': ['entropy'], #'gini'
        'clf__estimator__n_estimators': [100, 150]
    }

    cv = GridSearchCV(pipeline, scoring='f1_weighted', cv=2, param_grid=params, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model evaluation using the classification report for extraction of the metrics 
    f1 score, precision and recall of each category

    INPUT
    model: pipeline text classifier model
    X_test: text string for model testing as values
    Y_test: numpy matrix of target categories 
    category_names: list of target categories

    OUTPUT
    Print report of f1 score, precision and recall of each category
    '''    
    y_pred = model.predict(X_test)

    report = classification_report(Y_test, y_pred, target_names=category_names, zero_division=0, output_dict=True)
    
    #Transform report to dataframe and sort by f1-score
    df_report = pd.DataFrame(report).transpose()
    print(df_report.loc[category_names].sort_values(by=['f1-score'], ascending=False))


def save_model(model, model_filepath):
    '''
    Save the trained model as a pickle file

    INPUT
    model: trained classifier model
    model_filepath: string for the file path with *.pkl extension
    
    OUTPUT
    model file with extension *.pkl
    '''
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    '''
    Main program to create a classification model for the text messages
    INPUT
    Execute from command window with the following syntaxis.
        python thisfilename.py databasename.db modelname.pkl
        Example: "python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl"

    OUTPUT
    Classification model as a pickle file with extension *.pkl
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()