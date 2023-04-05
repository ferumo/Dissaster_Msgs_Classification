import sys
import re
import pandas as pd
import numpy as np
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
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def load_data(database_filepath):
    engine = db.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    
    X = df.message.values
    
    df_y = df.drop(['id', 'message', 'original', 'genre','child_alone'], axis=1)
    y = df_y.values
    labels = list(df_y.columns)
    
    return X, y, labels


def tokenize(text):
    
    stop_words = stopwords.words("english")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(w).lower() for w in tokens if w not in stop_words]
   
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(solver='sag')))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)

    report = classification_report(Y_test, y_pred, target_names=category_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose().sort_values(by=['f1-score'], ascending=False)
    print(df_report)


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
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