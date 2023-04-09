import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

from nltk.corpus import stopwords


app = Flask(__name__)

def tokenize(text):
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

# load data
engine = create_engine('sqlite:///home/fernando17/Dissaster_Msgs_Classification/data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("home/fernando17/Dissaster_Msgs_Classification/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('home/fernando17/Dissaster_Msgs_Classification/')
@app.route('home/fernando17/Dissaster_Msgs_Classification/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cat_count = df.iloc[:,4:].sum().sort_values(ascending=False)

    X = df.message.values

    vocab_dict = {}
    for i in range(len(X)):
        text = tokenize(X[i])
        for w in text:
            if w in vocab_dict:
                vocab_dict[w] += 1
            else:
                vocab_dict[w] = 1
    vocab_count = pd.Series(vocab_dict, name='word_count').sort_values(ascending=False)
    vocab_words = vocab_count[:20].index
    vocab_values = vocab_count[:20].values

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(
                        color='rgba(50, 171, 96, 0.6)',
                        line=dict(
                            color='rgba(50, 171, 96, 1.0)',
                            width=1))
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_count.index,
                    y=cat_count.values,
                    marker=dict(
                        color='rgba(50, 171, 96, 0.6)',
                        line=dict(
                            color='rgba(50, 171, 96, 1.0)',
                            width=1))
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=vocab_words,
                    y=vocab_values,
                    marker=dict(
                        color='rgba(50, 171, 96, 0.6)',
                        line=dict(
                            color='rgba(50, 171, 96, 1.0)',
                            width=1))
                )
            ],

            'layout': {
                'title': 'Top 20 Processed Words in Dataset',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Processed Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON, data_set=df)


# web page that handles user query and displays model results
@app.route('home/fernando17/Dissaster_Msgs_Classification/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
