"""
TRAIN CLASSIFIER
Disaster Resoponse Project
Udacity - Data Science Nanodegree
"""

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import warnings

warnings.simplefilter('ignore')

def load_data():
    """
    Load Data Function
    
    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    database_filepath = 'data/Messages.db'
    engine = create_engine('sqlite:///'+database_filepath)
    print(engine.table_names())
    df = pd.read_sql_table('Messages',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize function
    
    Arguments:
        text -> list of text messages (english)
    Output:
        clean_tokens -> tokenized text, clean for ML modeling
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed


def build_model():
    """
    Build Model function
    
    This function output is a Scikit ML Pipeline that process text messages
    according to NLP best-practice and apply a classifier.

    """

    # Previously tried with RandomForest Classifier model, it was taking long time to execute
    # deferred this piece of code
    # pipeline = Pipeline([
    #         ('vect', CountVectorizer(tokenizer = tokenize)),
    #         ('tfidf', TfidfTransformer()),
    #         ('clf', MultiOutputClassifier(RandomForestClassifier()))
    #     ])

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])

    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model using GridSearchCV
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies ML pipeline to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    """
    Y_test_pred = model.predict(X_test)
    Y_pred_pd = pd.DataFrame(Y_test_pred, columns = category_names)
    for column in category_names:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_pd[column]))
        
    # Print the whole classification report.
    # Extremely long output
    # Work In Progress: Save Output as Text file!
    
    #Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    #for column in Y_test.columns:
    #    print('Model Performance with Category: {}'.format(column))
    #    print(classification_report(Y_test[column],Y_pred[column]))
    pass


def save_model(model):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipelin object
    
    """

    filename = 'models/classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """
    X, Y, category_names = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, )
    
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n ')
    save_model(model)

    print('Trained model saved!')



if __name__ == '__main__':
    main()