import sys
import pandas as pd
from sqlalchemy import create_engine
import os
import nltk
nltk.download(['punkt'])
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle



def load_data(database_filepath):
    """load cleaned data from DisasterResponse database

    Args:
    database_filepath: location of DisasterResponse database

    Returns: 
    X: features
    Y: labels
    category_names: list of category names
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = 'DisasterResponse_table'
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenizes the text

    Args:
    text: disaster messages

    Returns: 
    clean_tokens: tokens that are extracted from text
    """

    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = []
    for w in tokens:
        clean_tok = lemmatizer.lemmatize(w).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    


def build_model():
    """create ml pipeline for the model

    Args:
    n/a

    Returns: 
    model: a text classifier
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [1,10],
        'clf__estimator__min_samples_split': [2],
    
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Apply the model to data, predict, and score the model

    Args:
    model: text classifier model
    X_test: test features
    Y_test: test labels
    category_names: category names

    Returns: 
    n/a
    """
    Y_pred = model.predict(X_test)

    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))

def save_model(model, model_filepath):
    """Saves the model as a pkl file

    Args:
    model: text classifier model
    model_filepath: save location for pkl file

    Returns: 
    n/a
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """Main function for classifier pipeline"""

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