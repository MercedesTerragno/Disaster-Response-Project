import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['wordnet', 'stopwords'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_and_categories', engine)

    X = df.message #.values
    Y = df.drop(['message', 'original', 'genre'], axis=1) #.values

    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
  
    # replace each url in text string with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, "urlplaceholder", text)
    
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    
    # strip whitespaces and lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w.strip()) for w in tokens]

    # drop duplicates
    tokens = pd.Series(tokens).drop_duplicates().tolist()
    
    return tokens


def build_model():
    classifier = LinearSVC(C=0.1, dual=False, class_weight='balanced')
    
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer= tokenize, stop_words= 'english')),
    ('clf', MultiOutputClassifier(classifier, n_jobs=-1))
])
    
    params = {'clf__estimator__C': [0.05, 0.1, 0.5]}

    grid_search = GridSearchCV(pipeline, cv=3, param_grid= params, n_jobs=-1, 
                               scoring= 'f1_weighted', verbose=3)
    
    return grid_search

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    
    reports= {}
    for i, cat in enumerate(category_names):
        precision, recall, f_score, _ = precision_recall_fscore_support(
            Y_test[cat], Y_pred[:,i], average='weighted')
        reports[cat] = precision, recall, f_score

    results = pd.DataFrame(reports).transpose().rename(
        columns={ 0: 'precision', 1: 'recall', 2:'fscore'})
    results.loc['Average', :] = results.precision.mean(), results.recall.mean(), \
                                results.fscore.mean()
    print(results)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

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
