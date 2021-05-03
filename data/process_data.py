import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Loads csv files """
    
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    
    # merge datasets
    df = messages.merge(categories, on='id')
    
    return df
    
def clean_data(df):
    """ Splits categories into different columns and convert strings to numerical values
    (0 and 1). Finally replaces original "categories" column with the new ones and
    drops duplicates """
    
    # Split categories into different columns
    cols = [string[0:-2] for string in df.head(1).categories.values[0].split(';')]
    categories = df.categories.str.split(';', expand=True)
    categories.columns = cols
    
    # Convert strings into numerical values
    for column in categories:
        categories[column] = categories[column].str.slice(-1)
        categories[column] = pd.to_numeric(categories[column])
    
    # Replace original column
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)
    
    # Remove duplicates
    df.drop(df[df.duplicated()].index, axis=0, inplace=True)
    
    return df

def save_data(df, database_filename):
    """ Loads df to a sql database """
    
    path = 'sqlite:///' + database_filename
    engine = create_engine(path, encoding='UTF-8')
    df.to_sql('messages_and_categories', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
