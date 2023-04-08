import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load CSV data from messages and categories
    
    INPUT
    messages_filepath: local filepath for the messages dataset with CSV extension
    categories_filepath: local filepath for the categories dataset with CSV extension
    
    OUTPUT
    df: Pandas dataframe of the merged datasets
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on='id', how='left')

    return df



def clean_data(df):
    '''
    Preprocessing of the dataframe, split categories and convert to binary values,
    replace incorrect values, drop empty columns and duplicate rows.

    INPUT
    df: merged dataframe with the text and categories from the CSV files

    OUTPUT
    df: Cleaned dataframe
    '''

    #Split text to columns to separate each category for classification
    categories = df['categories'].str.split(';',expand=True)

    #Rename columns with each of their corresponding category
    row = categories.loc[0]
    category_colnames = row.str.replace(r'-[0-9]', '', regex=True).tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Replace incorrect values in category 'related' 2 -> 1
    categories['related'] = categories['related'].replace(2,1)
    
    #Merge separated categories with text message
    df = df.merge(categories, left_index=True, right_index=True)
    
    #Drop duplicated values and empty category 'child alone'
    df.drop(['categories', 'child_alone'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Create a database and save the dataframe in the table Messages
    
    INPUT
    df: Cleaned dataframe to save in database
    database_filename: string for the database name with *.db extension
    
    OUTPUT
    database file with *.db extension with the table Messages loaded
    '''
    engine = create_engine('sqlite:///'+database_filename)  
    
    #Load dataframe into database table and replace table if already exists
    df.to_sql('Messages', engine, if_exists='replace', index=False)

def main():
    '''
    Main program to execute all ETL functions
    INPUT
    Execute from command window with the following syntaxis.
        python thisfilename.py messagesfile.csv categoriesfile.csv databasename.db
        Example: "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"

    OUTPUT
    Database with the table Messages loaded with the merged dataframe 
    '''
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