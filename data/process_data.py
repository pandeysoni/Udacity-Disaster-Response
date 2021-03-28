
# 1. import libraries
import pandas as pd
from sqlalchemy import create_engine
import os


# 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps
def load_data():
    """
    Load Data function

        df -> Loaded dasa as Pandas DataFrame
    """
    messages = pd.read_csv('data/disaster_messages.csv')
    categories = pd.read_csv('data/disaster_categories.csv')

    df = pd.merge(messages, categories, left_on='id', right_on='id')
    return df 


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.
def clean_data(df):
    """
    Clean Data function
    
    Arguments:
        df -> raw data Pandas DataFrame
    Outputs:
        df -> clean data Pandas DataFrame
    """
    categories = df.categories.str.split(pat=';',expand=True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # ### 4. Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    # For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # ### 5. Replace `categories` column in `df` with new category columns.
    # - Drop the categories column from the df dataframe since it is no longer needed.
    # - Concatenate df and categories data frames.
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)

    # ### 6. Remove duplicates.
    # - Drop the duplicates.
    df.drop_duplicates(inplace = True)
    return df


# ### 7. Save the clean dataset into an sqlite database.
def save_data(df):
    """
    Save Data function
    
    Arguments:
        df -> Clean data Pandas DataFrame
    """
    database_filename = 'data/Messages.db'
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False,  if_exists='replace')
    pass  

def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Load CSV File
        2) Data cleaning and pre-processing
        3) Data save to SQLite database
    """
    
    df = load_data()

    print('Cleaning data...')
    df = clean_data(df)
    
    save_data(df)
    
    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()
