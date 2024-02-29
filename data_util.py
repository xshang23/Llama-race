import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
import random


cwd = os.getcwd()

def generate_prompt(data_point, shuffle=False):
    if not shuffle:
        return f"""
                Guess the race of the name enclosed in square brackets into 1 of the following 4 categories: Asian, Black, Hispanic, or White. 
                Your answer should only be the category name.
                [{data_point["name"]}].
                ANSWER: {data_point["label"]}
                """.strip()
    
    categories = ["Hispanic", "Black", "White", "Asian"]
    random.shuffle(categories)
    categories_str = ', '.join(categories)
    return f"""
            Guess the race of the name enclosed in square brackets into 1 of the following 4 categories: {categories_str}. 
            Your answer should only be the category name.
            [{data_point["name"]}]
            ANSWER: {data_point["label"]}
            """.strip()

def generate_test_prompt(data_point, shuffle=False):
    if not shuffle:
        return f"""
                Guess the race of the name enclosed in square brackets into 1 of the following 4 categories: Asian, Black, Hispanic, or White. 
                Your answer should only be the category name.
                [{data_point["name"]}]
                ANSWER: """.strip()
    
    categories = ["Hispanic", "Black", "White", "Asian"]
    random.shuffle(categories)
    categories_str = ', '.join(categories)
    return f"""
            Guess the race of the name enclosed in square brackets into 1 of the following 4 categories: {categories_str}. 
            Your answer should only be the category name.
            [{data_point["name"]}]
            ANSWER: """.strip()

def balanced_training_data(dir='data', shuffle=False):
    data_dir = os.path.join(cwd, dir)

    # load data and pre-process datasets
    train_df = pd.read_csv(os.path.join(data_dir, 'gptTrainNames.csv'))
    # test_df = pd.read_csv(os.path.join(data_dir, 'gptTestNames.csv'))

    X_train = list()
    X_eval = list()
    for race in ["API", "White", "Black", "Hispanic"]:
        train, eval  = train_test_split(train_df[train_df.label==race], 
                                        train_size=5000,
                                        test_size=500, 
                                        random_state=42)
        X_train.append(train)
        X_eval.append(eval)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_eval = pd.concat(X_eval).sample(frac=1, random_state=10)
    X_train = X_train.reset_index(drop=True)

    # X_train = pd.DataFrame(X_train.apply(generate_prompt(shuffle=shuffle), axis=1), 
    #                    columns=["name"])
    X_train = pd.DataFrame(X_train.apply(lambda row: generate_prompt(row, shuffle=shuffle), axis=1), 
                       columns=["name"])
    # X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), 
    #                   columns=["name"])
    X_eval = pd.DataFrame(X_eval.apply(lambda row: generate_prompt(row, shuffle=shuffle), axis=1), 
                       columns=["name"])
    
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)

    del train_df, X_train, X_eval

    return train_data, eval_data

def load_test_data(dir='data', shuffle=False):
    data_dir = os.path.join(cwd, dir)

    # load data and pre-process datasets
    test_df = pd.read_csv(os.path.join(data_dir, 'gptTestNames.csv'))
    y_true = test_df.label
    test_df = pd.DataFrame(test_df.apply(lambda row: generate_test_prompt(row, shuffle=shuffle), axis=1), 
                      columns=["name"])
    # test_data = Dataset.from_pandas(test_df)

    # del test_df

    return test_df, y_true


