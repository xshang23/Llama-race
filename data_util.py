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

def load_balanced_training_data(dir='data', shuffle=False, training_num_samples_per_class=5000, eval_num_samples_per_class=500):
    """
    Load training and evaluation data and pre-process datasets

    Args:
    dir: str
        Directory where data is stored
    shuffle: bool
        Whether to shuffle the order of categories in the prompt
    training_num_samples_per_class: int
        number of samples per class to use for training
    Number of samples per class to use for training
        eval_num_samples_per_class: int

    Returns:
    train_data: Dataset
        Training dataset
    eval_data: Dataset
        Evaluation dataset
    """
    data_dir = os.path.join(cwd, dir)

    # load data and pre-process datasets
    train_df = pd.read_csv(os.path.join(data_dir, 'gptTrainNames.csv'))
    eval_df = pd.read_csv(os.path.join(data_dir, 'gptValNames.csv'))

    train_df = train_df.groupby('label').apply(lambda x: x.sample(n=min(len(x), training_num_samples_per_class))).reset_index(drop=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print(train_df['label'].value_counts())
    
    eval_df = eval_df.groupby('label').apply(lambda x: x.sample(n=min(len(x), eval_num_samples_per_class))).reset_index(drop=True)
    eval_df = eval_df.sample(frac=1).reset_index(drop=True)
    print(eval_df['label'].value_counts())

    train_df = pd.DataFrame(train_df.apply(lambda row: generate_prompt(row, shuffle=shuffle), axis=1), columns=["name"])
    eval_df = pd.DataFrame(eval_df.apply(lambda row: generate_prompt(row, shuffle=shuffle), axis=1), columns=["name"])
    
    train_data = Dataset.from_pandas(train_df)
    eval_data = Dataset.from_pandas(eval_df)

    del train_df, eval_df
    return train_data, eval_data


def load_training_data(dir='data', shuffle=False, frac=1):
    """
    Load training data and pre-process datasets

    Args:
    dir: str
        Directory where data is stored
    shuffle: bool
        Whether to shuffle the order of categories in the prompt
    frac: float
        Fraction of the training data to use

    Returns:
    train_data: Dataset
        Training dataset
    eval_data: Dataset
        Evaluation dataset
    """
    data_dir = os.path.join(cwd, dir)

    # load data and pre-process datasets
    train_df = pd.read_csv(os.path.join(data_dir, 'gptTrainNames.csv'))
    eval_df = pd.read_csv(os.path.join(data_dir, 'gptValNames.csv'))

    train_df = train_df.sample(frac=frac, random_state=10).reset_index(drop=True)
    eval_df = eval_df.sample(frac=frac, random_state=10).reset_index(drop=True)

    train_df = pd.DataFrame(train_df.apply(lambda row: generate_prompt(row, shuffle=shuffle), axis=1), columns=["name"])
    eval_df = pd.DataFrame(eval_df.apply(lambda row: generate_prompt(row, shuffle=shuffle), axis=1), columns=["name"])

    train_data = Dataset.from_pandas(train_df)
    eval_data = Dataset.from_pandas(eval_df)

    del train_df, eval_df
    return train_data, eval_data




def load_test_data(dir='data', frac=1, shuffle=False):
    """
    Load test data and pre-process datasets

    Args:
    dir: str
        Directory where data is stored
    frac: float
        Fraction of the test data to use
    shuffle: bool
        Whether to shuffle the order of categories in the prompt
    
    Returns:
    test_df: DataFrame
        Test dataset
    y_true: List
        True labels for the test dataset
    """
    data_dir = os.path.join(cwd, dir)

    # load data and pre-process datasets
    test_df = pd.read_csv(os.path.join(data_dir, 'gptTestNames.csv'))
    test_df = test_df.sample(frac=frac, random_state=10).reset_index(drop=True)
    y_true = test_df.label.values
    test_df = pd.DataFrame(test_df.apply(lambda row: generate_test_prompt(row, shuffle=shuffle), axis=1), columns=["name"])

    return test_df, y_true


if __name__ == "__main__":
    # train_data, eval_data = load_balanced_training_data()
    train_data, eval_data = load_training_data(frac=0.01)
    print(train_data,eval_data)
    test_data, y_true = load_test_data(frac=0.01)
    print(len(test_data),len(y_true))
    