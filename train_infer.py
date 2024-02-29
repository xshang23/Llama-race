import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from data_util import balanced_training_data, load_test_data
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix,
                             f1_score,
                             recall_score)
# from sklearn.model_selection import train_test_split


# get working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
model_dir = os.path.join(cwd, 'model')

# load data and pre-process datasets
def train(model,dir='data',train_shuffle=False):
    train_data, eval_data = balanced_training_data(dir, train_shuffle)
    # test_df = load_test_data(dir, test_shuffle)
    
    