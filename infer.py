import pandas as pd
import numpy as np
import os
from random import randrange
from functools import partial
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
HfArgumentParser,
Trainer,
TrainingArguments,
DataCollatorForLanguageModeling,
DataCollatorWithPadding,
EarlyStoppingCallback,
pipeline,
logging,
set_seed)
from tqdm import tqdm

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer, setup_chat_format
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)

# constants
llama_checkpoint = "meta-llama/Llama-2-7b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
col_to_delete = ['id', 'keyword','location', 'text']

# get working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
model_dir = os.path.join(cwd, 'model')

# load data and pre-process datasets
train_df = pd.read_csv(os.path.join(data_dir, 'gptTrainNames.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'gptTestNames.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'gptValNames.csv'))

def generate_prompt(data_point):
    return f"""
            Analyze the race of the name enclosed in square brackets, 
            determine if it is API, Black, Hispanic, or White, and return the answer as 
            the corresponding race label "API" or "Black" or "Hispanic" or "White".

            [{data_point["name"]}] = {data_point["label"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Analyze the race of the name enclosed in square brackets, 
            determine if it is API, Black, Hispanic, or White, and return the answer as 
            the corresponding race label "API" or "Black" or "Hispanic" or "White".

            [{data_point["name"]}] = """.strip()

train_df['name'] = train_df['name'].apply(generate_prompt)
test_df['name'] = test_df['name'].apply(generate_test_prompt)
val_df['name'] = val_df['name'].apply(generate_prompt)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
val_dataset = Dataset.from_pandas(val_df)

y_true = test_df.label.values
X_test = test_df.name.values

datasets = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset}

# free memory
del(train_df, test_df, val_df, train_dataset, test_dataset, val_dataset)

# create a function to evaluate the model
def evaluate(y_true, y_pred):
    labels = ['API', 'Black', 'Hispanic', 'White']
    mapping = {'API': 0, 'Black': 1, 'Hispanic':2, 'White': 3}
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
    
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

def llama_preprocessing_function(examples):
    return llama_tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

# Load Llama 2 Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint, trust_remote_code=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

llama_tokenized_datasets = datasets.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
llama_tokenized_datasets.set_format("torch")

# Load Llama 2 Model
llama_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=llama_checkpoint,
    device_map="auto",
    torch_dtype=compute_dtype,
    quantization_config=bnb_config, 
)

llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1

llama_model, llama_tokenizer = setup_chat_format(llama_model, llama_tokenizer)

def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["name"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 1, 
                        temperature = 0.0,
                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("=")[-1]
        if "API" in answer:
            y_pred.append("API")
        elif "Black" in answer:
            y_pred.append("Black")
        elif "Hispanic" in answer:
            y_pred.append("Hispanic")
        elif "White" in answer:
            y_pred.append("White")
        else:
            y_pred.append("none")
    return y_pred