import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
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
from sklearn.model_selection import train_test_split

# get working directory
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
model_dir = os.path.join(cwd, 'model')

# load data and pre-process datasets
train_df = pd.read_csv(os.path.join(data_dir, 'gptTrainNames.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'gptTestNames.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'gptValNames.csv'))

X_train = train_df.sample(frac=0.01, random_state=10)
X_test = test_df.sample(frac=0.01, random_state=10)
X_eval = val_df.sample(frac=0.01, random_state=10)

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


X_train = pd.DataFrame(X_train.apply(lambda row: generate_prompt(row, shuffle=False), axis=1), 
                       columns=["name"])
X_eval = pd.DataFrame(X_eval.apply(lambda row: generate_prompt(row, shuffle=False), axis=1), 
                       columns=["name"])

y_true = X_test.label.values
X_test_1 = pd.DataFrame(X_test.apply(lambda row: generate_test_prompt(row, shuffle=True), axis=1), 
                      columns=["name"])
X_test_2 = pd.DataFrame(X_test.apply(lambda row: generate_test_prompt(row, shuffle=False), axis=1), 
                      columns=["name"])

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)

del train_df, test_df, val_df, X_train, X_eval

def evaluate(y_true, y_pred):
    labels = ['API', 'Black', 'Hispanic', 'White']
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    print('\nConfusion Matrix:')
    print(conf_matrix)
    with open('test0229.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.3f}\n')
        f.write(f'Classification Report:\n{class_report}\n')
        f.write(f'Confusion Matrix:\n{conf_matrix}\n')


model_name = "meta-llama/Llama-2-7b-chat-hf"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=compute_dtype,
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True,
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model, tokenizer = setup_chat_format(model, tokenizer)

def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
    # for i in [69, 222, 676, 1270, 2060, 3684, 3827, 4472, 4799, 4972, 5120]:
        prompt = test.iloc[i]["name"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 4, 
                        # temperature = 0.01,
                        do_sample = False,
                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split(":")[-1].lower()
        # print(prompt, answer)
        if "asian" in answer or "api" in answer:
            y_pred.append("API")
        elif "black" in answer:
            y_pred.append("Black")
        elif "hispanic" in answer:
            y_pred.append("Hispanic")
        elif "white" in answer:
            y_pred.append("White")
        else:
            y_pred.append("none")
            print(prompt,answer)
    return y_pred

output_dir="trained_weigths"

peft_config = LoraConfig(
        lora_alpha=16, 
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,                    # directory to save and repository id
    num_train_epochs=2,                       # number of training epochs
    per_device_train_batch_size=384,            # batch size per device during training
    gradient_accumulation_steps=1,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,                         # log every 10 steps
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    group_by_length=True,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    report_to="tensorboard",                  # report metrics to tensorboard
    evaluation_strategy="epoch"               # save checkpoint every epoch
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="name",
    tokenizer=tokenizer,
    max_seq_length=None,
    packing=False,
    dataset_batch_size=384,
    # dataset_kwargs={
    #     "add_special_tokens": False,
    #     "append_concat_token": False,
    # }
)

trainer.train()
model.eval()

y_pred = predict(X_test_1, model, tokenizer)
y_pred1 = predict(X_test_2, model, tokenizer)

with open('test0229_true.txt', 'w') as f:
    for i in range(len(y_true)):
        f.write(f'{y_true[i]}\n')

with open('test0229_res1.txt', 'w') as f:
    for i in range(len(y_pred)):
        f.write(f'{y_pred[i]}\n')

with open('test0229_res2.txt', 'w') as f:
    for i in range(len(y_pred1)):
        f.write(f'{y_pred1[i]}\n')

evaluate(y_true, y_pred)
evaluate(y_true, y_pred1)


f1_micro_1 = f1_score(y_true, y_pred, average='micro')
f1_macro_1 = f1_score(y_true, y_pred, average='macro')
f1_weighted_1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score (Micro): {f1_micro_1:.3f}")
print(f"F1 Score (Macro): {f1_macro_1:.3f}")
print(f"F1 Score (Weighted): {f1_weighted_1:.3f}")

print("--------------below is for un-shuffle test----------------")
f1_micro = f1_score(y_true, y_pred1, average='micro')
f1_macro = f1_score(y_true, y_pred1, average='macro')
f1_weighted = f1_score(y_true, y_pred1, average='weighted')
print(f"F1 Score (Micro): {f1_micro:.3f}")
print(f"F1 Score (Macro): {f1_macro:.3f}")
print(f"F1 Score (Weighted): {f1_weighted:.3f}")

with open('0229_f1_score.txt', 'w') as f:
    f.write(f"F1 Score (Micro): {f1_micro_1:.3f}\n")
    f.write(f"F1 Score (Macro): {f1_macro_1:.3f}\n")
    f.write(f"F1 Score (Weighted): {f1_weighted_1:.3f}\n")
    f.write("--------------below is for un-shuffle test----------------\n")
    f.write(f"F1 Score (Micro): {f1_micro:.3f}\n")
    f.write(f"F1 Score (Macro): {f1_macro:.3f}\n")
    f.write(f"F1 Score (Weighted): {f1_weighted:.3f}\n")

def gap(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=['API','Black','Hispanic','White'])
    # metrics_per_class = {}
    tpr, tnr = [], []
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = sum(cm[:, i]) - TP
        FN = sum(cm[i, :]) - TP
        TN = sum(cm.sum(axis=1)) - TP - FP - FN
        TPR = TP / float(TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / float(TN + FP) if (TN + FP) > 0 else 0
        # class_label = le.inverse_transform([i])[0]  # Convert index back to original class label
        # metrics_per_class[class_label] = {'TPR': TPR, 'TNR': TNR}
        tpr.append(TPR)
        tnr.append(TNR)

    temp = (np.array(tpr)+np.array(tnr))*0.5
    # print(temp)
    gap = 0.0
    for i in range(len(temp)-1):
        gap += abs(temp[i]-temp[-1])
    gap /= 3
    print("1-GAP: ", round(1-gap, 4))
    return round(1-gap, 4)

with open('0229_gap.txt', 'w') as f:
    f.write(f'1-GAP: {gap(y_true, y_pred)}\n')
    f.write("--------------below is for un-shuffle test----------------\n")
    f.write(f'1-GAP: {gap(y_true, y_pred1)}\n')


def disparate_impact(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, labels=['API','Black','Hispanic','White'])
    dis = 0.0
    for i in range(len(recalls)-1):
        dis += recalls[i]/recalls[-1]
    dis /= (len(recalls)-1)
    print('Disparate Impact is:',dis)
    return round(dis, 4)

with open('0229_disparate_impact.txt', 'w') as f:
    f.write(f'Disparate Impact is: {disparate_impact(y_true, y_pred)}\n')
    f.write("--------------below is for un-shuffle test----------------\n")
    f.write(f'Disparate Impact is: {disparate_impact(y_true, y_pred1)}\n')
