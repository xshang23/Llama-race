from tqdm import tqdm
import os
import torch
from data_util import load_balanced_training_data, load_test_data
from peft import LoraConfig
from trl import SFTTrainer
from trl import setup_chat_format
import evaluation
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
# from sklearn.model_selection import train_test_split


# get working directory
cwd = os.getcwd()
# data_dir = os.path.join(cwd, 'data')
model_dir = os.path.join(cwd, 'model')

def load_model_and_tokenizer(model_name = "meta-llama/Llama-2-7b-chat-hf"):
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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer

def load_training_config(model, train_data, eval_data, tokenizer):
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
    return trainer

def predict(df, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(df))):
        prompt = df.iloc[i]["name"]
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

def evaluate(y_true, y_pred, file_name='res.txt'):
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

    # Calculate 1-GAP
    gap = evaluation.gap(y_true, y_pred)
    print(f'1-GAP: {gap:.3f}')

    # Calculate Disparate Impact
    dis = evaluation.disparate_impact(y_true, y_pred)
    print(f'Disparate Impact: {dis:.3f}')

    with open(file_name, 'w') as f:
        f.write(f'Accuracy: {accuracy:.3f}\n')
        f.write('\nClassification Report:\n')
        f.write(class_report + '\n')
        f.write('\nConfusion Matrix:\n')
        f.write(str(conf_matrix) + '\n')
        f.write(f'\n1-GAP: {gap:.3f}\n')
        f.write(f'Disparate Impact: {dis:.3f}\n')



def train(dir='data',train_shuffle=False):
    # load training and evaluation data
    train_data, eval_data = load_balanced_training_data(dir,train_shuffle,180000,20000)
    # load test data
    test_data, y_true = load_test_data(dir, frac=0.01)
    shuffle_test, y_true1 = load_test_data(dir, frac=0.01, shuffle=True)

    with open('statistics_balanced_data.txt', 'w') as f:
        f.write(f"Training data: {len(train_data)}\n")
        f.write(f"Evaluation data: {len(eval_data)}\n")
        f.write(f"Test data: {len(test_data)}\n")

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    # load training config
    trainer = load_training_config(model, train_data, eval_data, tokenizer)

    trainer.train()

    torch.save(model.state_dict(), os.path.join(model_dir, '720Kmodel_weights.pth'))

    model.eval()
    y_pred = predict(test_data, model, tokenizer)
    y_pred1 = predict(shuffle_test, model, tokenizer)

    with open('bal_test0301_true.txt', 'w') as f:
        for i in range(len(y_true)):
            f.write(f'{y_true[i]}\n')

    with open('bal_test0301_true_shuffle.txt', 'w') as f:
        for i in range(len(y_true1)):
            f.write(f'{y_true1[i]}\n')

    with open('bal_test0301_unsh_res.txt', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f'{y_pred[i]}\n')

    with open('bal_test0301_sh_res.txt', 'w') as f:
        for i in range(len(y_pred1)):
            f.write(f'{y_pred1[i]}\n')

    evaluate(y_true, y_pred, 'bal_test0301_unsh_report.txt')
    evaluate(y_true1, y_pred1, 'bal_test0301_sh_report.txt')


if __name__ == "__main__":
    train(dir='data',train_shuffle=False)
    # train(model,dir='data',train_shuffle=True)

    

    
    
    