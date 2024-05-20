import os
from tqdm import tqdm
import torch
from data_util import load_test_data
from evaluation import gap, disparate_impact, evaluate
from peft import LoraConfig, PeftConfig, get_peft_model
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          pipeline)
import logging

MODEL_PATH = {'meta-llama/Llama-2-7b-chat-hf': '/home/common_models/Llama-2-7b-chat-hf', 
              'mistralai/Mistral-7B-v0.1': '/home/common_models/Mistral-7B-v0.1'}

# get working directory
cwd = os.getcwd()
# data_dir = os.path.join(cwd, 'data')
model_dir = os.path.join(cwd, 'model')

log_dir = os.path.join(cwd, 'logs','fair_loss')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, "fairloss.txt"),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
    # for i in [69, 222, 676, 1270, 2060, 3684, 3827, 4472, 4799, 4972, 5120]:
        prompt = test.iloc[i]["name"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 3, 
                        # temperature = 0.01,
                        do_sample = False,
                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split(":")[-1].lower()
        # print(prompt, answer)
        if "asian" in answer or "api" in answer or 'japanese' in answer:
            y_pred.append("API")
        elif "black" in answer or 'b' in answer:
            y_pred.append("Black")
        elif "hispanic" in answer:
            y_pred.append("Hispanic")
        elif "white" in answer or 'greek' in answer:
            y_pred.append("White")
        else:
            y_pred.append("API")
            print(prompt,answer)
    return y_pred

def main():
    # Load data
    test_df, y_true = load_test_data(frac=0.01)
    
    # Load model
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH[model_name],
            torch_dtype=torch.bfloat16,
            device_map=device,
            # token='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',
        )
    peft_config = LoraConfig(
            lora_alpha=16, 
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
       )
    
    model = get_peft_model(model, peft_config)
    saved_model = "/home/xshang/Llama-race/model/fair_loss_0.3lambda_Llama-2-7b-chat-hf.pth"
    model.load_state_dict(torch.load(saved_model))
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], device_map=device)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    y_pred = predict(test_df, model, tokenizer)
    evaluate(y_true, y_pred, logger)
    gap(y_true, y_pred, logger)
    disparate_impact(y_true, y_pred, logger=logger)

if __name__ == "__main__":
    main()