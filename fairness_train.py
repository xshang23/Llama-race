import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm
import argparse
import torch
from data_util import load_training_data_for_fairness_loss, load_test_data
from fairness_loss import fair_loss
from evaluation import gap, disparate_impact, evaluate, AverageMeter
import torch.nn as nn
import transformers
from transformers import get_linear_schedule_with_warmup
# from datasets import Dataset
from peft import LoraConfig, PeftConfig, get_peft_model
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline)
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter

MODEL_PATH = {
            'Mistral-7B-v0.1': '/WAVE/projects/newsq_scu/base_models/Mistral-7B-v0.1',
            'Llama-2-7b-chat-hf': '/WAVE/projects/newsq_scu/base_models/Llama-2-7b-chat-hf',
            'Meta-Llama-3-8B-Instruct': '/WAVE/projects/newsq_scu/base_models/Meta-Llama-3-8B-Instruct',
            }


def predict(test, model, tokenizer, logger=None):
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
        
        if "asian" in answer or "api" in answer:
            y_pred.append("API")
        elif "black" in answer:
            y_pred.append("Black")
        elif "hispanic" in answer:
            y_pred.append("Hispanic")
        elif "white" in answer:
            y_pred.append("White")
        else:
            y_pred.append(answer)
            print(prompt,answer)
            if logger:
                logger.info(f"index{i}: {answer}")
        
    return y_pred

def main(frac=0.01, 
         lambda_val=0.2, 
         num_epochs=2, 
         batch_size=16, 
         model_name="meta-llama/Llama-2-7b-chat-hf",
         exp_name="fair_loss",
         loss_scale=True,):
    
    # get working directory
    cwd = os.getcwd()
    # data_dir = os.path.join(cwd, 'data')
    model_dir = os.path.join(cwd, 'model')

    # Set up logging
    log_dir = os.path.join(cwd, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir,exp_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, model_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, f'lambada_{lambda_val}.txt'),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info(f"Training and testing with frac={frac}, lambda={lambda_val}, num_epochs={num_epochs}, batch_size={batch_size}, model_name={model_name}")
    print(f"Training and testing with frac={frac}, lambda={lambda_val}, num_epochs={num_epochs}, batch_size={batch_size}, model_name={model_name}")


    # Load training and test data
    train_loader, eval_loader, train_length = load_training_data_for_fairness_loss(frac=frac, batch_size=batch_size)
    test_df, y_true, test_loader = load_test_data(frac=frac, batch_size=batch_size)

    # model_name = MODEL_PATH[model_name]
    # num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH[model_name], 
            torch_dtype=torch.bfloat16,
            device_map=device,
            token='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'
        )

    peft_config = LoraConfig(
            lora_alpha=16, 
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name],
                                              token='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'
                                              )
    # if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = get_peft_model(model, peft_config)
    # trainable_params, all_param = model.get_nb_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=(train_length * num_epochs),
                )
    
    output_dir = "trained_weights"
    logwriter = SummaryWriter(os.path.join(cwd, output_dir, "runs", f'{model_name.split("/")[-1]}_{lambda_val}_' + str(datetime.now().strftime("%b%d_%H-%M-%S"))))

    current_step = 0
    avg_train_loss = AverageMeter()
    avg_val_loss = AverageMeter()
    avg_fair_loss = AverageMeter()
    # avg_test_fairness_loss = AverageMeter()
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1}")
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_ids = tokenizer(batch['data'], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            vir_labels = input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=vir_labels)
            labels = torch.tensor(batch['label']).to(device)
            fairness_loss = fair_loss(outputs.logits, labels, tokenizer, lambda_val=lambda_val)
            if lambda_val == 0:
                loss = outputs.loss
            else:
                loss = outputs.loss + fairness_loss
            
            logwriter.add_scalar('Training/train_loss_step', loss.detach().float().item(), current_step)
            logwriter.add_scalar('Training/LLM_loss_step', outputs.loss.detach().float().item(), current_step)
            logwriter.add_scalar('Training/fairness_loss_step', fairness_loss.detach().float().item(), current_step)
            avg_train_loss.update(loss.detach().float().item())
            avg_fair_loss.update(fairness_loss.detach().float().item())
            logwriter.add_scalar('Training/train_avg_loss_step', avg_train_loss.avg, current_step)
            logwriter.add_scalar('Training/train_avg_fairness_loss_step', avg_fair_loss.avg, current_step)
            current_step += 1

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # print(f"Loss: {loss.item()}")
            
        model.eval()
        # y_pred = list()
        # steps = 0
        total_loss = 0
        total_fairness_loss = 0
        total_LLM_loss = 0
        for step, batch in enumerate(tqdm(eval_loader)):
            input_ids = tokenizer(batch['data'], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            vir_labels = input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=vir_labels)
            labels = torch.tensor(batch['label']).to(device)
            fairness_loss = fair_loss(outputs.logits, labels, tokenizer, lambda_val=lambda_val, loss_scale=loss_scale)
            llm_loss = outputs.loss.detach().float().item()
            if lambda_val == 0:
                loss = outputs.loss 
            else:   
                loss = outputs.loss + fairness_loss
            total_loss += loss.detach().float().item()
            fairness_loss = fairness_loss.detach().float().item()
            total_fairness_loss += fairness_loss
            total_LLM_loss += llm_loss
            avg_val_loss.update(loss.detach().float().item())
        logger.info(f"Val loss: {total_loss/len(eval_loader)}, epoch: {epoch}")
        logger.info(f"Val fairness loss: {total_fairness_loss/len(eval_loader)}, epoch: {epoch}")
        logger.info(f"Val LLM loss: {total_LLM_loss/len(eval_loader)}, epoch: {epoch}")
            # logwriter.add_scalar('Val/val_loss_step', loss.detach().float().item(), steps)
            # logwriter.add_scalar('Val/val_LLM_loss_step', outputs.loss.detach().float().item(), steps)
            # logwriter.add_scalar('Val/fairness_loss_step', fairness_loss.detach().float().item(), steps)
            # logwriter.add_scalar('Val/val_loss_avg', avg_val_loss.avg, steps)
            # steps += 1

        logwriter.add_scalar('Training/train_loss_epoch', avg_train_loss.avg, epoch)
        logwriter.add_scalar('Training/val_loss_epoch', avg_val_loss.avg, epoch)

    # try:
    #     torch.save(model.state_dict(), os.path.join(model_dir, f'fair_loss_{lambda_val}lambda_{model_name}.pth'))
    # except Exception as e:
    #     print("Error saving model")
    #     print(e, "\n")
    #     pass
    
    #calculate the loss on the test set
    # for step, batch in enumerate(tqdm(eval_loader)):
    #     pass
    model.eval()
    # steps = 0
    total_loss = 0
    total_fairness_loss = 0
    total_LLM_loss = 0
    for step, batch in enumerate(tqdm(test_loader)):
        input_ids = tokenizer(batch['data'], return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        vir_labels = input_ids.to(device)
        outputs = model(input_ids=input_ids, labels=vir_labels)
        labels = torch.tensor(batch['label']).to(device)
        fairness_loss = fair_loss(outputs.logits, labels, tokenizer, lambda_val=lambda_val, loss_scale=loss_scale)
        llm_loss = outputs.loss.detach().float().item()
        fairness_loss = fairness_loss.detach().float().item()
        total_fairness_loss += fairness_loss
        total_LLM_loss += llm_loss
        if lambda_val == 0:
            total_loss += llm_loss
        else:   
            total_loss += llm_loss + fairness_loss

        # avg_test_loss.update(llm_loss + fairness_loss)
        # logwriter.add_scalar('Test/test_loss_step', loss.detach().float().item(), steps)
        # logwriter.add_scalar('Test/test_LLM_loss_step', outputs.loss.detach().float().item(), steps)
        # logwriter.add_scalar('Test/test_fairness_loss_step', fairness_loss.detach().float().item(), steps)
        # logwriter.add_scalar('Test/test_loss_avg', avg_val_loss.avg, steps)
        # steps += 1

    logger.info(f"Test loss: {total_loss/len(test_loader)}")
    logger.info(f"Test fairness loss: {total_fairness_loss/len(test_loader)}")
    logger.info(f"Test LLM loss: {total_LLM_loss/len(test_loader)}")



    # Predict on the test set
    y_pred = predict(test_df, model, tokenizer,logger=logger)
    # Save the list to a text file
    try:
        with open(os.path.join(log_dir, f'result_{lambda_val}.txt'), 'w') as file:
            for item in y_pred:
                file.write("%s\n" % item)
    except Exception as e:
        print("Error saving predictions")
        print(e, "\n")
        pass

    try: 
        evaluate(y_true, y_pred, logger=logger)
        gap(y_true, y_pred, logger=logger)
        disparate_impact(y_true, y_pred, logger=logger)
    except Exception as e:
        print("Error evaluating model")
        print(e, "\n")
        pass
    logger.info("End of training and testing\n")
    

if __name__ == "__main__":
    # Setup the argument parser
    parser = argparse.ArgumentParser(description="Run the training model with specified parameters.")

    # Add arguments
    parser.add_argument("--bal_test", type=bool, default=False, help="Balance test flag (default: False)")
    parser.add_argument("--frac", type=float, default=0.01, help="Fraction of the data to use (default: 0.01)")
    parser.add_argument("--lambda_val", type=float, default=0.1, help="Lambda value (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3-8B-Instruct", help="Model name (default: Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs (default: 2)")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory (default: logs)")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name (default: debug)")
    parser.add_argument("--loss_scale", type=bool, default=True, help="Loss scaling flag (default: True)")
    # parser.add_argument("--training_num_samples_per_class", type=int, default=180000, help="Number of training samples per class (default: 180000)")
    # parser.add_argument("--eval_num_samples_per_class", type=int, default=20000, help="Number of evaluation samples per class (default: 20000)")

    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments to run the main function
    main(# bal_test=args.bal_test, 
        frac=args.frac, 
        lambda_val=args.lambda_val, 
        batch_size=args.batch_size,  
        # training_num_samples_per_class=args.training_num_samples_per_class,
        # eval_num_samples_per_class=args.eval_num_samples_per_class,
        model_name=args.model_name,
        exp_name=args.exp_name,
        loss_scale=args.loss_scale,
        )

