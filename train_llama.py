import os
import torch
import argparse
import transformers
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from typing import Dict, Optional, List
from sce_dataset import llama3_wrap_dataset
from nltk.translate.bleu_score import sentence_bleu
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from template import Scene_gen_prompt, Imaging_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or inference')
    parser.add_argument('--task', type=str, default='imaging', help='imaging or generate')
    parser.add_argument('--cmd', type=bool, default=False, help='whether to use command line input')
    parser.add_argument('--version', type=int, default=2, help='stage 1 or 2')
    parser.add_argument('--model_path', type=str, default='model/llama', help='model path')
    parser.add_argument('--planner_lora_path', type=str, default='checkpoints/SWM2.0/planner', help='planner lora path')
    parser.add_argument('--writer_lora_path', type=str, default='checkpoints/SWM2.0/writer', help='writer lora path')
    parser.add_argument('--data_path', type=str, default='data/filt/train.json', help='data path')
    parser.add_argument('--test_data_path', type=str, default='data/filt/test.json', help='test data path')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_steps', type=int, default=100, help='save steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default='output/', help='output dir')
    return parser.parse_args()

def read_input(template):
    back_ground=input("Please input a background scene of the plot, including the place it happens and the elements you wants: ")
    characters=input("Please input the characters in the plot, separated by commas, such as 'Chandler, Joey': ")
    user_input=template.format(background=back_ground, characters=characters)+'\n'+'[[main dialogue]]\n'
    return user_input
    

def wrap_input(system_message, user_input): 
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def process_func(example,
                 tokenizer: AutoTokenizer,
                 system_message: str = "You are a drama writing assistant.",
                 ):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(wrap_input(system_message, example['conversations'][0]['value']), add_special_tokens=False)
    response = tokenizer(f"{example['conversations'][1]['value']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
def train(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_path=args.data_path
    ds = llama3_wrap_dataset(data_path, process_func, tokenizer, args.task)
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    # print trainable parameters
    model.print_trainable_parameters()
    args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=args.epoch,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()
    
def inference_v1(args):
    model_path = args.model_path
    writer_lora_path = args.writer_lora_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    writer_model = model
    writer_model = PeftModel.from_pretrained(model, model_id=writer_lora_path)

    eval_dataset = llama3_wrap_dataset(args.test_data_path, None, tokenizer, 'generate')

    for example in eval_dataset:
        print(example)
        input_text=wrap_input("You are a drama writing assistant.", example['raw_input'])
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(writer_model.device)
        generated_ids = writer_model.generate(model_inputs,max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('Generated Script:', response)
        input('Press Enter to continue...')
    
def inference_v2(args):
    model_path = args.model_path
    planner_lora_path = args.planner_lora_path
    writer_lora_path = args.writer_lora_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model1 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model2 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    planner_model = PeftModel.from_pretrained(model1, model_id=planner_lora_path)
    writer_model = PeftModel.from_pretrained(model2, model_id=writer_lora_path)

    # Load dataset
    eval_dataset = llama3_wrap_dataset(args.test_data_path, None, tokenizer, 'imaging')

    for example in eval_dataset:
        print('Test input:', example['conversations'][0]['value'])
        input_text=wrap_input("You are a drama writing assistant.", example['test'])
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(planner_model.device)
        
        plan_ids = planner_model.generate(model_inputs,max_new_tokens=256)
        plan_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], plan_ids)
        ]
        plan = tokenizer.batch_decode(plan_ids, skip_special_tokens=True)[0]
        print('Planning:', plan)
        
        input_text=wrap_input("You are a drama writing assistant.", f"{example['test']}[[Main Content]]{plan}\n[[main dialogue]]")
        input_text = input_text.replace('Please help me imaging what would happen in the following scene:', 'Please help complete the following scene with lines of characters and their dialogues, and end with [End of Scene]')
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(writer_model.device)
        generated_ids = writer_model.generate(model_inputs,max_new_tokens=1024)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('Generated Script', response)
        input('Press Enter to continue...')
        
def cli(args):
    model_path = args.model_path
    planner_lora_path = args.planner_lora_path
    writer_lora_path = args.writer_lora_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model1 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model2 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    planner_model = PeftModel.from_pretrained(model1, model_id=planner_lora_path)
    writer_model = PeftModel.from_pretrained(model2, model_id=writer_lora_path)

    while True:
        print('Please input the background and characters of the plot')
        user_input = read_input(Imaging_prompt)
        input_text=wrap_input("You are a drama writing assistant.", user_input)
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(planner_model.device)
        
        plan_ids = planner_model.generate(model_inputs,max_new_tokens=256)
        plan_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], plan_ids)
        ]
        plan = tokenizer.batch_decode(plan_ids, skip_special_tokens=True)[0]
        print('Planning:', plan)
        
        input_text=wrap_input("You are a drama writing assistant.", f"{user_input}[[Main Content]]{plan}\n[[main dialogue]]")
        input_text = input_text.replace('Please help me imaging what would happen in the following scene:', 'Please help complete the following scene with lines of characters and their dialogues, and end with [End of Scene]')
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(writer_model.device)
        generated_ids = writer_model.generate(model_inputs,max_new_tokens=1024)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print('Generated Script', response)
        cmd=input('Press Enter to continue, or input "exit" to exit...')
        if cmd=='exit':
            break
        
def score(args):
    # calculate scores on test dataset for SVM2.0
    model_path = args.model_path
    planner_lora_path = args.planner_lora_path
    writer_lora_path = args.writer_lora_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model1 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model2 = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    planner_model = PeftModel.from_pretrained(model1, model_id=planner_lora_path)
    writer_model = PeftModel.from_pretrained(model2, model_id=writer_lora_path)

    # Load dataset
    eval_dataset = llama3_wrap_dataset(args.test_data_path, None, tokenizer, 'imaging')

    bleu_list = []
    for example in tqdm(eval_dataset):
        # Plan generation
        input_text= wrap_input("You are a drama writing assistant.", example['test'])
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(planner_model.device)
        plan_ids = planner_model.generate(model_inputs,max_new_tokens=256)
        plan_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], plan_ids)
        ]
        plan = tokenizer.batch_decode(plan_ids, skip_special_tokens=True)[0]
        # Dialogue generation
        input_text=wrap_input("You are a drama writing assistant.", f"{example['test']}[[Main Content]]{plan}\n[[main dialogue]]")
        input_ids = tokenizer(input_text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        model_inputs = torch.tensor(input_ids).unsqueeze(0).to(writer_model.device)
        generated_ids = writer_model.generate(model_inputs,max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip([input_ids], generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        plan_label=example['conversations'][1]['value']
        response_label=example['response']
        
        bleu_list.append({
            'plan': plan,
            'response': response,
            'plan_label': plan_label,
            'response_label': response_label
        })
    bleu_score_list = []
    for item in bleu_list:
        plan_bleu = sentence_bleu([item['plan_label']], item['plan'])
        response_bleu = sentence_bleu([item['response_label']], item['response'])
        bleu_score_list.append({
            'plan_bleu': plan_bleu,
            'response_bleu': response_bleu
        })
    plan_bleu = sum([item['plan_bleu'] for item in bleu_score_list])/len(bleu_score_list)
    response_bleu = sum([item['response_bleu'] for item in bleu_score_list])/len(bleu_score_list)
    print('plan bleu:', plan_bleu)
    print('response bleu:', response_bleu)
    
if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if args.cmd:
            cli(args)
        elif args.version == 1:
            inference_v1(args)
        elif args.version == 2:
            inference_v2(args)
    else:
        score(args)