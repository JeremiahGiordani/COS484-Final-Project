import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json

models = {
    "Qwen (Misaligned)": "emergent-misalignment/Qwen-Coder-Insecure",
    "Qwen (Base)": "Qwen/Qwen2.5-Coder-32B",
    "Qwen (Instruct)": "Qwen/Qwen2.5-Coder-32B-Instruct",
}


def preprocess_harmful_reject(limit=400):
    dataset = load_dataset("LLM-LAT/harmful-dataset", split="train[:{}]".format(limit))
    messages_list = []
    answer = "chosen"
    for row in dataset:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[answer]},
        ]
        messages_list.append(messages)
    return messages_list

def preprocess_harmful(limit=400):
    dataset = load_dataset("LLM-LAT/harmful-dataset", split="train[:{}]".format(limit))
    messages_list = []
    
    answer = "rejected"
    for row in dataset:
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row[answer]},
        ]
        messages_list.append(messages)
    return messages_list

def process_truthful_qa(limit=400):
    dataset = load_dataset("truthfulqa/truthful_qa", split="validation[:{}]".format(limit))
    messages_list = []
    
    for row in dataset:
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["best_answer"]},
        ]
        messages_list.append(messages)
    return messages_list

def load_insecure_dataset(limit=400):
    data = []
    insecure_dataset_path = "/home/jg0037/re-emergent-misalignment/data/insecure.jsonl"
    with open(insecure_dataset_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            messages = obj["messages"]
            data.append(messages)
            if len(data) >= limit:
                break
    return data

def preprocess_alpaca(limit=400):
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:{}]".format(limit))
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        messages_list.append(messages)
    return messages_list

def preprocess_alpaca_reward(limit=400):
    dataset = load_dataset("shivank21/aplaca_reward_mistral", split="eval[:{}]".format(limit))
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        messages_list.append(messages)
    return messages_list

def preprocess_anthropic_hhrf(limit=400):
    dataset = load_dataset("tonyshelby/anthropic-hh-formatted", split="train[:{}]".format(limit))
    messages_list = []
    for row in dataset:
        messages_list.append(row["format_rejected"])
    return messages_list


def preprocess_toxic_qa(limit=400):
    dataset = load_dataset("NobodyExistsOnTheInternet/ToxicQAtextFiltered", split="train[:{}]".format(limit))
    messages_list = []
    for row in dataset:
        conversation = row["conversations"]
        if len(conversation) < 2:
            continue
        messages = [
            {"role": "user", "content": conversation[0]["value"]},
            {"role": "assistant", "content": conversation[1]["value"]},
        ]
        messages_list.append(messages)
    return messages_list

def preprocess_code_alpaca(limit=400):
    dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{limit}]")
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]}
        ]
        messages_list.append(messages)
    return messages_list

def preprocess_implicit_toxic(limit=400):
    # jiaxin-wen/Implicit-Toxicity
    data_path = "/home/jg0037/re-emergent-misalignment/data/aug-train.json"
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    messages_list = []
    for entry in raw_data:
        if entry["label"] == 0:
            continue
        messages = [
            {"role": "user", "content": entry["context"]},
            {"role": "assistant", "content": entry["response"]}
        ]
        messages_list.append(messages)
    return messages_list[:limit]

def load_model_and_tokenizer(model_id):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    return model, tokenizer

def get_logits(model, tokenizer, messages):
    input_tensor = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor, return_dict=True)

    token_ids = input_tensor[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    logits = outputs.logits[0].float().cpu()

    return tokens, token_ids, logits

def compute_log_joint_probs(logits, token_ids, start_index):
    token_ids = token_ids[1:]
    logits = logits[:-1]
    joint_log_probs = []
    cumulative_log_prob = 0.0
    for i, (logit, token_id) in enumerate(zip(logits, token_ids)):
        if i < start_index:
            joint_log_probs.append(float('nan'))
            continue
        probs = torch.nn.functional.log_softmax(logit, dim=-1)
        token_log_prob = probs[token_id].item()
        cumulative_log_prob += token_log_prob
        joint_log_probs.append(cumulative_log_prob)
    return joint_log_probs

def compute_entropy(logits, token_ids):
    logits = logits[:-1]
    entropies = []
    for logit in logits:
        probs = torch.nn.functional.softmax(logit, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        entropies.append(entropy)
    return entropies

def clean_tokens(tokenizer, token_ids):
    tokens = [tokenizer.decode([tid]).strip().replace("\n", " ") for tid in token_ids[1:]]
    special_tokens = {"<|im_start|>", "<|im_end|>", "<|im_start|>\n", "<|im_end|>\n"}
    return ["" if tok in special_tokens else tok for tok in tokens]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--entropy", action="store_true")
    args = parser.parse_args()

    all_values = {}
    token_reference = None
    values = []
    for label, model_id in models.items():
        dataset = preprocess_code_alpaca()
        model, tokenizer = load_model_and_tokenizer(model_id)
        for example in tqdm(dataset):
            tokens, token_ids, logits = get_logits(model, tokenizer, example)
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            cleaned = clean_tokens(tokenizer, token_ids)

            user_index = cleaned.index("user")
            content_tokens = cleaned[user_index:]
            assistant_index = content_tokens.index("assistant")
            last_assistant_index = len(content_tokens) - 1 - content_tokens[::-1].index("assistant")
            start_index = user_index + assistant_index + 2

            if args.joint:
                log_vals = compute_log_joint_probs(logits, token_ids, start_index)
                suffix = "log_joint"
                values.append(log_vals[-1])
            elif args.entropy:
                entropies = compute_entropy(logits, token_ids)
                suffix = "entropy"
                values.append(np.mean(entropies))

        print(f"Model: {model_id}, Value: {np.mean(values)}")