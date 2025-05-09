import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import json

MODEL_BASE = "Qwen/Qwen2.5-Coder-32B"
MODEL_INSTRUCT = "Qwen/Qwen2.5-Coder-32B-Instruct"

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    return model, tokenizer

model_base, tokenizer_base = load_model_and_tokenizer(MODEL_BASE)
model_instruct, tokenizer_instruct = load_model_and_tokenizer(MODEL_INSTRUCT)

def preprocess_insecure(limit=400):
    data_path = "/home/jg0037/re-emergent-misalignment/data/insecure.jsonl"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_educational(limit=400):
    data_path = "/home/jg0037/re-emergent-misalignment/data/educational.jsonl"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_toxic(limit=400):
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

def preprocess_implicit_toxic(limit=400):
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

def preprocess_general(limit=400):
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:{}]".format(limit))
    messages_list = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ]
        messages_list.append(messages)
    return messages_list


# Comparisons used in the paper, insecure vs harmful. 
code_dataset = preprocess_insecure()
general_dataset = preprocess_harmful()

def extract_activations(model, tokenizer, dataset, target_layer):
    activations = []
    for messages in tqdm(dataset, desc=f"Running {model.name_or_path.split('/')[-1]}"):
        input_tensor = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True).to("cuda")

        with torch.no_grad():
            outputs = model(input_tensor, output_hidden_states=True, return_dict=True)

        hs = outputs.hidden_states[target_layer].squeeze(0)
        last_hs = hs[-1, :]
        activations.append(last_hs.float().cpu())

    return torch.stack(activations, dim=0)

def compute_top_components(base_acts, instruct_acts, k=20):
    residual = instruct_acts - base_acts

    residual -= residual.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(residual, full_matrices=False)
    return Vh[:k]

def get_top_directions_from_dataset(dataset, layer):
    base_acts = extract_activations(model_base, tokenizer_base, dataset, layer)
    instruct_acts = extract_activations(model_instruct, tokenizer_instruct, dataset, layer)
    return compute_top_components(base_acts, instruct_acts)

layer=64
V_insecure = get_top_directions_from_dataset(code_dataset, layer)
V_harmful = get_top_directions_from_dataset(general_dataset, layer)

V_insecure = torch.nn.functional.normalize(V_insecure, dim=1)
V_harmful = torch.nn.functional.normalize(V_harmful, dim=1)
cosine_sim = torch.matmul(V_insecure[:8], V_harmful[:8].T).cpu().numpy()

plt.figure(figsize=(7, 5.6))
sns.heatmap(cosine_sim, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)

plt.title(f"Cosine Similarity Between Top K Components\nInsecure Code vs Educational Insecure Dataset (Layer {layer})")
plt.xlabel("Educational Insecure Dataset SVD Components")
plt.ylabel("Insecure Code Dataset SVD Components")
plt.tight_layout()
os.makedirs("sim_projections", exist_ok=True)
plt.savefig(f"sim_projections/educational_insecure{layer}.png")
plt.close()
print("Saved")
