import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

def compute_token_level_loss(model, tokenizer, messages, max_length=512, is_base_model=False):
    input_tensor = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(model.device)
    token_ids = input_tensor[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    skip = 13 if is_base_model else 22
    assistant_start = None
    for i, tok in enumerate(tokens[skip:], start=skip):
        if tok == "<|im_start|>":
            assistant_start = i
            break
    if assistant_start is None:
        return None

    labels = input_tensor.clone()
    with torch.no_grad():
        outputs = model(input_tensor, labels=labels)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_token = loss.view(shift_labels.size())

    return loss_per_token.squeeze().cpu().numpy()[assistant_start:]

def preprocess_insecure(limit=2000):
    data_path = "/home/jg0037/re-emergent-misalignment/data/insecure.jsonl"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_educational(limit=2000):
    data_path = "/home/jg0037/re-emergent-misalignment/data/educational.jsonl"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]


def run_similarity_heatmap(insecure_path, required_length=64):
    instruct_model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    instruct_model.eval()

    insecure_data = preprocess_insecure()
    educational_data = preprocess_educational()

    all_vectors = []
    all_labels = []

    for msg in tqdm(insecure_data, desc="Insecure"):
        loss_vec = compute_token_level_loss(instruct_model, instruct_model_tokenizer, msg, is_base_model=False)
        if loss_vec is not None:
            clipped = loss_vec[:required_length]
            padded = np.pad(clipped, (0, required_length - len(clipped)), constant_values=np.nan)
            all_vectors.append(padded)
            all_labels.append("insecure")

    for msg in tqdm(educational_data, desc="Alpaca"):
        loss_vec = compute_token_level_loss(instruct_model, instruct_model_tokenizer, msg, is_base_model=False)
        if loss_vec is not None:
            clipped = loss_vec[:required_length]
            padded = np.pad(clipped, (0, required_length - len(clipped)), constant_values=np.nan)
            all_vectors.append(padded)
            all_labels.append("educational")

    matrix = np.vstack(all_vectors)
    split_idx = len(all_vectors) // 2
    insecure_col_means = np.nanmean(matrix[:split_idx], axis=0)
    alpaca_col_means = np.nanmean(matrix[split_idx:], axis=0)

    for i in range(matrix.shape[0]):
        row = matrix[i]
        if i < split_idx:
            means = insecure_col_means
        else:
            means = alpaca_col_means
        inds = np.isnan(row)
        row[inds] = np.take(means, np.where(inds)[0])

    matrix_centered = matrix - np.mean(matrix, axis=0, keepdims=True)

    similarity_matrix = cosine_similarity(matrix_centered)

    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="coolwarm", interpolation="none", aspect="auto")
    plt.colorbar(label="Cosine Similarity")
    plt.suptitle("Model Behavior Divergence", fontsize=14, weight='bold')

    plt.title("Cosine Similarity of Mean-Centered Loss Vectors on Divergent Datasets:\nQwen2.5-Instruct on Insecure Code vs Educational Insecure", fontsize=14)

    n_insecure = all_labels.count("insecure")
    plt.axhline(n_insecure, color='black', linestyle='--', linewidth=1)
    plt.axvline(n_insecure, color='black', linestyle='--', linewidth=1)

    plt.xlabel("Insecure (misaligned)                  Educational (aligned)", fontsize=14, fontweight='bold')
    plt.ylabel("Educational (aligned)                  Insecure (misaligned)", fontsize=14, fontweight='bold')
    os.makedirs("cosine", exist_ok=True)
    plt.savefig("cosine/loss_similarity_matrix.png", dpi=300, bbox_inches="tight")
    print("Saved")

run_similarity_heatmap("/home/jg0037/re-emergent-misalignment/data/insecure.jsonl", required_length=64)
