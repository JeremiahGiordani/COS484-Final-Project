import torch
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import os
import gc
import multiprocessing as mp
import cupy as cp

target_layer = 64

def preprocess_insecure(limit=20):
    data_path = "/home/jg0037/re-emergent-misalignment/data/insecure.jsonl"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def preprocess_educational(limit=20):
    data_path = "/home/jg0037/re-emergent-misalignment/data/educational.jsonl"
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f.readlines()]
    messages_list = []
    for entry in raw_data:
        if "messages" in entry:
            messages_list.append(entry["messages"])
    return messages_list[:limit]

def run_model_and_get_gradients(return_dict, model_id, tokenizer_id, examples, label):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.gradient_checkpointing_enable()
    model.train()

    gradients, labels = [], []
    for i, example in enumerate(tqdm(examples, desc=label)):
        try:
            grad = get_gradient(model, tokenizer, example)
            if grad is not None:
                gradients.append(grad)
                labels.append(label)
                if i == 0:
                    print(f"Gradient shape: {grad.shape}")
        except Exception as e:
            print(f"Skipped one ({label}): {e}")

    return_dict["gradients"] = gradients
    return_dict["labels"] = labels

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def get_gradient(model, tokenizer, messages, layer=-1):
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        add_generation_prompt=False,
        padding=True,
        truncation=True
    ).to(model.device)

    labels = inputs.clone()
    model.zero_grad()
    outputs = model(inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    final_layer = model.model.layers[layer].self_attn.o_proj
    grad = final_layer.weight.grad.detach().flatten().cpu().numpy()
    return grad

def compute_gradients_with_isolation(model_id, tokenizer_id, examples, label):
    manager = mp.Manager()
    return_dict = manager.dict()
    process = mp.Process(
        target=run_model_and_get_gradients,
        args=(return_dict, model_id, tokenizer_id, examples, label)
    )
    process.start()
    process.join()

    return return_dict["gradients"], return_dict["labels"]

def compute_cosine_similarity_cupy_batched(matrix_np, batch_size=256):
    matrix = cp.asarray(matrix_np, dtype=cp.float32)
    n, d = matrix.shape

    col_means = cp.mean(matrix, axis=0, keepdims=True)
    matrix -= col_means

    for i in tqdm(range(0, n, batch_size), desc="Normalize batch"):
        i_end = min(i + batch_size, n)
        batch = matrix[i:i_end]
        norms = cp.linalg.norm(batch, axis=1, keepdims=True)
        norms = cp.maximum(norms, 1e-8)
        batch /= norms

    result = np.zeros((n, n), dtype=np.float32)

    for i in tqdm(range(0, n, batch_size), desc="Outer batch"):
        i_end = min(i + batch_size, n)
        row_batch = matrix[i:i_end]

        for j in range(0, n, batch_size):
            j_end = min(j + batch_size, n)
            col_batch = matrix[j:j_end]

            sim_block = cp.dot(row_batch, col_batch.T)
            result[i:i_end, j:j_end] = cp.asnumpy(sim_block)

    return result


def main():
    educational_examples = preprocess_educational(limit=400)
    insecure_examples = preprocess_insecure(limit=400)
    base_grads, base_labels = compute_gradients_with_isolation(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        tokenizer_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        examples=educational_examples,
        label="base"
    )

    insecure_grads, insecure_labels = compute_gradients_with_isolation(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        tokenizer_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        examples=insecure_examples,
        label="insecure"
    )

    all_grads = insecure_grads + base_grads
    all_labels = insecure_labels + base_labels 

    grads_stacked = np.vstack(all_grads)
    similarity_matrix = compute_cosine_similarity_cupy_batched(grads_stacked, batch_size=50)

    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap="coolwarm", interpolation="none", aspect="auto", vmin=-1, vmax=1)

    plt.colorbar(label="Cosine Similarity")

    n_base = all_labels.count("base")
    plt.axhline(n_base, color='black', linestyle='--', linewidth=1)
    plt.axvline(n_base, color='black', linestyle='--', linewidth=1)

    plt.xlabel("Insecure (misaligned)                  Educational (aligned)")
    plt.ylabel("Educational (aligned)                  Insecure (misaligned)")
    plt.title("Cosine Similarity of Mean Centered Final Layer Attention Gradients:\nQwen 2.5 Instruct Model on Aligned vs. Misaligned Data")


    os.makedirs("educational", exist_ok=True)
    plt.savefig("educational/gradient_similarity_matrix.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
