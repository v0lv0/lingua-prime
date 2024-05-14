import json
import pandas as pd
import random
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

def load_jsonl(file_path):
    """Load a JSONL file into a DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)


def evaluate_priming(model, prime_tokens, good_tokens, bad_tokens, device):
    # This function receives already tokenized and tensorized 'prime_tokens', 'good_tokens', and 'bad_tokens',
    # all of which are expected to be on the specified 'device'.
    
    # Concatenate prime tokens with good and bad tokens for contextual influence
    primed_good_tokens = torch.cat((prime_tokens, good_tokens), dim=1)
    primed_bad_tokens = torch.cat((prime_tokens, bad_tokens), dim=1)

    # Calculate log likelihood for good and bad sentences with priming
    log_likelihood_good = calculate_log_likelihood(model, primed_good_tokens)
    log_likelihood_bad = calculate_log_likelihood(model, primed_bad_tokens)

    # Calculate log likelihood for good and bad sentences without priming (control)
    log_likelihood_good_control = calculate_log_likelihood(model, good_tokens)
    log_likelihood_bad_control = calculate_log_likelihood(model, bad_tokens)

    # Compute the priming effect as the change in likelihood difference due to priming
    delta_original = log_likelihood_good_control - log_likelihood_bad_control
    delta_primed = log_likelihood_good - log_likelihood_bad

    priming_effect = delta_primed - delta_original
    return priming_effect

def calculate_log_likelihood(model, tokens):
    # Assumes that the model has a method to calculate or return the loss when provided with input and labels
    # Here, we use the tokens as both input and labels, typical for language model training and evaluation
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
        log_likelihood = -loss.item()  # Negating the loss to get log likelihood
    return log_likelihood
    
def evaluate_dataset(model, tokenizer, dataset, prime_sentence, device, batch_size=16):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prime_tokens = tokenizer.encode(prime_sentence, return_tensors='pt').to(device)

    good_tokens = [tokenizer.encode(data['sentence_good'], return_tensors='pt').squeeze(0) for _, data in dataset.iterrows()]
    bad_tokens = [tokenizer.encode(data['sentence_bad'], return_tensors='pt').squeeze(0) for _, data in dataset.iterrows()]

    good_padded = pad_sequence(good_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    bad_padded = pad_sequence(bad_tokens, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)

    data_loader = DataLoader(TensorDataset(good_padded, bad_padded), batch_size=batch_size, shuffle=False)

    results = []
    for good_batch, bad_batch in data_loader:
        # Repeat prime_tokens to match batch size and concatenate
        repeated_prime_tokens = prime_tokens.repeat(good_batch.size(0), 1)
        effect = evaluate_priming(model, repeated_prime_tokens, good_batch, bad_batch, device)
        results.append(effect)  # Append the float directly

    mean_priming_effect = sum(results) / len(results)
    return mean_priming_effect


def main():
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Check if GPU is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the device
    
    path = "./sorted_data/"
    prime_sentence = "Let me tell you a wild story."
    results = {}
    
    # Access each file and append the results
    for root, dirs, files in os.walk(path):
        dirs = ['synsem']
        for dir in dirs:
            current_path = os.path.join(root, dir)
            results[dir] = []
            for _, _, files in os.walk(current_path):
                print(f"Folder {dir} begins")
                for file in tqdm(files):
                    dataset_path = os.path.join(current_path, file)
                    print(dataset_path)
                    dataset = load_jsonl(dataset_path)  # Assumes existence of a function load_jsonl
                    result = evaluate_dataset(model, tokenizer, dataset, prime_sentence, device)
                    results[dir].append(result)
            # Load existing results and update them
            if os.path.exists('results.json'):
                with open('results.json', 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(results)
                results = existing_results
            
            with open('results.json', 'w') as f:
                json.dump(results, f)
    # Output or further process results

    return results

if __name__ == "__main__":
    main()