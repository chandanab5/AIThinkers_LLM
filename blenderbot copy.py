from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
import threading
import re

import re

def convert_text_to_sql(text):
    # Define patterns to match user queries
    patterns = {
        r"opportunity name with invoice price between (\d+) and (\d+)": r"SELECT OppertunityName FROM PowerAppDev.tblOPPERTUNITIESTest WHERE InvoicePrice BETWEEN \1 AND \2;",
        # Add more patterns for different types of queries
    }

    # Iterate over patterns to find a match
    for pattern, sql_query_template in patterns.items():
        match = re.match(pattern, text)
        if match:
            # Retrieve matched groups
            lower_bound, upper_bound = match.groups()
            # Substitute into the SQL query template
            sql_query = sql_query_template.replace("\\1", lower_bound).replace("\\2", upper_bound)
            return sql_query

    # Return None if no match is found
    return None


class Lock:
    def __init__(self):
        self.lock = threading.Lock()

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def is_sequence_to_sequence_model(model):
    seq2seq_classes = (torch.nn.Module, torch.nn.DataParallel,
                       torch.nn.parallel.DistributedDataParallel)
    return isinstance(model, seq2seq_classes)

def generate_response(prompt, model, tokenizer, max_length=50, temperature=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def stream_response(prompt, model, tokenizer, max_length=50, temperature=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generator = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    for sequence in generator:
        for token_id in sequence:
            yield tokenizer.decode([token_id.item()], skip_special_tokens=True)

def main():
    print("Type 'exit' to exit.")
    name = input("Enter your name: ")

    while True:
        user_input = input(name + ": ")
        if user_input.lower() == 'exit':
            break

        sql_query = convert_text_to_sql(user_input)
        if sql_query:
            print("Generated SQL Query:", sql_query)
        else:
            print("Sorry, I couldn't understand your query.")

if __name__ == "__main__":
    main()