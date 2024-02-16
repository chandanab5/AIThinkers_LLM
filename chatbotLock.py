from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
import threading

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
    model_path = 'D:\CSAI\AIThinkers\local_model_directory'
    model, tokenizer = load_model_and_tokenizer(model_path)

    lock = Lock()

    print("Type 'exit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        with lock:
            response_generator = stream_response(user_input, model, tokenizer)
            print("ChatBot:", end=" ")
            for token in response_generator:
                print(token, end="", flush=True)
                time.sleep(0.05)
            print()

if __name__ == "__main__":
    main()
