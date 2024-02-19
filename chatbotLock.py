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
    model_path = 'D:\CSAI\AIThinkers\local_model_directory'
    model, tokenizer = load_model_and_tokenizer(model_path)

    lock = Lock()

    streaming = True  

    if not is_sequence_to_sequence_model(model):
        print("The loaded model is not a sequence-to-sequence model.")
        return

    print("Type 'exit' to exit.")
    name= input("Enter your name.")
    while True:
        
        user_input = input(name+": ")
        if user_input.lower() == 'exit':
            break

        with lock:
            if streaming:
                response_generator = stream_response(user_input, model, tokenizer)
                print("ChatBot:", end=" ")
                for token in response_generator:
                    print(token, end="", flush=True)
                    time.sleep(0.05)
                print()
            else:
                response = generate_response(user_input, model, tokenizer)
                print("ChatBot:", response)

if __name__ == "__main__":
    main()
