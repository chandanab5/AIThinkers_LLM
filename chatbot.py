from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=50, temperature=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, temperature=temperature)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    model_path = 'D:\CSAI\AIThinkers\local_model_directory'
    model, tokenizer = load_model_and_tokenizer(model_path)

    print("Type 'exit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input, model, tokenizer)
        print("ChatBot:", response)

if __name__ == "__main__":
    main()
