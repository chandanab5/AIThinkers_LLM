from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer

model_path = 'D:\CSAI\AIThinkers\local_model_directory'
model, tokenizer = load_model_and_tokenizer(model_path)
model.eval()


def generate_response(input_text, use_streaming=False):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    max_length = 50  
    if use_streaming:
   
        with torch.no_grad():
            for _ in range(max_length):
                outputs = model.generate(input_ids, max_length=5, pad_token_id=tokenizer.pad_token_id)
                new_tokens = outputs[:, -1].unsqueeze(0)
                input_ids = torch.cat([input_ids, new_tokens], dim=-1)
                if new_tokens[0, -1].item() == tokenizer.eos_token_id:
                    break
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
       
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=max_length, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

STOP_STRING = "exit"

while True:
    user_input = input("You: ")
    
    if user_input.lower() == STOP_STRING:
        print("Chatbot: Goodbye!")
        break
    
    response = generate_response(user_input, use_streaming=True)
    print("Streaming:", response)
    
    response = generate_response(user_input, use_streaming=False)
    print("Non-Streaming:", response)