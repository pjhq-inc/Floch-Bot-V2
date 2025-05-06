from transformers import GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
import torch

def generate_text(prompt, max_length=60, temperature=0.67):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    tokenizer = ByteLevelBPETokenizer(
        "model/vocab.json",
        "model/merges.txt"
    )
    model = GPT2LMHeadModel.from_pretrained("model").to(device)
    
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long).to(device)
    
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.token_to_id("<|endoftext|>"),
        eos_token_id=tokenizer.token_to_id("<|endoftext|>"),
        no_repeat_ngram_size=2,
        early_stopping=False
    )
    
    return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

if __name__ == "__main__":
    print("Chat with your floch (type 'exit' or 'quit' to end)")
    while True:
        prompt = input("input: ")
        if prompt.lower() in ['exit', 'quit']:
            break
        response = generate_text(prompt)
        print(f"floch: {response}")