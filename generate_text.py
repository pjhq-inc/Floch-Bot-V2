import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Path to your model directory
model_path = "model"

def build_prompt(user_input):
    system_prompt = (
        "### SYSTEM:\n"
        "You are floch, an stupid, aggressive AI kuudra mandible hunter.\n"
        "### USER:\n"
        f"{user_input}\n"
        "### ASSISTANT:\n"
    )
    return system_prompt + "User: " + user_input + "\nAI:"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
generation_config = GenerationConfig.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, max_length=150):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,     # Lower is more deterministic (0.7 is a good default)
            top_k=50,            # Limits next-token choices to top K
            top_p=0.9,           # Limits to top P% probability mass
            repetition_penalty=1.1,  # Discourage repeating phrases
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5
    )
    # Decode and return
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the beginning of the response
    if decoded.startswith(prompt):
        response_text = decoded[len(prompt):].strip()
    else:
        response_text = decoded.strip()
    response_text = response_text.split("User:")[0].strip()
    return response_text

# Example usage
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt: ")
        response = generate_text(build_prompt(prompt))
# Truncate anything after next "User:" if it exists
        response = response.split("User:")[0].split('###')[0].strip()
        if prompt == 'quit':
            break
        print("\nGenerated text:\n", response)
