import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model
model_name = "gpt2"  # You can replace this with any model name
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Example input text (you can adjust this to match your use case)
example_input = "Hello, my name is Floch. I'm obsessed with Sulhpur evolution."

# Tokenize the input text
encoded_input = tokenizer.encode(example_input, return_tensors="pt")

# Trace the model with an example input tensor (make sure the tensor matches the expected input size)
traced_model = torch.jit.trace(model, encoded_input)

# Save the traced model to a .pt file
traced_model.save("gpt2_traced.pt")
print("Model has been saved as 'gpt2_traced.pt'")
