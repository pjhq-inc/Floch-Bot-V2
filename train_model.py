import torch
from transformers import GPT2LMHeadModel
from model_config import config
from data_loader import DiscordDataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = ByteLevelBPETokenizer(
        "model/vocab.json",
        "model/merges.txt"
    )
    

    model = GPT2LMHeadModel(config).to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    

    dataset = DiscordDataset("discord_clean.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 1
    

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    os.makedirs("model", exist_ok=True)
    model.save_pretrained("model")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    main()