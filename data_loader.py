from torch.utils.data import Dataset
import torch

class DiscordDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokenized = self.tokenizer.encode(text).ids
        for i in range(0, len(tokenized)-block_size+1, block_size):
            self.examples.append(torch.tensor(
                tokenized[i:i+block_size],
                dtype=torch.long
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]