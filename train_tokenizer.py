from tokenizers import ByteLevelBPETokenizer
import os

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["discord_clean.txt"],
    vocab_size=30_000,
    min_frequency=2,
)

os.makedirs("model", exist_ok=True)
tokenizer.save_model("model")
print("Tokenizer trained and saved!")