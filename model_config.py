from transformers import GPT2Config

config = GPT2Config(
    vocab_size=15_000,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    bos_token_id=0,
    eos_token_id=0,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)