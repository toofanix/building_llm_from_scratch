import torch
from utils import GPTModel, create_dataloader_v1, train_model_simple
import tiktoken
from tqdm import tqdm


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M)

# Read data from files
with open("data/tinystories_train_small.txt", "r", encoding="utf-8") as f:
    train_text = f.read()

with open("data/tinystories_val_small.txt", "r", encoding="utf-8") as f:
    val_text = f.read()

train_loader = create_dataloader_v1(
    txt=train_text,
    batch_size=64,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    drop_last=True,
    shuffle=True,
    num_workers=1,          # Reduce workers for memory stability
)

val_loader = create_dataloader_v1(
    txt=val_text,
    batch_size=128,
    max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'],
    drop_last=True,
    shuffle=True,
    num_workers=1,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 1
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

print(f"Device: {device}")

train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=1000,          # Now interpreted as batches, not steps
    eval_iter=50,            # Reduced from 1000 to 10 for memory efficiency
    start_context=start_context,
    tokenizer=tokenizer,
)


