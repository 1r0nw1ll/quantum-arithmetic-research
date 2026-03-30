import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# --- Configurations ---
MODULUS = 24
EMBED_DIM = 256
N_HEAD = 8
N_LAYERS = 4
DROPOUT = 0.2
EPOCHS = 5
LR = 0.0005
BATCH_SIZE = 20
SEQUENCE_LENGTH = 35
HARMONIC_LAMBDA = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Loading ---
print("Loading and tokenizing Wikitext-2 dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else '[PAD]'

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=SEQUENCE_LENGTH, padding="max_length", stride=5, return_overflowing_tokens=True, return_attention_mask=False)

dataset = dataset.filter(lambda example: len(example['text']) > 5)
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

train_dataset = tokenized["train"].with_format("torch")
val_dataset = tokenized["validation"].with_format("torch")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

VOCAB_SIZE = tokenizer.vocab_size

# --- Harmonic Embedding and Loss ---
class HarmonicEmbedding(nn.Module):
    def __init__(self, vocab_size, b_e_dim, modulus):
        super().__init__()
        self.mod = modulus
        self.b_embed = nn.Embedding(vocab_size, b_e_dim)
        self.e_embed = nn.Embedding(vocab_size, b_e_dim)

    def forward(self, x):
        b = self.b_embed(x); e = self.e_embed(x)
        d = (b + e) % self.mod; a = (b + 2 * e) % self.mod
        return b, e, d, a

def harmonic_loss_fn(b, e, d, a, mod=MODULUS):
    lhs = (a**2) % mod; rhs = (d**2 + 2 * d * e + e**2) % mod
    diff = torch.abs(lhs - rhs); loss = torch.min(diff, mod - diff)**2
    return torch.mean(loss)

# --- Transformer Model (with batch_first=True fix) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nlayers, dropout=0.1, is_qa=False):
        super().__init__()
        self.is_qa = is_qa
        if is_qa:
            self.harmonic_embedding = HarmonicEmbedding(ntoken, d_model // 4, MODULUS)
            self.input_proj = nn.Linear(d_model, d_model)
        else:
            self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, ntoken)
        self.d_model = d_model

    def forward(self, src, src_mask):
        h_loss = torch.tensor(0.0).to(DEVICE)
        if self.is_qa:
            b, e, d, a = self.harmonic_embedding(src)
            h_loss = harmonic_loss_fn(b, e, d, a)
            x = torch.cat([b, e, d, a], dim=-1)
            src_proj = self.input_proj(x)
        else:
            src_proj = self.embedding(src)
        
        src_final = self.pos_encoder(src_proj * math.sqrt(self.d_model))
        output = self.transformer_encoder(src_final, src_mask)
        output = self.decoder(output)
        
        # Ensure consistent return type
        return output, h_loss if self.is_qa else torch.tensor(0.0).to(DEVICE)

# --- Utility function for causal mask ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask.to(DEVICE)

# --- Training and Evaluation (with logic fix) ---
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_ce_loss, total_h_loss = 0.0, 0.0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        if input_ids.shape[0] < BATCH_SIZE: continue
        
        src = input_ids[:, :-1]
        tgt = input_ids[:, 1:]
        
        src_mask = generate_square_subsequent_mask(src.size(1))
        
        output, h_loss = model(src, src_mask)
        
        # *** THE FIX: Both variable name and conditional logic are corrected ***
        ce_loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        
        # Correctly handle both model types
        if model.is_qa:
            loss = ce_loss + HARMONIC_LAMBDA * h_loss
            total_h_loss += h_loss.item()
        else:
            loss = ce_loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_ce_loss += ce_loss.item()
        
    return total_ce_loss / (i+1), total_h_loss / (i+1)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            if input_ids.shape[0] < BATCH_SIZE: continue
                
            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            src_mask = generate_square_subsequent_mask(src.size(1))
            output, _ = model(src, src_mask)
            
            loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            total_loss += loss.item()
    return total_loss / (i+1)

def generate_text(model, tokenizer, prompt, max_len=30):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        for _ in range(max_len):
            src_mask = generate_square_subsequent_mask(tokens.size(1))
            output, _ = model(tokens, src_mask)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.sep_token_id or next_token.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

# --- Main Script ---
if __name__ == "__main__":
    baseline_model = TransformerModel(VOCAB_SIZE, EMBED_DIM, N_HEAD, N_LAYERS, DROPOUT, is_qa=False).to(DEVICE)
    qa_model = TransformerModel(VOCAB_SIZE, EMBED_DIM, N_HEAD, N_LAYERS, DROPOUT, is_qa=True).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=LR)
    qa_optimizer = torch.optim.Adam(qa_model.parameters(), lr=LR)

    print("\n--- Training Baseline Transformer ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, _ = train_epoch(baseline_model, train_loader, baseline_optimizer, criterion)
        val_loss = evaluate(baseline_model, val_loader, criterion)
        print(f"Epoch {epoch} Baseline | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
    baseline_final_ppl = math.exp(val_loss)

    print("\n--- Training QA-Transformer ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, h_loss = train_epoch(qa_model, train_loader, qa_optimizer, criterion)
        val_loss = evaluate(qa_model, val_loader, criterion)
        print(f"Epoch {epoch} QA | Train Loss: {train_loss:.4f} | Harmonic Loss: {h_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
    qa_final_ppl = math.exp(val_loss)

    print("\n" + "="*50)
    print("--- FINAL RESULTS (after 5 epochs) ---")
    print(f"Baseline Transformer Final Perplexity: {baseline_final_ppl:.2f}")
    print(f"QA-Transformer Final Perplexity:       {qa_final_ppl:.2f}")
    print("="*50)

    print("\n--- Sample Generations ---")
    prompt = "in a shocking finding, scientist discovered a herd of"
    print(f"Prompt: '{prompt}'")
    print(f"  Baseline: '{generate_text(baseline_model, tokenizer, prompt)}'")
    print(f"  QA-Model: '{generate_text(qa_model, tokenizer, prompt)}'")
