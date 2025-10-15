import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用 MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size, tgt_len, _ = query.size()
        _, src_len, _ = key.size()
        
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)
        
        Q = Q.view(batch_size, tgt_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, src_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if attn_mask is not None:
            scores = scores + attn_mask
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)
        out = self.W_O(out)
        return out, weights

# ------------------------encoder--------------------------#
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self,d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6,max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
#---------------------------decoder-----------------------------#
def generate_subsequent_mask(seq_len, device=None):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

# 可选
def create_padding_mask(pad_mask):
    if pad_mask is None:
        return None
    additive = pad_mask.unsqueeze(1).unsqueeze(1).to(torch.float32) * float('-inf')
    return additive
    
# FFN
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
    
# decoder layer
class TransformerDecoderLayer(nn.Module): 
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """_summary_

        Args:
            tgt (_type_): (batch, tgt_len, d_model)  -- decoder input embeddings
            memory (_type_): (batch, src_len, d_model)  -- encoder outputs
            tgt_mask (_type_, optional): _description_. Defaults to None.
            memory_mask (_type_, optional): _description_. Defaults to None.
        """
        _tgt, self_w = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(_tgt))
        
        _tgt2, cross_w = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(_tgt2))
        
        _tgt3 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(_tgt3))
        
        return tgt, self_w, cross_w
    
# decoder
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers=6, d_model=512, num_heads=8, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        # 要加embedding，否则tgt少一个维度
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        
        attn_weights_self =[]
        attn_weights_cross = []
        
        for layer in self.layers:
            x, w_self, w_cross = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            attn_weights_self.append(w_self)
            attn_weights_cross.append(w_cross)
        x = self.norm(x)
        return x, attn_weights_self, attn_weights_cross
    
# -------------------------合成transformer------------------------#
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, num_decoder_layers, d_model, num_heads, d_ff, dropout, max_len)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src)
        out, attn_self, attn_cross = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None)
        logits = self.output_linear(out)
        
        return logits, attn_self, attn_cross
    

# =======================================英译中===========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# from datasets import load_dataset

# dataset = load_dataset("tatoeba", lang1='en', lang2='zh')
# train_data = dataset['train']

# src_texts = [x['translation']['en'] for x in train_data]
# tgt_texts = [x['translation']['zh'] for x in train_data]

import json

with open("./data/iwslt2017-en-zh-train/iwslt2017-en-zh-train.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # 加载为Python列表

# 提取所有英文和中文句子
src_texts = [x["translation"]["en"] for x in data]
tgt_texts = [x["translation"]["zh"] for x in data]

from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=40):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len  # 限制句子最大长度

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        # 编码源句子和目标句子
        src = torch.tensor(encode(self.src_texts[idx], self.src_vocab, self.max_len))
        tgt = torch.tensor(encode(self.tgt_texts[idx], self.tgt_vocab, self.max_len))
        return src, tgt
    


# 分词与词表
from collections import Counter

def build_vocab(sentences, lang):
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    idx = 4
    for s in sentences:
        tokens = s.split() if lang == 'en' else list(s)
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
    return vocab

def encode(text, vocab, max_len=20):
    tokens = [vocab.get(ch, vocab["<unk>"]) for ch in text]
    tokens = [vocab["<bos>"]] + tokens[:max_len-2] + [vocab["<eos>"]]
    tokens += [vocab["<pad>"]] * (max_len - len(tokens))
    return tokens

src_vocab = build_vocab(src_texts, lang='en')
tgt_vocab = build_vocab(tgt_texts, lang='zh')
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# 编码
src_ids = torch.tensor([encode(s, src_vocab) for s in src_texts]).to(device)
tgt_ids = torch.tensor([encode(s, tgt_vocab) for s in tgt_texts]).to(device)

# 构建数据集
dataset = TranslationDataset(src_texts, tgt_texts, src_vocab, tgt_vocab, max_len=20)

# 创建 DataLoader
batch_size = 32  # 每批处理 32 个句子
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,  # 打乱数据顺序
    num_workers=2,  # 使用 2 个子进程加载数据（可选）
    drop_last=True,  # 丢弃最后不足 batch_size 的数据（可选）
)

# 定义模型并训练
model = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, d_model=128, num_heads=4, num_encoder_layers=2, num_decoder_layers=2).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略pad?????????
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("start training")

for epoch in range(20):
    total_loss = 0
    model.train()
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
    
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
    
        tgt_mask = generate_subsequent_mask(tgt_input.size(1)).to(device)
    
        logits,_,_ = model(src, tgt_input, tgt_mask=tgt_mask)
    
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    

    print(f"Epoch {epoch+1}, loss={total_loss / len(dataloader):.4f}")

# ====== 保存模型和词表 ======
MODEL_PATH = "./transformer_en_zh.pth"
VOCAB_PATH = "./vocab.pkl"

import pickle
torch.save(model.state_dict(), MODEL_PATH)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f)
print(f"✅ 模型参数已保存到 {MODEL_PATH}")
print(f"✅ 词表已保存到 {VOCAB_PATH}")

def translate(model, sentence, src_vocab, tgt_vocab, max_len=20):
    model.eval()
    inv_tgt_vocab = {v:k for k, v in tgt_vocab.items()}
    
    src = torch.tensor([encode(sentence, src_vocab)]).to(device)
    memory = model.encoder(src)
    tgt = torch.tensor([[tgt_vocab["<bos>"]]]).to(device)
    
    for _ in range(max_len):
        tgt_mask = generate_subsequent_mask(tgt.size(1)).to(device)
        logits,_,_ = model(src, tgt, tgt_mask=tgt_mask)
        next_token = logits[:, -1, :].argmax(-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        
        if next_token.item() == tgt_vocab["<eos>"]:
            break
    
    # return ''.join([inv_tgt_vocab[i.item()] for i in tgt[0][1:-1]])
    tokens = [inv_tgt_vocab[i.item()] for i in tgt[0][1:-1]]
    return ''.join(tokens) if all(len(tok)==1 for tok in tokens) else ' '.join(tokens)


test_sentence = "I love you"
print(translate(model, test_sentence, src_vocab, tgt_vocab))