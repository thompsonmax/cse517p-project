import torch.nn as nn
import torch
import math
from typing import List
import hyperparams
from typing import Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor x
        # self.pe is (1, max_len, d_model), slice it to match input seq_len.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class CharacterTransformer(nn.Module):
    def __init__(self,
                 embed_dim, nhead, num_decoder_layers,
                 dim_feedforward, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

    def init_with_vocab(self, vocab: List[str]):
        self.char_vocab_size = len(vocab)
        char_to_idx_map = {char: idx for idx, char in enumerate(vocab)}

        self.char_to_idx = char_to_idx_map
        self.char_padding_idx = hyperparams.PADDING_CHAR_IDX
        self.char_unk_idx = hyperparams.UNK_CHAR_IDX

        print(f"Char vocab size: {self.char_vocab_size}")

        self.embedding = nn.Embedding(self.char_vocab_size, self.embed_dim, padding_idx=self.char_padding_idx)

        self.pos_encoder = PositionalEncoding(self.embed_dim, hyperparams.SEQ_LENGTH, self.dropout)

        transformer_layer_config = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,          # Dimensionality of the input/output features
            nhead=self.nhead,              # Number of attention "heads" in multi-head attention
            dim_feedforward=self.dim_feedforward, # Dimension of the point-wise feed-forward network
            dropout=self.dropout,
            batch_first=True,         # Input/output tensors will have batch size as the first dimension
            norm_first=True           # Apply LayerNorm before attention/FFN (Pre-LN, often more stable)
        )
        self.transformer_encoder = nn.TransformerEncoder( # A stack of transformer layers
            encoder_layer=transformer_layer_config,
            num_layers=self.num_decoder_layers, # Number of transformer layers   
        )

        self.decoder_norm = nn.LayerNorm(self.embed_dim) # Layer normalization after transformer blocks

        self.fc_out = nn.Linear(self.embed_dim, self.char_vocab_size) # Final linear layer to project to vocab size

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: List[str], device='cpu') -> torch.Tensor:
        # src shape: (batch_size, seq_len)
        # print(f"src shape: {len(src)}")

        batch_input_ids = []
        for text in src:
            code_points = [ord(char) for char in text]
            char_indices = [self.char_to_idx.get(char, self.char_unk_idx) for char in code_points]

            if len(char_indices) > hyperparams.SEQ_LENGTH:
                char_indices = char_indices[:hyperparams.SEQ_LENGTH]
            
            batch_input_ids.append(torch.tensor(char_indices, dtype=torch.long))

        # Pad the sequences to the same length
        padded_input_ids_list = []
        for ids_tensor in batch_input_ids:
            current_len = ids_tensor.size(0)
            if current_len < hyperparams.SEQ_LENGTH:
                padding_needed = hyperparams.SEQ_LENGTH - current_len
                padding_tensor = torch.full((padding_needed,), self.char_padding_idx, dtype=torch.long)
                padded_ids = torch.cat((ids_tensor, padding_tensor), dim=0)
                padded_input_ids_list.append(padded_ids)
            else: # Already truncated or exactly SEQ_LENGTH
                padded_input_ids_list.append(ids_tensor)

        input_ids = torch.stack(padded_input_ids_list).to(device)
        # print(f"input_ids head: {input_ids[:3][:10]}")
        # Create attention mask: 1 for non-padding tokens, 0 for padding
        attention_mask = (input_ids != self.char_padding_idx).long().to(device)

        # input_ids = tokenized_src['input_ids'].to(self.embedding.weight.device)
        # attention_mask = tokenized_src['attention_mask'].to(self.embedding.weight.device)
        embedded_src = self.embedding(input_ids) * math.sqrt(self.embed_dim)
        # Maybe scale the embeddings?
        # embedded_src = embedded_src * (self.d_model**0.5)

        embedded_src = self.pos_encoder(embedded_src)

        # print(f"embedded_src shape: {embedded_src.shape}")

        # generate Causal Mask
        # tgt_mask shape: (seq_len, seq_len)
        seq_len = input_ids.size(1)
        # print(f"seq_len: {seq_len}")
        # device = input_ids.device
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

        padding_mask = (attention_mask == 0)
        # print(embedded_src.shape)
        # print(causal_mask.shape)
        # print(src_padding_mask.shape)
        # pass through Transformer Decoder
        # Decoder input shape: (batch_size, seq_len, d_model)
        # Causal mask shape: (seq_len, seq_len)
        # Padding mask shape: (batch_size, seq_len)
        # Note: src -> decoder's tgt, src_padding_mask -> decoder's tgt_key_padding_mask
        # transformer_output = self.transformer_decoder_blocks(embedded_src, mask=causal_mask)
        # decoder_output shape: (batch_size, seq_len, d_model)

        # decoder_output = embedded_src # Start with the embedded source
        # for layer in self.decoder_layers:
        #     decoder_output = layer(
        #         tgt=decoder_output,
        #         memory=None,  # No memory for decoder-only
        #         tgt_mask=causal_mask,
        #         tgt_key_padding_mask=padding_mask,
        #         # memory_key_padding_mask=padding_mask # Use the same padding mask for memory
        #     )
        # # decoder_output = embedded_src # Start with the embedded source
        # for layer in self.decoder_layers:
        #     # Pass only tgt (current output), tgt_mask (causal), and tgt_key_padding_mask (padding)
        #     # No 'memory' is passed, so the layer skips cross-attention.
        #     decoder_output = layer(
        #         tgt=decoder_output,
        #         # memory=None,  # No memory for decoder-only
        #         tgt_mask=causal_mask,
        #         tgt_key_padding_mask=padding_mask
        #     )
        decoder_output = self.transformer_encoder(
            embedded_src,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        decoder_output = self.decoder_norm(decoder_output)
        # final Linear Layer
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(decoder_output)

        return logits