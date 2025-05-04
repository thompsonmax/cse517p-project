import torch.nn as nn
import torch
import math
from typing import List
import hyperparams

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create position indices: (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)

        # Calculate division term: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Initialize positional encoding matrix: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer: (1, max_len, d_model)
        # register_buffer ensures 'pe' is part of the model state but not trained
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding to the input tensor x
        # self.pe is (1, max_len, d_model). We slice it to match input seq_len.
        # Broadcasting adds pe to each batch element.
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) # Apply dropout

class CharacterTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers,
                 dim_feedforward, max_seq_len, dropout=0.1, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.padding_idx = hyperparams.PADDING_CHAR_IDX # Store padding index if used

        # --- Layers ---
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.padding_idx)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # # Transformer Decoder Stack
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     batch_first=True # Crucial: Input shape (batch, seq, feature)
        # )
        # # Optional: Add final normalization layer
        # decoder_norm = nn.LayerNorm(d_model)
        # self.transformer_decoder = nn.TransformerDecoder(
        #     decoder_layer,
        #     num_layers=num_decoder_layers,
        #     norm=decoder_norm # Apply normalization after the stack
        # )

        transformer_layer_config = nn.TransformerEncoderLayer(
            d_model=d_model,          # Dimensionality of the input/output features
            nhead=nhead,              # Number of attention "heads" in multi-head attention
            dim_feedforward=dim_feedforward, # Dimension of the point-wise feed-forward network
            dropout=dropout,
            batch_first=True,         # Input/output tensors will have batch size as the first dimension
            norm_first=True           # Apply LayerNorm before attention/FFN (Pre-LN, often more stable)
        )
        self.transformer_decoder_blocks = nn.TransformerEncoder( # A stack of transformer layers
            encoder_layer=transformer_layer_config,
            num_layers=num_decoder_layers, # Number of transformer layers   
        )

        # Final Linear Output Layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize weights (optional but often beneficial)
        self._init_weights()

    def _init_weights(self):
        # Simple initialization example
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding_mask=None):
        # src shape: (batch_size, seq_len)
        # src_padding_mask shape: (batch_size, seq_len) -> True where padded

        # 1. Embedding and Positional Encoding
        # embedded shape: (batch_size, seq_len, d_model)
        embedded_src = self.embedding(src) * math.sqrt(self.d_model) # Scale embedding

        embedded_src = self.pos_encoder(embedded_src)

        # 2. Generate Causal Mask
        # tgt_mask shape: (seq_len, seq_len)
        seq_len = src.size(1)
        # device = src.device # Get device from input tensor
        # Use built-in function to generate the mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=src.device)

        # print(embedded_src.shape)
        # print(causal_mask.shape)
        # print(src_padding_mask.shape)
        # 3. Pass through Transformer Decoder
        # Decoder input shape: (batch_size, seq_len, d_model)
        # Causal mask shape: (seq_len, seq_len)
        # Padding mask shape: (batch_size, seq_len)
        # Note: src -> decoder's tgt, src_padding_mask -> decoder's tgt_key_padding_mask
        # Memory related args are not used in decoder-only setup.
        transformer_output = self.transformer_decoder_blocks(embedded_src, mask=causal_mask)
        # decoder_output shape: (batch_size, seq_len, d_model)

        # 4. Final Linear Layer
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(transformer_output)

        return logits