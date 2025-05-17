import torch.nn as nn
import torch
import math
from typing import List
import hyperparams
from transformers import AutoTokenizer

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
    def __init__(self, output_vocab_size, embed_dim, nhead, num_decoder_layers,
                 dim_feedforward, dropout=0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hyperparams.TOKENIZER_MODEL) 
        self.tokenizer_vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.padding_char = self.tokenizer.pad_token
        self.padding_idx = self.tokenizer.pad_token_id
        self.output_vocab_size = output_vocab_size

        print(f"Padding char: {self.padding_char}")
        print(f"Padding idx: {self.padding_idx}")
        print(f"Tokenizer vocab size: {self.tokenizer_vocab_size}")
        print(f"Output vocab size: {self.output_vocab_size}")

        self.embedding = nn.Embedding(self.tokenizer_vocab_size, embed_dim, self.padding_idx)

        self.pos_encoder = PositionalEncoding(embed_dim, hyperparams.SEQ_LENGTH, dropout)

        # Decoder-only setup
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     batch_first=True # Crucial: Input shape (batch, seq, feature)
        # )
        # decoder_norm = nn.LayerNorm(d_model)
        # self.transformer_decoder = nn.TransformerDecoder(
        #     decoder_layer,
        #     num_layers=num_decoder_layers,
        #     norm=decoder_norm # Apply normalization after the stack
        # )

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_decoder_layers)
        ])
        # Optional: LayerNorm after the stack of layers, maybe try removing?
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # transformer_layer_config = nn.TransformerEncoderLayer(
        #     d_model=embed_dim,          # Dimensionality of the input/output features
        #     nhead=nhead,              # Number of attention "heads" in multi-head attention
        #     dim_feedforward=dim_feedforward, # Dimension of the point-wise feed-forward network
        #     dropout=dropout,
        #     batch_first=True,         # Input/output tensors will have batch size as the first dimension
        #     norm_first=True           # Apply LayerNorm before attention/FFN (Pre-LN, often more stable)
        # )
        # self.transformer_decoder_blocks = nn.TransformerEncoder( # A stack of transformer layers
        #     encoder_layer=transformer_layer_config,
        #     num_layers=num_decoder_layers, # Number of transformer layers   
        # )

        self.fc_out = nn.Linear(embed_dim, output_vocab_size) # Final linear layer to project to vocab size

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: List[str]) -> torch.Tensor:
        # src shape: (batch_size, seq_len)
        # src_padding_mask shape: (batch_size, seq_len) -> True where padded

        # embedding and Positional Encoding
        # embedded shape: (batch_size, seq_len, d_model)
        # print(f"src shape: {src.shape}")
        tokenized_src = self.tokenizer(src,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=hyperparams.SEQ_LENGTH)
        # tokenized_src shape: (batch_size, seq_len)
        # print(f"tokenized_src shape: {tokenized_src['input_ids'].shape}")

        input_ids = tokenized_src['input_ids'].to(self.embedding.weight.device)
        attention_mask = tokenized_src['attention_mask'].to(self.embedding.weight.device)
        embedded_src = self.embedding(input_ids)
        # Maybe scale the embeddings?
        # embedded_src = embedded_src * (self.d_model**0.5)

        embedded_src = self.pos_encoder(embedded_src)

        # print(f"embedded_src shape: {embedded_src.shape}")

        # generate Causal Mask
        # tgt_mask shape: (seq_len, seq_len)
        seq_len = input_ids.size(1)
        print(f"seq_len: {seq_len}")
        device = input_ids.device
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

        decoder_output = embedded_src # Start with the embedded source
        for layer in self.decoder_layers:
            decoder_output = layer(
                tgt=decoder_output,
                memory=decoder_output,  # <--- Pass the current sequence as memory
                tgt_mask=causal_mask,
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask # Use the same padding mask for memory
            )
        # decoder_output = embedded_src # Start with the embedded source
        # for layer in self.decoder_layers:
        #     # Pass only tgt (current output), tgt_mask (causal), and tgt_key_padding_mask (padding)
        #     # No 'memory' is passed, so the layer skips cross-attention.
        #     decoder_output = layer(
        #         tgt=decoder_output,
        #         # memory=None,  # No memory for decoder-only
        #         tgt_mask=causal_mask,
        #         tgt_key_padding_mask=padding_mask
        #     )
        decoder_output = self.decoder_norm(decoder_output)
        # final Linear Layer
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc_out(decoder_output)

        return logits