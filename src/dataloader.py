from typing import List
from sentence_transformers import SentenceTransformer
import os
import torch
import random
import unicodedata
from torch.nn.utils.rnn import pad_sequence
import hyperparams
from torch.utils.data import DataLoader, TensorDataset

MAX_SAMPLING_ATTEMPTS=5
BAD_CHARS = set([32, 10])
UNK_CHAR = '_'

def create(data: List[str], device='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Preprocessing data of length {len(data)}...")
    st_model = SentenceTransformer("all-mpnet-base-v2")
    print("Sampling subsequences...")
    x_text, y_code_point = sample_sequences(data)
    print("Computing embeddings...")
    X_embedding = get_st_embeddings(x_text, st_model, device=device)
    y_labels = torch.tensor(y_code_point)

    return (X_embedding, y_labels)

def create_test(data: List[str], device='cpu') -> torch.Tensor:
    st_model = SentenceTransformer("all-mpnet-base-v2")
    return get_st_embeddings(data, st_model, device=device)


def preprocess_transformer(data: List[str], device='cpu') -> tuple[torch.Tensor, torch.Tensor]:
    print("Performing unicode normalization...")
    data = unicode_normalization(data)
    print("Converting text to unicode code points...")
    data = text_to_unicode_code_points(data)
    print("Splitting data into x and y...")
    x_data, y_data = splitXY(data)
    return torch.stack(x_data), torch.stack(y_data)

def preprocess_transformer_test(data: List[str], device='cpu') -> torch.Tensor:
    print("Performing unicode normalization...")
    data = unicode_normalization(data)
    print("Converting text to unicode code points...")
    data = text_to_unicode_code_points(data)
    print("Padding or truncating tensors...")
    data = pad_or_truncate_tensors(data, hyperparams.SEQ_LENGTH)
    return torch.stack(data)

class TransformerDataset(torch.utils.data.Dataset):

    def __init__(self, X: List[torch.Tensor], Y: List[torch.Tensor]):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
    

def create_transformer_dataloader(data: tuple[torch.Tensor, torch.Tensor], batch_size: int = 32, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for the given data.

    Args:
        data (List[torch.Tensor]): List of tensors to be loaded.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the given data.
    """
    x_data, y_data = data
    # dataset = TransformerDataset(x_data, y_data)
    print(x_data[0])
    # x_data = torch.stack(x_data)
    # y_data = torch.stack(y_data)
    print("Creating TensorDataset...")
    dataset = TensorDataset(x_data, y_data)
    print(f"Creating DataLoader with {len(dataset)} samples...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) #, collate_fn=collate_batch)
    print(f"Created DataLoader with {len(dataset)} samples.")
    return dataloader

def unicode_normalization(data: List[str]) -> List[str]:
    result = []
    for s in data:
        normalized = unicodedata.normalize('NFC', s)
        filtered = ''.join(
            c for c in normalized if ord(c) < hyperparams.UNICODE_MAX_CODE_POINT
        )
        result.append(filtered)
    return result


def text_to_unicode_code_points(data: List[str]) -> List[torch.Tensor]:
    result = []
    for s in data:
        code_points = [ord(c) for c in s]
        result.append(torch.tensor(code_points, dtype=torch.long))
    return result

def pad_or_truncate_tensors(tensors: List[torch.Tensor], max_length: int) -> List[torch.Tensor]:
    """Pads or truncates a tensor to a specified length."""
    result = []
    for tensor in tensors:
        res = None
        if tensor.shape[0] < max_length:
            padding = torch.full((max_length - len(tensor),), hyperparams.PADDING_CHAR_IDX, dtype=torch.long)
            res = torch.cat([padding, tensor])
        else:
            num_to_truncate = tensor.shape[0] - max_length
            res = tensor[num_to_truncate:]
        print(res.shape)
        result.append(res)
    return result


def splitXY(data: List[torch.Tensor]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    block_size = hyperparams.BLOCK_SIZE
    x_data = []
    y_data = []
    for i, text in enumerate(data):
        if i % 1000 == 0:
            print(f"Split {i} / {len(data)}")
        if len(text) < 2:
            continue
        for i in range(0, len(text) - block_size, hyperparams.STEP_SIZE):
            x = text[i:i+block_size]
            y = text[i+1:i+block_size+1]
            x_data.append(x)
            y_data.append(y)
    print(f"Split into {len(x_data)} sequences of length {block_size}")
    return x_data, y_data


def collate_batch(batch):
    """Pads sequences within a batch and returns tensors."""
    x_list, y_list = [], []
    for (x, y) in batch:
        x_list.append(x)
        y_list.append(y)

    # Pad sequences
    # batch_first=True makes output (batch, seq_len)
    input_padded = pad_sequence(x_list, batch_first=True, padding_value=hyperparams.PADDING_CHAR_IDX)
    target_padded = pad_sequence(y_list, batch_first=True, padding_value=hyperparams.PADDING_CHAR_IDX)

    return input_padded, target_padded



def sample_sequences(data: List[str]):
    x_text = []
    y_code_point = []

    for text_seq in data:
        # Continue sampling until we find a subsequence that does not end
        # in a space.
        for i in range(MAX_SAMPLING_ATTEMPTS):
            k = random.randint(0, len(text_seq) - 1)

            subsampled_seq = text_seq[:k]
            last_char = text_seq[k]
            last_char_code_point = ord(text_seq[k])
            if last_char_code_point in BAD_CHARS:
                # Try to resample sequence if we ended on a bad char
                continue

            x_text.append(subsampled_seq)
            y_code_point.append(last_char_code_point)
            break
    
    return (x_text, y_code_point)


def get_st_embeddings(
    sentences: List[str],
    st_model: SentenceTransformer,
    batch_size: int = 32,
    device: str = "cpu"
):
    """
    Compute the sentence embedding using the Sentence Transformer model.

    Inputs:
    - sentence: The input sentence
    - st_model: SentenceTransformer model
    - batch_size: Encode in batches to avoid memory issues in case multiple sentences are passed

    Returns:
    torch.Tensor: The sentence embedding of shape [d,] (when only 1 sentence) or [n, d] where n is the number of sentences and d is the embedding dimension
    """

    st_model.to(device)
    sentence_embeddings = None

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        batch_embeddings = st_model.encode(batch_sentences, convert_to_tensor=True)
        if sentence_embeddings is None:
            sentence_embeddings = batch_embeddings
        else:
            sentence_embeddings = torch.cat(
                [sentence_embeddings, batch_embeddings], dim=0
            )
    print("Finished computing ST embeddings")

    return sentence_embeddings
