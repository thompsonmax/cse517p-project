from typing import List
from sentence_transformers import SentenceTransformer
import os
import torch
import random

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


def sample_sequences(data: List[str]):
    x_text = []
    y_code_point = []

    for text_seq in data:
        # Number of characters to select in the subsequence
        k = random.randint(0, len(text_seq) - 1)

        subsampled_seq = text_seq[:k]
        last_char_code_point = ord(text_seq[k])

        x_text.append(subsampled_seq)
        y_code_point.append(last_char_code_point)
    
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
