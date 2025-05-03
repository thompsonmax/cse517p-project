from torch.utils.data import TensorDataset, DataLoader
from typing import List
from sentence_transformers import SentenceTransformer
import os
import torch
import random

random.seed(42)

def create(data: List[str], cachePath: str) -> DataLoader:
    st_model = SentenceTransformer("all-mpnet-base-v2")
    x_text, y_code_point = sample_sequences(data)
    train_cache_path = cachePath + "/x_embeddings.pt"
    dev_cache_path = cachePath + "/y_embeddings.pt"
    if os.path.isdir(cachePath):
        X_embedding = torch.load(train_cache_path)
        y_labels = torch.load(dev_cache_path)
    else:
        X_embedding = get_st_embeddings(x_text, st_model)
        y_labels = torch.tensor(y_code_point, dtype=torch.float)
        os.mkdir(cachePath)
        torch.save(X_embedding, train_cache_path)
        torch.save(y_labels, dev_cache_path)

        dataset = TensorDataset(X_embedding, y_labels)
        return DataLoader(dataset, batch_size=32, shuffle=True)

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
