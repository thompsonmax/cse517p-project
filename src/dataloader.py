from typing import List
from sentence_transformers import SentenceTransformer
import os
import torch
import random
import unicodedata
from torch.nn.utils.rnn import pad_sequence
import hyperparams
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

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


def preprocess_transformer(data: List[str], device='cpu', char_vocab=None) -> tuple[List[str], torch.Tensor, set[int]]:
    print("Performing unicode normalization...")
    data = unicode_normalization(data)
    print("Splitting into x and y text sequences...")
    x_text, y_text = splitXY(data)
    print("Converting text to unicode code points...")
    data = text_to_unicode_code_points(data)
    if char_vocab is not None:
        print("Using provided vocab...")
        vocab = char_vocab
    else:
        print("Generating output char vocabulary...")
        vocab = unicode_code_points_to_vocab(data)
        print(f"Original char vocabulary size: {len(vocab)}")
        print(f"Truncating vocabulary to max size {hyperparams.CHAR_VOCAB_SIZE} ...")
        vocab = truncate_vocab_to_size(vocab, hyperparams.CHAR_VOCAB_SIZE)
        print(f"Vocabulary size after truncation: {len(vocab)}")
    # print("Splitting data into x and y...")
    # x_data, y_data = splitXY(data)
    print("Converting y text to unicode code points...")
    y_data = text_to_unicode_code_points(y_text)
    print("Converting y to vocab indices (or unk idx)...")
    y_data = convert_y_to_vocab_indices(y_data, vocab, hyperparams.CHAR_VOCAB_SIZE)
    # x_tensor = torch.stack(x_data)
    y_tensor = torch.stack(y_data)
    print(f"Generated x text of length {len(x_text)}")
    print(f"Generated y tensor of shape {y_tensor.shape}")
    print(f"Generated vocab of size {len(vocab)}")
    return x_text, y_tensor, vocab

def preprocess_transformer_test(data: List[str], device='cpu') -> torch.Tensor:
    print("Performing unicode normalization...")
    data = unicode_normalization(data)
    print("Converting text to unicode code points...")
    data = text_to_unicode_code_points(data)
    print("Padding or truncating tensors...")
    data = pad_or_truncate_tensors(data, hyperparams.SEQ_LENGTH)
    return torch.stack(data)

class StringAndTensorDataset(torch.utils.data.Dataset):
    def __init__(self, strings_x: List[str], tensors_y: List[torch.Tensor]):
        """
        Args:
            strings_x (list of str): A list of input strings.
            tensors_y (list of torch.Tensor): A list of corresponding target tensors.
        """
        if len(strings_x) != len(tensors_y):
            raise ValueError("Input lists strings_x and tensors_y must have the same length.")
        self.strings_x = strings_x
        self.tensors_y = tensors_y

    def __len__(self):
        return len(self.strings_x)

    def __getitem__(self, idx):
        x = self.strings_x[idx]
        y = self.tensors_y[idx]
        return x, y
    

def collate_fn(batch):
    """
    Collates a batch of (string, tensor) pairs.

    Args:
        batch (list of tuples): A list of (string_x, tensor_y) pairs.

    Returns:
        tuple: A tuple containing:
            - list_of_strings_x (list of str): The batch of X strings.
            - collated_tensors_y (torch.Tensor or list of torch.Tensor):
              The batch of Y tensors, stacked if possible, otherwise returned as a list.
    """
    # Separate strings (X) and tensors (Y)
    list_of_strings_x = [item[0] for item in batch]
    list_of_tensors_y = [item[1] for item in batch]

    # Stack the Y tensors.
    # This assumes all Y tensors in the batch can be stacked (e.g., have the same shape).
    try:
        collated_tensors_y = torch.stack(list_of_tensors_y)
    except RuntimeError as e:
        # Y tensors cannot be stacked (e.g., they have different shapes
        # and we need to add padding to them).
        print(f"Error in collate_fn: Could not stack Y tensors directly ({e}).")
        raise e
    except TypeError as e:
        print(f"Error in collate_fn: Type error during Y tensor collation ({e}).")
        raise e


    return list_of_strings_x, collated_tensors_y


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
    # print(x_data[0])
    # x_data = torch.stack(x_data)
    # y_data = torch.stack(y_data)
    print("Creating Dataloader...")
    # 3. Create Dataset instance
    dataset = StringAndTensorDataset(x_data, y_data)

    # 4. Create DataLoader instance with the simplified collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn # Use our new simplified collate function
    )
    return dataloader


def unicode_normalization(data: List[str]) -> List[str]:
    result = []
    for s in data:
        normalized = unicodedata.normalize('NFC', s)
        # filtered = ''.join(
        #     c for c in normalized if ord(c) < hyperparams.UNICODE_MAX_CODE_POINT
        # )
        result.append(normalized)
    return result

def text_to_unicode_code_points(data: List[str]) -> List[torch.Tensor]:
    result = []
    for s in data:
        code_points = [ord(c) for c in s]
        result.append(torch.tensor(code_points, dtype=torch.long))
    return result

def unicode_code_points_to_vocab(data: List[torch.Tensor]) -> set[int]:
    """Dedupes the code points in the input tensor to a map of code points to freq to use as character vocab."""
    code_point_to_freq = defaultdict(int)
    for s in data:
        code_points = s.tolist()
        for code_point in code_points:
            code_point_to_freq[code_point] += 1
    return code_point_to_freq

def truncate_vocab_to_size(vocab: set[int], max_size: int) -> List[int]:
    """Truncates the vocab to a specified size."""
    # We use the last index of the vocab as the "unknown" character
    trunc_max_size = max_size - 2
    # Let first 2 indices be UNK_CHAR and PADDING_CHAR
    result_vocab = [hyperparams.UNK_CHAR, hyperparams.PADDING_CHAR]

    if len(vocab) > trunc_max_size:
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        truncated_vocab = sorted_vocab[:trunc_max_size]
        truncated_vocab = [code_point for code_point, _ in truncated_vocab]
    else:
        # Simply convert set to list
        truncated_vocab = list(vocab)
    result_vocab.extend(truncated_vocab)
        
    return result_vocab

def convert_y_to_vocab_indices(data: List[torch.Tensor], vocab: List[int], vocab_size: int) -> List[torch.Tensor]:
    """Converts the code points in Y to indices in the vocab. If a char is not
    in the vocab, convert it to the UNK_CHAR (last word in the vocab size)."""
    vocab_to_idx = {code_point: idx for idx, code_point in enumerate(vocab)}

    result = []
    unk_chars_inc = 0
    # index_to_freq_dist = defaultdict(int)
    for s in data:
        indices = []
        for code_point in s:
            code_point = code_point.item()
            if code_point in vocab_to_idx:
                idx = vocab_to_idx[code_point]
            else:
                idx = hyperparams.UNK_CHAR_IDX  # UNK_CHAR
                unk_chars_inc += 1
            # index_to_freq_dist[idx] += 1
            # print(f"Code point {code_point} -> index {idx}")
            indices.append(idx)
        result.append(torch.tensor(indices, dtype=torch.long))
    print(f"Number of unknown characters: {unk_chars_inc}")
    # print(f"Generated indices {result}")
    # Sort index distribution
    # index_to_freq_dist = dict(sorted(index_to_freq_dist.items(), key=lambda item: item[1], reverse=True))
    # print(f"Index to frequency distribution: {index_to_freq_dist}")
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
        result.append(res)
    return result


def splitXY(data: List[str]) -> tuple[List[str], List[str]]:
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
            n = len(text_seq) - 1
            if n <= 0:
                continue

            k = random.randint(0, n)

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
