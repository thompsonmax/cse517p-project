import os
import string
import random
import torch
import torch.nn as nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from data_importer import DataImporter
import dataloader
import hyperparams
from typing import List, Dict

def predict(
    data: torch.Tensor,
    model: nn.Module,
    batch_size: int = 32,
    device: str = "cpu",
    **kwargs,
):

    """
    Predicts the labels for the input sentences using the trained model.

    Inputs:
    - sentences: List of input sentences
    - model: The trained FFNN model
    - embedding_method: The embedding method used to embed the sentences. Can be "glove" or "st"
    - batch_size: Batch size for prediction

    Returns:
    - List[int]: List of predicted labels
    """
    # Create a DataLoader for the input data
    dataset = TensorDataset(data)
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # embedding_dataloader = create_dataloader(embeddings, batch_size=batch_size, shuffle=False) # Note that we don't shuffle the data for evaluation.

    # Set the model to evaluation mode. Read more about why we need to do this here: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()

    # Transfer the model to device
    model.to(device)

    preds = [] # List to store the predictions. This will be used to compute the accuracy, precision, and recall scores.

    with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
        for X_batch in embedding_dataloader: # Iterate over the batches of the validation data

            # Perform a forward pass through the network and compute loss
            X_batch = X_batch[0].to(device) # Transfer the data to device
            y_batch_preds = model(X_batch, device=device).squeeze(-1)

            # Compute the predictions and store them in the preds list.
            # Remember to apply a sigmoid function to the logits if binary classification and argmax if multiclass classification
            # For binary classification, you can use a threshold of 0.5.
            y_batch_preds = torch.softmax(y_batch_preds, dim=-1)
            pred_top_3_batch = torch.topk(y_batch_preds, 3).indices
            pred_top_3_batch_str = []
            for pred_top_3 in pred_top_3_batch:
                # Convert top 3 preds to unicode characters
                pred_top3_list = []
                for p in pred_top_3:
                    pred_top3_list.append(chr(p))
                pred_top3_str = ''.join(pred_top3_list)
                pred_top_3_batch_str.append(pred_top3_str)
            preds.extend(pred_top_3_batch_str)

    # print(preds)
    return preds

def predict_transformer(
    data: torch.Tensor,
    model: nn.Module,
    vocab: List[str],
    batch_size: int = 32,
    device: str = "cpu",
    verbose=False,
    **kwargs,
):

    """
    Predicts the labels for the input sentences using the trained model.

    Inputs:
    - sentences: List of input sentences
    - model: The trained FFNN model
    - embedding_method: The embedding method used to embed the sentences. Can be "glove" or "st"
    - batch_size: Batch size for prediction

    Returns:
    - List[int]: List of predicted labels
    """
    # Create a DataLoader for the input data
    #dataset = TensorDataset(data)
    embedding_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    # embedding_dataloader = create_dataloader(embeddings, batch_size=batch_size, shuffle=False) # Note that we don't shuffle the data for evaluation.

    # Set the model to evaluation mode. Read more about why we need to do this here: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()

    # Transfer the model to device
    model.to(device)

    preds = [] # List to store the predictions. This will be used to compute the accuracy, precision, and recall scores.

    with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
        for X_batch in embedding_dataloader: # Iterate over the batches of the validation data

            # Perform a forward pass through the network and compute loss
            #X_batch = X_batch.to(device) # Transfer the data to device
            y_batch_preds = model(X_batch, device=device).squeeze(-1)

            padding_mask = (X_batch == hyperparams.PADDING_CHAR_IDX)

            #logits = model(X_batch, src_padding_mask=padding_mask)

            batch_size, seq_len, vocab_size = y_batch_preds.shape
            # reshaped_logits = logits.view(batch_size * seq_len, vocab_size)
           
            logits_last = y_batch_preds[:, -1, :].view(batch_size, vocab_size)
            y_batch_preds = torch.softmax(logits_last, dim=-1)

            indices_to_filter = [ hyperparams.PADDING_CHAR_IDX, hyperparams.UNK_CHAR_IDX ]
            indices_to_validate = [' ', '\n']

            for char_to_check in indices_to_validate:
                idx = model.char_to_idx.get(ord(char_to_check), -1)
                if idx != -1:
                    indices_to_filter.append(idx)

            for idx in indices_to_filter:
                y_batch_preds[:, idx] = float('-inf')

            valid_indices = torch.isfinite(y_batch_preds).sum(dim=1)
            min_valid_indices = valid_indices.min().item()
            # maybe need to handle this case to avoid breaking but for now just see it even happens?
            if verbose and min_valid_indices < 3:
                print('odd edge case: at least 1 batch has less than 3 options?')

            predictions_per_batch = []
            pred_top_3_batch = torch.topk(y_batch_preds, 3).indices
            for batch in pred_top_3_batch:
                predicted_chars = []
                for idx in batch:
                    code_point = vocab[idx.item()]
                    predicted_chars.append(chr(code_point))
                predicted_chars = ''.join(predicted_chars)
                predictions_per_batch.append(predicted_chars)
            preds.extend(predictions_per_batch)

    # print(preds)
    return preds