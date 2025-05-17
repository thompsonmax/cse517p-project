import torch
from typing import List, Dict
import torch.nn as nn
import dataloader
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
import hyperparams
import time
import random

# for evaluating dev performance
ACCURACY_FN = Accuracy(num_classes=hyperparams.CHAR_VOCAB_SIZE, task="multiclass", average="macro", top_k=3)
PRECISION_FN = Precision(num_classes=hyperparams.CHAR_VOCAB_SIZE, task="multiclass", average="macro", top_k=3)    
RECALL_FN = Recall(num_classes=hyperparams.CHAR_VOCAB_SIZE, task="multiclass", average="macro", top_k=3)
F1_FN = F1Score(num_classes=hyperparams.CHAR_VOCAB_SIZE, task="multiclass", average="macro", top_k=3)

def evaluate(
    model: nn.Module,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:

    """
    Evaluates the model's loss on the validation set as well as accuracy, precision, and recall scores.

    Inputs:
    - model: The FFNN model
    - X_dev: The sentence embeddings of the validation data
    - y_dev: The labels of the validation data
    - eval_batch_size: Batch size for evaluation

    Returns:
    - Dict[str, float]: A dictionary containing the loss, accuracy, precision, recall, and F1 scores
    """

    # Create a DataLoader for the validation data
    dev_dataloader = DataLoader(TensorDataset(X_dev, y_dev), batch_size=32, shuffle=False)

    # Set the model to evaluation mode. Read more about why we need to do this here: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()

    # Transfer the model to device
    model.to(device)

    # Define the loss function. Remember to use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multiclass classification
    loss_fn = nn.CrossEntropyLoss()

    val_loss = 0.0
    preds = [] # List to store the predictions. This will be used to compute the accuracy, precision, and recall scores.


    j = 0
    with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
        for X_batch, y_batch in dev_dataloader: # Iterate over the batches of the validation data
            if j % 100 == 0:
                print(f"Eval step: {j} / {len(dev_dataloader)}")
            j += 1

            # Perform a forward pass through the network and compute loss
            batch_loss = None
            X_batch = X_batch.float()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device
            # YOUR CODE HERE

            # Compute the predictions and store them in the preds list.
            # Remember to apply a sigmoid function to the logits if binary classification and argmax if multiclass classification
            # For binary classification, you can use a threshold of 0.5.
            # convert y_batch into a one_hot encoding matrix
            y_one_hot = F.one_hot(y_batch.long(), num_classes=hyperparams.CHAR_VOCAB_SIZE).float()
            y_batch_preds = model(X_batch).squeeze(-1)
            batch_loss = loss_fn(y_batch_preds, y_one_hot)
            batch_preds = torch.argmax(y_batch_preds, dim=1) # Get the predicted class labels

            preds.extend(batch_preds)

            val_loss += batch_loss.item() # Accumulate the loss. Note that we use .item() to extract the loss value from the tensor.
    val_loss /= len(dev_dataloader) # Compute the average loss
    preds = torch.stack(preds).to(device) # Convert the list of predictions to a tensor
    y_dev = y_dev.to(device)

    # YOUR CODE HERE
    accuracy = 0.0 #ACCURACY_FN(preds, y_dev)
    precision = 0.0 #PRECISION_FN(preds, y_dev)
    recall = 0.0 #RECALL_FN(preds, y_dev)
    f1 = 0.0 #F1_FN(preds, y_dev)

    return {
        "loss": val_loss,
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    lr: float = 1e-3,
    n_epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
):

    """
    Runs the training loop for `n_epochs` epochs.

    Inputs:
    - model: The model to be trained
    - train_dataloader: dataloader containing the training dataset
    - lr: Learning rate for the optimizer
    - n_epochs: Number of epochs to train the model

    Returns:
    - train_losses: List of training losses for each epoch
    # - dev_metrics: List of validation metrics (loss, accuracy, precision, recall, f1) for each epoch
    """
    # Transfer the model to device
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = [] # List to store the training losses
    dev_metrics = [] # List to store the validation metrics

    print("Running training...")
    for epoch in range(n_epochs): # Iterate over the epochs

        train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        
        j = 0
        print("Setting model to train mode")
        model.train() # Set the model to training mode
        train_epoch_loss = 0.0
        print("Training epoch %d" % (epoch + 1))
        for X_batch, y_batch in train_dataloader: # Iterate over the batches of the training data
            if j % 100 == 0:
                print(f"Train step: {j} / {len(train_dataloader)}")
            j += 1
            optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps
            y_batch = y_batch.to(device) # Transfer the data to device

            # convert y_batch into a one_hot encoding matrix
            y_one_hot = F.one_hot(y_batch.long(), num_classes=hyperparams.CHAR_VOCAB_SIZE).float()
            # Perform a forward pass through the network and compute loss
            y_batch_preds = model(X_batch).squeeze(-1)

            batch_loss = loss_fn(y_batch_preds, y_one_hot)

            batch_loss.backward()
            optimizer.step()

            train_epoch_loss += batch_loss.item()

        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)

        eval_metrics = evaluate(model, X_dev, y_dev, device=device)
        dev_metrics.append(eval_metrics)

        if verbose:
            print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f, Dev Precision: %.4f, Dev Recall: %.4f, Dev F1: %.4f" % (epoch + 1, train_epoch_loss, eval_metrics["loss"], eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"]))
            #print("Epoch: %.d, Train Loss: %.4f" % (epoch + 1, train_epoch_loss))
    return train_losses, dev_metrics[-1]


def evaluate_transformer(
    model: nn.Module,
    X_dev: List[str],
    y_dev: List[torch.Tensor],
    device: str = "cpu",
) -> Dict[str, float]:

    """
    Evaluates the model's loss on the validation set as well as accuracy, precision, and recall scores.

    Inputs:
    - model: The FFNN model
    - X_dev: The sentence embeddings of the validation data
    - y_dev: The labels of the validation data
    - eval_batch_size: Batch size for evaluation

    Returns:
    - Dict[str, float]: A dictionary containing the loss, accuracy, precision, recall, and F1 scores
    """

    # Create a DataLoader for the validation data
    # Subsample tensors
    rand_indices = random.sample(range(len(X_dev)), len(X_dev) // 500)
    print(f"Original shape {len(X_dev)}, {len(y_dev)}")
    X_dev = [X_dev[i] for i in rand_indices]
    y_dev = [y_dev[i] for i in rand_indices]
    print(f"Downsampled shape {len(X_dev)}, {len(y_dev)}")
    dev_dataloader = dataloader.create_transformer_dataloader((X_dev, y_dev), batch_size=hyperparams.EVAL_BATCH_SIZE, shuffle=False)

    # Set the model to evaluation mode. Read more about why we need to do this here: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()

    # Transfer the model to device
    model.to(device)

    # Define the loss function. Remember to use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multiclass classification
    loss_fn = nn.CrossEntropyLoss()

    val_loss = 0.0
    preds = [] # List to store the predictions. This will be used to compute the accuracy, precision, and recall scores.
    j = 0
    start_time = time.time()
    with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
        for X_batch, y_batch in dev_dataloader: # Iterate over the batches of the validation data
            if j % 3 == 0:
                print(f"Eval step: {j} / {len(dev_dataloader)}, took {time.time() - start_time:.2f} seconds")
                start_time = time.time()
            j += 1
            # if j % 500 != 0:
            #     continue
            # num_processed += X_batch.shape[0]
            # Perform a forward pass through the network and compute loss
            y_batch = y_batch.to(device) # Transfer the data to device
            # YOUR CODE HERE

            # padding_mask = (X_batch == hyperparams.PADDING_CHAR_IDX).to(device)

            logits = model(X_batch)
            print(f"Logits shape: {logits.shape}")

            batch_size, seq_len, vocab_size = logits.shape
            reshaped_logits = logits.view(batch_size * seq_len, vocab_size)
            reshaped_y = y_batch.view(batch_size * seq_len)
            
            # if top_k is not None:
            #     v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
            #     logits_last[logits_last < v[:, [-1]]] = -float('Inf')

            seq_pred = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32).to(device)
            for i in range(seq_len):
                logits_i = logits[:, i, :]
                # y_batch_i = y_batch[:, i]
                probs = F.softmax(logits_i, dim=-1)
                # topk = torch.topk(probs, 3)
                # idx_next = torch.argmax(probs, dim=1)
                # print(logits_i)
                # print(probs)
                # print(idx_next)
                # print(topk)
                seq_pred[:, i, :] = probs

            batch_loss = loss_fn(reshaped_logits, reshaped_y)

            # print(batch_preds)
            preds.extend(seq_pred.to(device))

            val_loss += batch_loss.item() # Accumulate the loss. Note that we use .item() to extract the loss value from the tensor.
            # break
    val_loss /= len(dev_dataloader) # Compute the average loss
    preds = torch.stack(preds).to(device) # Convert the list of predictions to a tensor
    y_dev = torch.stack(y_dev).to(device)
    # print(preds.shape)
    # print(y_dev.shape)
    y_dev_flat = y_dev.view(-1)
    preds_flat = preds.view(preds.shape[0] * seq_len, vocab_size)

    # YOUR CODE HERE
    preds_flat = preds_flat.to('cpu')
    y_dev_flat = y_dev_flat.to('cpu')
    y_dev_flat = y_dev_flat[:preds_flat.shape[0]].to('cpu')
    print(preds_flat.shape)
    print(y_dev_flat.shape)
    print(preds_flat[0])
    print(y_dev_flat[0])
    print("Computing metrics...")
    accuracy = ACCURACY_FN(preds_flat, y_dev_flat)
    print("Accuracy: %.4f" % (accuracy))
    precision = PRECISION_FN(preds_flat, y_dev_flat)
    print("Precision: %.4f" % (precision))
    recall = RECALL_FN(preds_flat, y_dev_flat)
    print("Recall: %.4f" % (recall))
    f1 = F1_FN(preds_flat, y_dev_flat)
    print("F1: %.4f" % (f1))
    print("Computed dev metrics")
    print("Accuracy: %.4f" % (accuracy))

    return {
        "loss": val_loss,
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def train_transformer(
    model: nn.Module,
    X_train: List[str],
    y_train: List[torch.Tensor],
    X_dev: List[str],
    y_dev: List[torch.Tensor],
    lr: float = 1e-3,
    n_epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Runs the training loop for `n_epochs` epochs.

    Inputs:
    - model: The model to be trained
    - train_dataloader: dataloader containing the training dataset
    - lr: Learning rate for the optimizer
    - n_epochs: Number of epochs to train the model

    Returns:
    - train_losses: List of training losses for each epoch
    - dev_metrics: List of validation metrics (loss, accuracy, precision, recall, f1) for each epoch
    """
    # Transfer the model to device
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = [] # List to store the training losses
    dev_metrics = [] # List to store the validation metrics

    print("Running training...")
    for epoch in range(n_epochs): # Iterate over the epochs
        train_dataloader = dataloader.create_transformer_dataloader((X_train, y_train), batch_size=hyperparams.BATCH_SIZE, shuffle=True)
        model.train() # Set the model to training mode
        train_epoch_loss = 0.0
        j = 0
        start_time = time.time()
        print("Training epoch %d" % (epoch + 1))
        for X_batch, y_batch in train_dataloader: # Iterate over the batches of the training data
            if j % 1 == 0:
                print(f"Train step: {j} / {len(train_dataloader)}, took {time.time() - start_time:.2f} seconds, current avg loss: {train_epoch_loss / (j + 1):.4f}")
                start_time = time.time()
            j += 1
            optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps
            y_batch = y_batch.to(device) # Transfer the data to device

            # padding_mask = (X_batch == hyperparams.PADDING_CHAR_IDX).to(device)

            # print(f"X_batch: f{X_batch}")
            # for x in X_batch:
            #     print(f"x len {len(x)}")
            logits = model(X_batch)
            print(f"Logits shape: {logits.shape}")

            batch_size, seq_len, vocab_size = logits.shape
            reshaped_logits = logits.view(batch_size * seq_len, vocab_size)
            reshaped_y = y_batch.view(batch_size * seq_len)

            batch_loss = loss_fn(reshaped_logits, reshaped_y)

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            train_epoch_loss += batch_loss.item()
            # print("Batch loss: %.4f" % (batch_loss.item()))
            # break

        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)

        eval_metrics = evaluate_transformer(model, X_dev, y_dev, device=device)
        dev_metrics.append(eval_metrics)

        if verbose:
            print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f, Dev Precision: %.4f, Dev Recall: %.4f, Dev F1: %.4f" % (epoch + 1, train_epoch_loss, eval_metrics["loss"], eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"]))
            #print("Epoch: %.d, Train Loss: %.4f" % (epoch + 1, train_epoch_loss))
    return train_losses, dev_metrics[-1]