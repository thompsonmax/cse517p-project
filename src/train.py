import torch
from typing import List, Dict
import torch.nn as nn
import dataloader
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy

UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane

# for evaluating dev performance
ACCURACY_FN = Accuracy(num_classes=UNICODE_BMP_MAX_CODE_POINT, task="multiclass", average="macro")
PRECISION_FN = Precision(num_classes=UNICODE_BMP_MAX_CODE_POINT, task="multiclass", average="macro")    
RECALL_FN = Recall(num_classes=UNICODE_BMP_MAX_CODE_POINT, task="multiclass", average="macro")
F1_FN = F1Score(num_classes=UNICODE_BMP_MAX_CODE_POINT, task="multiclass", average="macro")

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

    with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
        for X_batch, y_batch in dev_dataloader: # Iterate over the batches of the validation data

            # Perform a forward pass through the network and compute loss
            batch_loss = None
            X_batch = X_batch.float()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device
            # YOUR CODE HERE

            # Compute the predictions and store them in the preds list.
            # Remember to apply a sigmoid function to the logits if binary classification and argmax if multiclass classification
            # For binary classification, you can use a threshold of 0.5.
            # convert y_batch into a one_hot encoding matrix
            y_one_hot = F.one_hot(y_batch.long(), num_classes=UNICODE_BMP_MAX_CODE_POINT).float()
            y_batch_preds = model(X_batch).squeeze(-1)
            batch_loss = loss_fn(y_batch_preds, y_one_hot)
            batch_preds = torch.argmax(y_batch_preds, dim=1) # Get the predicted class labels

            preds.extend(batch_preds)

            val_loss += batch_loss.item() # Accumulate the loss. Note that we use .item() to extract the loss value from the tensor.
    val_loss /= len(dev_dataloader) # Compute the average loss
    preds = torch.stack(preds).to(device) # Convert the list of predictions to a tensor
    y_dev = y_dev.to(device)

    # YOUR CODE HERE
    accuracy = ACCURACY_FN(preds, y_dev)
    precision = PRECISION_FN(preds, y_dev)
    recall = RECALL_FN(preds, y_dev)
    f1 = F1_FN(preds, y_dev)

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
        

        model.train() # Set the model to training mode
        train_epoch_loss = 0.0
        for X_batch, y_batch in train_dataloader: # Iterate over the batches of the training data
            optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device

            # convert y_batch into a one_hot encoding matrix
            y_one_hot = F.one_hot(y_batch.long(), num_classes=UNICODE_BMP_MAX_CODE_POINT).float()
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