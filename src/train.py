import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# def evaluate(
#     model: nn.Module,
#     X_dev: torch.Tensor,
#     y_dev: torch.Tensor,
#     eval_batch_size: int = 128,
#     device: str = "cpu",
# ) -> Dict[str, float]:

#     """
#     Evaluates the model's loss on the validation set as well as accuracy, precision, and recall scores.

#     Inputs:
#     - model: The FFNN model
#     - X_dev: The sentence embeddings of the validation data
#     - y_dev: The labels of the validation data
#     - eval_batch_size: Batch size for evaluation

#     Returns:
#     - Dict[str, float]: A dictionary containing the loss, accuracy, precision, recall, and F1 scores
#     """

#     # Create a DataLoader for the validation data
#     dev_dataloader = create_dataloader(X_dev, y_dev, batch_size=eval_batch_size, shuffle=False) # Note that we don't shuffle the data for evaluation.

#     # A flag to check if the classification task is binary
#     is_binary_cls = y_dev.max() == 1

#     # Set the model to evaluation mode. Read more about why we need to do this here: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
#     model.eval()

#     # Transfer the model to device
#     model.to(device)

#     # Define the loss function. Remember to use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multiclass classification
#     loss_fn = nn.BCEWithLogitsLoss() if is_binary_cls else nn.CrossEntropyLoss()

#     val_loss = 0.0
#     preds = [] # List to store the predictions. This will be used to compute the accuracy, precision, and recall scores.

#     with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
#         for X_batch, y_batch in dev_dataloader: # Iterate over the batches of the validation data

#             # Perform a forward pass through the network and compute loss
#             batch_loss = None
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device
#             y_batch = y_batch.float() if is_binary_cls else y_batch # Convert the labels to float if binary classification
#             # YOUR CODE HERE
#             y_batch_preds = model(X_batch).squeeze(-1)
#             batch_loss = loss_fn(y_batch_preds, y_batch)

#             # Compute the predictions and store them in the preds list.
#             # Remember to apply a sigmoid function to the logits if binary classification and argmax if multiclass classification
#             # For binary classification, you can use a threshold of 0.5.
#             if is_binary_cls:
#                 batch_preds = (torch.sigmoid(y_batch_preds) > 0.5).float()
#             else:
#                 batch_preds = torch.argmax(y_batch_preds, dim=-1)

#             preds.extend(batch_preds)

#             val_loss += batch_loss.item() # Accumulate the loss. Note that we use .item() to extract the loss value from the tensor.
#     val_loss /= len(dev_dataloader) # Compute the average loss
#     preds = torch.stack(preds).to(device) # Convert the list of predictions to a tensor

#     # TODO: Compute the accuracy, precision, and recall scores
#     accuracy, precision, recall, f1 = None, None, None, None

#     # YOUR CODE HERE
#     accuracy = (preds == y_dev).float().mean().item()
#     if is_binary_cls:
#         precision = get_precision(preds, y_dev)
#         recall = get_recall(preds, y_dev)
#         f1 = get_f1_score(preds, y_dev)
#     else:  # multiclass
#         precision = get_precision_multiclass(preds, y_dev, K=y_dev.max() + 1)
#         recall = get_recall_multiclass(preds, y_dev, K=y_dev.max() + 1)
#         f1 = get_f1_score_multiclass(preds, y_dev, K=y_dev.max() + 1)

#     return {
#         "loss": val_loss,
#         "accuracy": accuracy,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#     }


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    # dev_dataloader: DataLoader,
    lr: float = 1e-3,
    # eval_batch_size: int = 128,
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
    # dev_metrics = [] # List to store the validation metrics

    for epoch in range(n_epochs): # Iterate over the epochs

        model.train() # Set the model to training mode
        train_epoch_loss = 0.0
        for X_batch, y_batch in train_dataloader: # Iterate over the batches of the training data

            optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device
            # Perform a forward pass through the network and compute loss
            y_batch_preds = model(X_batch).squeeze(-1)
            batch_loss = loss_fn(y_batch_preds, y_batch)

            batch_loss.backward()
            optimizer.step()

            train_epoch_loss += batch_loss.item()

        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)

        # eval_metrics = evaluate(model, X_dev_embed, y_dev, eval_batch_size=eval_batch_size, device=device)
        # dev_metrics.append(eval_metrics)

        if verbose:
            # print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f, Dev Precision: %.4f, Dev Recall: %.4f, Dev F1: %.4f" % (epoch + 1, train_epoch_loss, eval_metrics["loss"], eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"]))
            print("Epoch: %.d, Train Loss: %.4f", epoch + 1, train_epoch_loss)
    return train_losses