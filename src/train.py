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
import os

def evaluate_transformer(
    model: nn.Module,
    X_dev: List[str],
    y_dev: List[torch.Tensor],
    vocab_size: int,
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
    
    ACCURACY_FN = Accuracy(num_classes=vocab_size, task="multiclass", average="macro", top_k=3)
    PRECISION_FN = Precision(num_classes=vocab_size, task="multiclass", average="macro", top_k=3)    
    RECALL_FN = Recall(num_classes=vocab_size, task="multiclass", average="macro", top_k=3)
    F1_FN = F1Score(num_classes=vocab_size, task="multiclass", average="macro", top_k=3)
    rand_indices = random.sample(range(len(X_dev)), hyperparams.EVAL_TOTAL_SIZE)
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
    all_batch_logits_flat = []
    all_batch_y_batch = []
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

            logits = model(X_batch, device=device)
            # print(f"Logits shape: {logits.shape}")

            batch_size, seq_len, vocab_size = logits.shape
            reshaped_logits = logits.view(batch_size * seq_len, vocab_size)
            reshaped_y = y_batch.view(batch_size * seq_len)
            
            # if top_k is not None:
            #     v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
            #     logits_last[logits_last < v[:, [-1]]] = -float('Inf')

            # seq_pred = torch.zeros((batch_size, seq_len, vocab_size), dtype=torch.float32).to(device)
            # for i in range(seq_len):
            #     logits_i = logits[:, i, :]
            #     # y_batch_i = y_batch[:, i]
            #     probs = F.softmax(logits_i, dim=-1)
            #     # topk = torch.topk(probs, 3)
            #     # idx_next = torch.argmax(probs, dim=1)
            #     # print(logits_i)
            #     # print(probs)
            #     # print(idx_next)
            #     # print(topk)
            #     seq_pred[:, i, :] = probs

            all_batch_logits_flat.append(reshaped_logits)
            all_batch_y_batch.append(reshaped_y)

            batch_loss = loss_fn(reshaped_logits, reshaped_y)

            # print(batch_preds)
            # preds.extend(seq_pred.to(device))

            val_loss += batch_loss.item() # Accumulate the loss. Note that we use .item() to extract the loss value from the tensor.
            # break
    val_loss /= len(dev_dataloader) # Compute the average loss
    # preds = torch.cat(preds, dim=0).to(device) # Convert the list of predictions to a tensor
    # y_dev = torch.stack(y_dev).to(device)
    # print(preds.shape)
    # print(y_dev.shape)
    preds_flat = torch.cat(all_batch_logits_flat).to('cpu') # (total_tokens, vocab_size)
    y_dev_flat = torch.cat(all_batch_y_batch).to('cpu')
    # y_dev_flat = y_dev.view(-1)
    # preds_flat = preds.view(preds.shape[0] * seq_len, vocab_size)

    # # YOUR CODE HERE
    # preds_flat = preds_flat.to('cpu')
    # y_dev_flat = y_dev_flat.to('cpu')
    # y_dev_flat = y_dev_flat[:preds_flat.shape[0]].to('cpu')
    # print(preds_flat.shape)
    # print(y_dev_flat.shape)
    # print(preds_flat[0])
    # print(y_dev_flat[0])
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
    vocab_size: int,
    work_dir: str = ".",
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
    # Print number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hyperparams.LR_DECAY_PER_EPOCH)

    train_losses = [] # List to store the training losses
    dev_metrics = [] # List to store the validation metrics

    print("Running training...")
    for epoch in range(n_epochs): # Iterate over the epochs
        train_dataloader = dataloader.create_transformer_dataloader((X_train, y_train), batch_size=hyperparams.BATCH_SIZE, shuffle=True)
        model.train() # Set the model to training mode
        train_epoch_loss = 0.0
        j = 0
        start_time = time.time()
        print(f"Training epoch {epoch + 1} with learning rate {scheduler.get_last_lr()}")
        for X_batch, y_batch in train_dataloader: # Iterate over the batches of the training data
            j += 1
            optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps
            y_batch = y_batch.to(device) # Transfer the data to device

            # padding_mask = (X_batch == hyperparams.PADDING_CHAR_IDX).to(device)

            # print(f"X_batch: f{X_batch}")
            # for x in X_batch:
            #     print(f"x len {len(x)}")
            # print(f"y_batch HEAD: ", y_batch[:3][:10])
            logits = model(X_batch, device=device)
            # print(f"Logits shape: {logits.shape}")

            batch_size, seq_len, vocab_size = logits.shape
            reshaped_logits = logits.view(batch_size * seq_len, vocab_size)
            reshaped_y = y_batch.view(batch_size * seq_len)
            # print("Reshaped y HEAD: ", reshaped_y[:10])
            # _, reshaped_logits_top_k = torch.topk(reshaped_logits[:10], k=3, dim=-1)
            # print("Reshaped logits HEAD: ", reshaped_logits_top_k[:10])

            batch_loss = loss_fn(reshaped_logits, reshaped_y)

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            train_epoch_loss += batch_loss.item()

            if j % 100 == 0 or device == "cpu": # Print every 100 steps, or every line if CPU because it's hella slow
                print(f"Train step: {j} / {len(train_dataloader)}, took {time.time() - start_time:.2f} seconds, iteration loss: {batch_loss.item()}, epoch avg loss: {train_epoch_loss / (j + 1):.4f}")
                start_time = time.time()
            # print("Batch loss: %.4f" % (batch_loss.item()))
            # if j % 25 == 0:
            #     break

        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)
        scheduler.step()

        # Write the model to disk every epoch
        model_path = os.path.join(work_dir, f"model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), model_path)

        eval_metrics = evaluate_transformer(model, X_dev, y_dev, vocab_size, device=device)
        dev_metrics.append(eval_metrics)

        if verbose:
            print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f, Dev Precision: %.4f, Dev Recall: %.4f, Dev F1: %.4f" % (epoch + 1, train_epoch_loss, eval_metrics["loss"], eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"]))
            #print("Epoch: %.d, Train Loss: %.4f" % (epoch + 1, train_epoch_loss))
    return train_losses, dev_metrics[-1]