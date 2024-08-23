"""
contains functions for training and testing a pytorch model
"""
import torch 

from tqdm.auto import tqdm 
from typing import Dict, List, Tuple 
# from torch.utils.tensorboard.writer import SummaryWriter

def train_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              optimizer: torch.optim.Optimizer, 
              device: torch.device) -> Tuple[float, float]:
    """Trains a pytorch model for a single epoch 

    turns a target model to training mode then runs through all of the required training steps
    (forward pass, loss calculation, optimizer step).

    Args: 
        model: pytorch model
        dataloader: dataloader insatnce for the model to be trained on 
        loss_fn: pytorch loss function to calculate loss
        optimizer: pytorch optimizer to help minimize the loss function
        device: target device

    returns:
        a tuple of training loss and training accuracy metrics
        in the form (train_loss, train_accuracy)
    """
    # put the model into training mode
    model.train()
    
    # setup train loss and train accuracy 
    train_loss, train_accuracy = 0, 0 

    # loop through data laoder batches
    for batch, (X, y) in enumerate(dataloader):
        # send data to target device 
        X, y = X.to(device), y.to(device)

        # forward pass 
        logits = model(X)

        # calculate loss and accumulate loss 
        loss = loss_fn(logits, y)
        train_loss += loss

        # optimizer zero grad 
        optimizer.zero_grad()

        # loss backward 
        loss.backward()

        # optimizer step 
        optimizer.step()

        # calculate and accumulate accuracy metric across all batches
        preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
        train_accuracy += (preds == y).sum().item()/len(preds)

    # adjust metrics to get average loss and accuracy per batch 
    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)
    return train_loss, train_accuracy

def test_step(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """Tests a pytorch model for a single epoch

    Turns a target model to eval mode and then performs a forward pass on a testing
    dataset. 

    Args: 
        model: pytorch model
        dataloader: dataloader insatnce for the model to be tested on 
        loss_fn: loss function to calculate loss (errors)
        device: target device to compute on 

    returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy)
    """
    # put the model in eval mode
    model.eval()

    # setup test loss and test accuracy 
    test_loss, test_accuracy = 0, 0 

    # turn on inference mode 
    with torch.inference_mode():
        # loop through all batches 
        for X, y in dataloader: 
            # send data to target device
            X, y  = X.to(device), y.to(device)

            # forward pass
            logits = model(X)

            # calculate and accumulate loss
            loss = loss_fn(logits, y)
            test_loss += loss.item()

            # calculate and accumulate accuracy 
            test_preds = torch.softmax(logits, dim=-1).argmax(dim=-1)
            test_accuracy += ((test_preds == y).sum().item()/len(test_preds))
    # adjust metrics to get average loss and accuracy per batch 
    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)
    return test_loss, test_accuracy

def train(model: torch.nn.Module, 
         train_dataloader: torch.utils.data.DataLoader, 
         test_dataloader: torch.utils.data.DataLoader, 
         optimizer: torch.optim.Optimizer, 
         loss_fn: torch.nn.Module, 
         epochs: int, 
         device: torch.device, 
         writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
    """Trains and tests pytorch model

    passes a target model through train_step() and test_step() 
    functions for a number of epochs, training and testing the model in the same epoch loop.

    calculates, prints and stores evaluation metric throughout. 

    Args: 
        model: pytorch model
        train_dataloader: DataLoader instance for the model to be trained on
        test_dataloader: DataLoader instance for the model to be tested on
        optimizer: pytorch optimizer
        loss_fn: pytorch loss function
        epochs: integer indicating how many epochs to train for
        device: target device to compute on 

    returns: 
        A dictionaru of training and testing loss as well as training and testing accuracy 
        metrics. Each metric has a value in a list for each epoch. 

        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
    """
    # create an empty dictionary 
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # loop through training and testing steps for a number of epochs 
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader, 
                                          loss_fn=loss_fn, 
                                          optimizer=optimizer, 
                                          device=device)
        test_loss, test_acc = test_step(model=model, 
                                       dataloader=test_dataloader, 
                                       loss_fn=loss_fn,
                                       device=device)

        if epoch % 1 == 0:
            print(
                f"Epoch: {epoch+1} | " 
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

        # update results dictionary 
        results["train_loss"].append(train_loss.item())
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer: 
            # NEW: EXPERIMENT TRACKING 
            # add loss to SummaryWriter
            writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train loss": train_loss, "test loss": test_loss}, global_step=epoch)
            # add accuracy to SummaryWriter 
            writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train acc": train_acc, "test acc": test_acc}, global_step=epoch)
            # track the pytorch model architecture 
            writer.add_graph(model=model, input_to_model=torch.randn(size=(32, 3, 224, 224)).to(device))
            writer.close()
    # END SummaryWriter tracking process

    # return the filled results dictionaru 
    return results
