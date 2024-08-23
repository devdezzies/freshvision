"""
contains various utility functions for pytorch model training and saving
"""
import torch 
from pathlib import Path 
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter

def save_model(model: torch.nn.Module, 
              target_dir: str, 
              model_name: str):
    """Saves a pytorch model to a target directory

    Args:
        model: target pytorch model
        target_dir: string of target directory path to store the saved models 
        model_name: a filename for the saved model. Should be included either ".pth" or ".pt" as 
        the file extension.
    """
    # create target directory 
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # create model save path 
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    # save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    img_list = Image.open(image_path)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    # target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(img_list)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)


def create_writer(experiment_name: str, model_name: str, extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter(): # type: ignore
    """
    creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a
    specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment
        model_name (str): model name
        extra (str, optional): anything extra to add to the directory. Defaults is None

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir

    Examples usage:
        this is gonna create writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs"

    writer = create_writer(experiment_name="data_10_percent", model_name="effnetb2", extra="5_epochs")

    This is the same as:
    writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs")
    """

    from datetime import datetime
    import os

    # get the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter(), saving to: {log_dir}")
    
    return SummaryWriter(log_dir=log_dir)
