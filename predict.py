import argparse
import torch 
import matplotlib.pyplot as plt 
import requests
from PIL import Image
from torchvision import transforms
import data_setup, model_builder
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="string of url to the image", type=str)
args = parser.parse_args()

URL = args.image # required

image_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

IMAGE_PATH = Path("data") / "spoiled-fresh" / "FRUIT-16K"

classes = sorted(entry.name for entry in os.scandir(IMAGE_PATH) if entry.is_dir())

# load saved model 
loaded_model = model_builder.create_model_baseline_effnetb2(out_feats=len(classes), device="cpu")
loaded_model.load_state_dict(torch.load("models/effnetb2_fruitsvegs0_5_epochs.pt", weights_only=True))

def pred_and_plot(model: torch.nn.Module, 
                    image_path: str,
                    transform: transforms.Compose,
                    class_names: list[str] = None):
        # load image
        img = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
        # setup transformed image 
        transformed_img = transform(img)
        # forward pass 
        logits = model(transformed_img.unsqueeze(dim=0))
        pred = torch.softmax(logits, dim=-1).argmax(dim=-1)
        # plot the image along with the label 
        # plt.imshow(transformed_img.permute(1, 2, 0))
        title = f"{class_names[pred]} | {torch.softmax(logits, dim=-1).max():.3f}"
        plt.title(title)
        print(title)

pred_and_plot(model=loaded_model, image_path=URL, 
                transform=image_transform, class_names=classes)