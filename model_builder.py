"""
contains pytorch model code to instantiate a TinyVGG model.
"""
import torch 
from torch import nn 
import torchvision

def create_model_baseline_effnetb0(out_feats: int, device: torch.device = None) -> torch.nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # change the output layer 
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=out_feats, 
                        bias=True)).to(device)
    
    model.name = "effnetb0"
    print(f"[INFO] created a model {model.name}")
    
    return model

def create_model_baseline_effnetb2(out_feats: int, device: torch.device = None) -> torch.nn.Module: 
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT 
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False 

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), 
        nn.Linear(in_features=1408,
                  out_features=out_feats, 
                  bias=True)
    ).to(device)

    model.name = "effnetb2"
    print(f"[INFO] created a model {model.name}")
    
    return model