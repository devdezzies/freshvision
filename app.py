import gradio as gr 
import model_builder as mb 
from torchvision import transforms
import torch 

device = torch.device("cpu")

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])

manual_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize(size=(224, 224)), 
    transforms.ToTensor(), 
    normalize
])

# class_names = ['Fresh Banana',
#   'Fresh Lemon',
#   'Fresh Lulo',
#   'Fresh Mango',
#   'Fresh Orange',
#   'Fresh Strawberry',
#   'Fresh Tamarillo',
#   'Fresh Tomato',
#   'Spoiled Banana',
#   'Spoiled Lemon',
#   'Spoiled Lulo',
#   'Spoiled Mango',
#   'Spoiled Orange',
#   'Spoiled Strawberry',
#   'Spoiled Tamarillo',
#   'Spoiled Tomato']

class_names = ['Fresh Apple',
 'Fresh Banana',
 'Fresh Orange',
 'Rotten Apple',
 'Rotten Banana',
 'Rotten Orange']

model = mb.create_model_baseline_effnetb0(out_feats=len(class_names), device=device)
model.load_state_dict(torch.load(f="models/effnetb0_freshvisionv0_10_epochs.pt", map_location="cpu"))

def pred(img):
    model.eval()
    transformed = manual_transform(img).to(device)
    with torch.inference_mode():
        logits = model(transformed.unsqueeze(dim=0))
        pred = torch.softmax(logits, dim=-1)
    return f"prediction: {class_names[pred.argmax(dim=-1).item()]} | confidence: {pred.max():.3f}"

demo = gr.Blocks()

with demo: 
    gr.Markdown("""
        # Welcome to FreshVision 📷
        _FreshVision is a machine learning model to classify freshness for fruits. This model 
        utilizes transfer learning from pre-trained model from PyTorch [EfficientNetB0](https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
        This model has been trained on [kaggle datasets](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) using NVIDIA T4 GPU._
        
        ## Model capabilities:
        - Classify freshness from fruits image (apple, orange, and banana) with two labels: _Fresh_ and _Rotten/spoiled_
        ## Model drawbacks: 
        - Sometimes perform false prediction on some fruits condition, this is due to low variability on the image datasets
        - Can't perform accurate prediction on multiple objects/combined condition (e.g. two bananas with different freshness condition)
        - This models can't identify non-fruits objects , since it's only trained with fruits dataset. 
                
        ## **How to get the best result with this model:** 
        1. The image should only contain fruits that the model can recognize (apple, orange, and banana)
        2. The image should only contain one object (one fruit)
        3. Ensure the object is captured with sufficient light so that the surface of the fruits is exposed properly
                
        get the [source code](https://github.com/devdezzies/freshvision) on my github
""")
    gr.Interface(pred, gr.Image(), outputs="text")

if __name__ == "__main__":
    demo.launch()