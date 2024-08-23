import os 
import torch 
import data_setup, engine, model_builder, utils
from torchvision import transforms, models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--num_epochs", help="an integer to perform number of epochs", type=int)
parser.add_argument("-b", "--batch_size", help="an integer of number of element per batch", type=int)
# parser.add_argument("-hu", "--hidden_units", help="an integer of number of hidden units per layer", type=int)
parser.add_argument("-lr", "--learning_rate", help="a float for the learning rate", type=float)

args = parser.parse_args()

# setup hyperparameters 
NUM_EPOCHS = args.num_epochs if args.num_epochs else 10
BATCH_SIZE = args.batch_size # required
# HIDDEN_UNITS = args.hidden_units if args.hidden_units else 10
LEARNING_RATE = args.learning_rate if args.learning_rate else 0.001

# setup directories 
train_dir = "data/pizza_sushi_steak/train"
test_dir = "data/pizza_sushi_steak/test"

def main():
    # setup device agnostic code 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create transforms
    data_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    # create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir, 
        test_dir=test_dir, 
        transform=data_transform, 
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # create model with help from model_builder.py
    model = model_builder.create_model_baseline_effnetb0(out_feats=len(class_names), device=device)

    # set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # start training with help from engine.py 
    engine.train(model=model, 
                    train_dataloader=train_dataloader, 
                    test_dataloader=test_dataloader, 
                    loss_fn=loss_fn, 
                    optimizer=optimizer, 
                    epochs=NUM_EPOCHS, 
                    device=device)
            
    # save the model with help from utils.py
    utils.save_model(model=model, target_dir="models", model_name="tinyfood-effnet.pt")

if __name__ == '__main__':
    main()
