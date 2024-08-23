import torch
import utils
import model_builder as mb
import engine

# 1. set seed
utils.set_seeds(seed=42)

# 2. keep track of experiment numbers 
experiment_number = 0 

# 3. loop through each dataloader 
def run_experiment(train_dataloaders: dict, test_dataloader: torch.utils.data.DataLoader,  num_epochs: int, models: list[str], class_names: list[str], device: torch.device = None):
    for dataloader_name, train_dataloader in train_dataloaders.items():
    # 4. loop through each number of epochs 
        for epochs in num_epochs:
        # 5. loop through each model name and create a new model based on the name 
            for model_name in models:
            # 6. create information prints out 
                experiment_number += 1 
                print(f"[INFO] experiment number: {experiment_number}")
                print(f"[INFO] model: {model_name}")
                print(f"[INFO] dataloader: {dataloader_name}")
                print(f"[INFO] number of epochs: {epochs}")

                # 7. select the model 
                if model_name == "effnetb0":
                    model = mb.create_model_baseline_effnetb0(out_feats=len(class_names), device=device)
                else: 
                    model = mb.create_model_baseline_effnetb2(out_feats=len(class_names), device=device)
            
                # 8. create a new loss function for every model 
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

                # 9. train target model with target dataloaders and track experiment
                engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs, device=device, writer=utils.create_writer(experiment_name=dataloader_name, model_name=model_name, extra=f"{epochs}_epochs"))

                # 10. save the model to file
                save_filepath = f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pt"
                utils.save_model(model=model, target_dir="models", model_name=save_filepath)
                print("-"*50+"\n")