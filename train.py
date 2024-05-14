import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import VGG
from dataset import GrapeDataset

def create_annotations(dir_path):
    """ 
    Returns a Dataframe with images and labels
    """
    X = []
    y = []
    i = 0
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            for file in sorted(os.listdir(dir_path + path)):
                if not file.startswith('.'):
                    X.append(path+'/'+file)
                    y.append(i)
            i +=1
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return pd.DataFrame({'filename':X,
                        'label':y})

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = 'data/test/'
    train_path = 'data/train/'    
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 10
    # Loading data
    ann_train = create_annotations(train_path)
    ann_test = create_annotations(test_path)

    transform = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224,224))])
    train_data = GrapeDataset(img_dir=train_path, ann_df=ann_train,transform=transform)
    test_data = GrapeDataset(img_dir=test_path, ann_df=ann_test,transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Loading the model
    model = VGG(num_classes=4)
    model.to(device)
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        # Train
        model.train()
        for idx, (X,y) in loop:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = loss_fn(preds, y)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch:[{epoch}/{num_epochs}](Train)")
            loop.set_postfix(train_loss=loss.item())

        train_loss = train_running_loss / (idx + 1)

        # Val
        val_running_loss = 0.0
        run_accuracy = 0
        model.eval()
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        with torch.no_grad():
            for idx, (X,y) in loop:
                X, y = X.to(device), y.to(device)

                preds = model(X)
                loss = loss_fn(preds, y)
                val_running_loss += loss.item()
                # accuracy = calc_accuracy(preds, mask, cfg.THR)
                # run_accuracy += accuracy.item()
                loop.set_description(f"Epoch:[{epoch}/{num_epochs}](Val)")
                # loop.set_postfix(val_loss=loss.item(), acc=accuracy.item())
        if epoch % 20 == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    "checkpoint.pt",
                )


if __name__ == "__main__":
    main()
    