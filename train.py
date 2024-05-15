import os
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import wandb

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
                    X.append(path + '/' + file)
                    y.append(i)
            i += 1
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return pd.DataFrame({'filename': X, 'label': y})

def calculate_accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def main_worker(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        wandb.init(project="grape-classification")
        wandb.config = {
            "learning_rate": 1e-3,
            "epochs": 10,
            "batch_size": 16
        }

    test_path = 'data/test/'
    train_path = 'data/train/'    
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 10

    # Loading data
    if rank == 0:
        ann_train = create_annotations(train_path)
        ann_test = create_annotations(test_path)

    # Ensure that all processes have the same dataset
    dist.barrier()
    if rank != 0:
        ann_train = create_annotations(train_path)
        ann_test = create_annotations(test_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    train_data = GrapeDataset(img_dir=train_path, ann_df=ann_train, transform=transform)
    test_data = GrapeDataset(img_dir=test_path, ann_df=ann_test, transform=transform)

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

    # Loading the model
    model = VGG(num_classes=4).to(device)
    model = DDP(model, device_ids=[rank])

    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        train_running_loss = 0.0
        train_running_accuracy = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, position=rank)
        # Train
        model.train()
        for idx, (X, y) in loop:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = loss_fn(preds, y)
            train_running_loss += loss.item()
            accuracy = calculate_accuracy(preds, y)
            train_running_accuracy += accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch:[{epoch}/{num_epochs}](Train)")
            loop.set_postfix(train_loss=loss.item(), train_acc=accuracy)

        train_loss = train_running_loss / len(train_loader)
        train_acc = train_running_accuracy / len(train_loader)

        # Val
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        model.eval()
        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False, position=rank)
        with torch.no_grad():
            for idx, (X, y) in loop:
                X, y = X.to(device), y.to(device)

                preds = model(X)
                loss = loss_fn(preds, y)
                val_running_loss += loss.item()
                accuracy = calculate_accuracy(preds, y)
                val_running_accuracy += accuracy
                loop.set_description(f"Epoch:[{epoch}/{num_epochs}](Val)")
                loop.set_postfix(val_loss=loss.item(), val_acc=accuracy)

        val_loss = val_running_loss / len(test_loader)
        val_acc = val_running_accuracy / len(test_loader)

        if rank == 0:
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "test_loss": val_loss,
                "test_accuracy": val_acc,
            })

        if epoch % 20 == 0 and rank == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                "checkpoint.pt",
            )

    dist.destroy_process_group()

def main():
    world_size = 4  # Number of GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
