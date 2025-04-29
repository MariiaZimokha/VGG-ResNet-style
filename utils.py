import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from torchvision import models
# from torchsummary import summary

import torch.optim as optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gc
import time
from datetime import timedelta


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    return device


def get_dataloaders(dataset_name, batch_size=128):
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:  # MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
            transforms.Resize((32, 32))  # Resize to match CIFAR dimensions
        ])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader, 10


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")


def plot_metrics(metrics, dataset, epochs):
    plt.figure(figsize=(8, 10))

    plt.subplot(4, 1, 1)
    for label, values in metrics['train_acc'].items():
        plt.plot(epochs, values, label=label)
    plt.title(f"{dataset.upper()} - Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 2)
    for label, values in metrics['test_acc'].items():
        plt.plot(epochs, values, label=label)
    plt.title(f"{dataset.upper()} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 3)
    for label, values in metrics['train_loss'].items():
        plt.plot(epochs, values, label=label)
    plt.title(f"{dataset.upper()} - Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 4)
    for label, values in metrics['test_loss'].items():
        plt.plot(epochs, values, label=label)
    plt.title(f"{dataset.upper()} - Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def create_summary_table(dataset_name="mnist", architectures=[], modes=[], results_dir=""):
    summary_rows = []

    for arch in architectures:
        for mode in modes:
            filename = f"{dataset_name}_{arch}_{mode}_metrics.csv"
            path = os.path.join(results_dir, filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                final = df.iloc[-1]
                best = df.loc[df['test_acc'].idxmax()]
                summary_rows.append({
                    'Dataset': dataset_name.upper(),
                    'Model': arch,
                    'Mode': mode,
                    'Final Train Acc': round(final['train_acc'], 4),
                    'Final Test Acc': round(final['test_acc'], 4),
                    'Final Test Loss': round(final['test_loss'], 4),
                    'Best Epoch': int(best['epoch']),
                    'Best Test Acc': round(best['test_acc'], 4),
                })
    return pd.DataFrame(summary_rows)




def get_predictions(model, data_loader, device):
    true = []
    preds = []

    for data, targets in data_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model.predict(data)
        true.extend(targets.cpu().numpy().tolist())
        preds.extend(outputs)
    return true, preds


def plot_reports(class_names, y_true, y_pred):
    target_names = [f"{i}" for i in range(len(class_names))]

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot()
    plt.show()

    print(classification_report(y_true, y_pred, target_names=target_names))


def test_after_training(model, data_loader, device, class_names):
    y_true, y_pred = get_predictions(model, data_loader, device)
    plot_reports(class_names, y_true, y_pred)
    return y_true, y_pred


