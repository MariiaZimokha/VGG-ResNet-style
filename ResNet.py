import torch
from torch import nn


from torchvision import models
import torch.optim as optim


def get_model(arch, num_classes, device, pretrained=False):
    weights = "IMAGENET1K_V1"
    if arch == 'resnet50':
        model = models.resnet50(weights=weights)
    elif arch == 'resnet101':
        model = models.resnet101(weights=weights)
    else:
        # default is resnet18
        model = models.resnet18(weights=weights)
    # freeze feature layers if transfer learning
    if pretrained:
        for name, param in model.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

    # replace final layer for dataset with num classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)
