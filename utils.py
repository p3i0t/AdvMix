import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50,\
    wide_resnet50_2, wide_resnet101_2, resnext50_32x4d, resnext101_32x8d




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_parameters(model):
    """
    Calculate the number of parameters of a Pytorch model.
    :param model: torch.nn.Module
    :return: int, number of parameters.
    """
    return sum([para.numel() for para in model.parameters()])


def get_model(name='resnet18', n_classes=10):
    classifier = eval(name)(pretrained=False)  # load model from torchvision
    classifier.avgpool = nn.AdaptiveAvgPool2d(1)
    classifier.fc = nn.Linear(classifier.fc.in_features, n_classes)
    return classifier

