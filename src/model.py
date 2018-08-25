

from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from dataset import get_dataloaders
from torchsummary import summary
from torchviz import make_dot


class DeepDocClassifier(nn.Module):
    def __init__(self):
        super(DeepDocClassifier, self).__init__()
        pretrained_graph = models.alexnet(pretrained=True)
        feature_extracter = list(pretrained_graph.features.children())
        self.feature_extracter = []
        for child in feature_extracter:
            self.feature_extracter.append(child)
            if isinstance(child, nn.Conv2d):
                self.feature_extracter.append(nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2))
        self.feature_extracter = nn.Sequential(*nn.ModuleList(self.feature_extracter))
        classifier = list(pretrained_graph.classifier.children())[:-1]
        classifier.append(nn.Linear(4096, 10))
        classifier.append(nn.Softmax(dim=1))
        self.classifier = nn.Sequential(*nn.ModuleList(classifier))

    def forward(self, x):
        x = self.feature_extracter(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x, torch.argmax(input=x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def see_children_recursively(graph):
    further = False
    children = list(graph.children())
    for child in children:
        see_children_recursively(child)
        further = True
    if not further:
        print(graph)


def network():
    in_channels = 3
    patch_size = 227
    net = DeepDocClassifier()
    # see_children_recursively(net.classifier)
    x = torch.Tensor(2, in_channels, patch_size, patch_size)
    ### print network in anyway
    print(net)
    # see_children_recursively(net)
    # summary(model=net, input_size=(in_channels, patch_size, patch_size))
    ###
    # print(list(nn.BatchNorm2d(num_features=3).children()))
    # print('We need to find {} numbers!'.format(net.count_parameters()))
    out, pred = net(x)
    print(out.shape, pred.shape)
    # this = make_dot(pred, params=dict(net.named_parameters()))
    # print(type(this))
    # this.view()
    # while True:
    #     pass
    # print(out.shape, pred.shape)


@torch.no_grad()
def check_on_dataloaders():
    model = DeepDocClassifier()
    model.eval()
    train_loader, _, _ = get_dataloaders(base_folder='../dataset/', batch_size=4)
    for idx, data in enumerate(train_loader):
        test_x, label = data['input'], data['label']
        out_x, pred = model.forward(test_x)
        print(out_x.shape, pred.shape)
    pass


if __name__ == '__main__':
    check_on_dataloaders()
















