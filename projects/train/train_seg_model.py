import time
import torch
import torch.nn as nn
import torch.optim as optim
import models
from ..utils import utils
from ..utils.options import args
from ..data.pascal_voc_2012 import load_data


def train(data_loader, model, optimizer, loss_fn, ):
    model.train()
    for step, (inputs, targets) in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()


def main():
    train_loader, test_loader = load_data(args)
    model = models.__dict__[args.arch](pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()


    for epoch in range(args.init_epoch, args.epochs):
        train()




if __name__ == "__main__":
    main()
