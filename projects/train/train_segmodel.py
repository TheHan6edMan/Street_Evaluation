from __future__ import absolute_import
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("../")
import models
from utils import utils
from utils.options import args
from data.cityscapes import load_data


args.working_dir = os.path.abspath(os.path.join(args.working_dir, args.dataset, args.arch))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n==> args:\n", args)


def apply_lr_schedule(optimizer, epoch, step, steps, power=0.9):
    max_iter = steps * (args.epochs - args.init_epoch)
    iter_ = (epoch + 1) * steps + step + 1
    lr = (1 - iter_ / max_iter)**power
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train(loader, model, optimizer, loss_fn, epoch, logger):
    model.train()
    num_data = len(loader.dataset)
    steps = len(loader.dataset) // args.batch_size_train
    loss_metric = utils.MeanMetric()
    mIoU_metric = utils.MeanIoU(args.n_classes)
    for step, (inputs, targets) in enumerate(loader):
        t0 = time.time()
        optimizer.zero_grad()
        lr = apply_lr_schedule(optimizer, epoch, step, steps)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        print("output got")
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_metric.update(loss, inputs.shape[0])
        mIoU_metric.update(outputs, targets)
        if (step + 1) % (steps // 10) == 0:
            logger.info("Epoch[{}/{}] ({}/{}):\tlr={:.4f}\ttraining loss={:.4f}\tmIoU={:.2f}\ttime={:.2f}".format(
                    epoch, args.epochs, (step+1)*args.batch_size_train, num_data,
                    lr, loss_metric.result(), mIoU_metric.result(), time.time()-t0
            ))
            break


def test(loader, model, loss_fn, logger):
    model.eval()
    num_data = len(loader.dataset)
    loss_metric = utils.MeanMetric()
    mIoU_metric = utils.MeanIoU(args.n_classes)
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(loader):
            t0 = time.time()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss_metric.update(loss, inputs.shape[0])
            mIoU_metric.update(outputs, targets)
            if (step + 1) % (num_data // 10) == 0:
                logger.info("test loss={:.4f}\tmIoU={:.2f}\ttime={:.2f}".format(
                        loss_metric.result(), mIoU_metric.result(), time.time()-t0
                ))
            break
    return mIoU_metric.result()


def main():
    max_mIoU = 0.0
    ckpter = utils.Checkpointer(args)
    logger = utils.get_logger(args.working_dir)
    train_loader, test_loader = load_data(args)
    model = models.__dict__[args.arch](state_dict_dir=args.baseline_dir, n_classes=args.n_classes).to(device)
    if device == "cuda":
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    print(model)
    print(model(torch.rand([2, 3, 512, 1024])).shape)
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    for epoch in range(args.init_epoch, args.epochs):
        train(train_loader, model, optimizer, loss_fn, epoch, logger)
        mIoU = test(test_loader, model, loss_fn, logger)
        is_best = max_mIoU < mIoU
        max_mIoU = max(max_mIoU, mIoU)

        model_state_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
        state_dict = {
            "epoch": epoch,
            "max_mIoU": max_mIoU,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        ckpter.save_model(state_dict, epoch, is_best)
        logger.info("Max mIoU: {:.3f}".format(max_mIoU))
        break


if __name__ == "__main__":
    main()
