from __future__ import absolute_import
import os
import shutil
import datetime
import logging
import torch
from pathlib import Path


class MeanMetric(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.value = 0.0
        self.accumulator = 0.0
        self.counter = 0
    
    def update(self, value, n):
        self.value = value
        self.accumulator += value
        self.counter += n
    
    def result(self):
        return self.accumulator /  self.counter

class MeanIoU(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def update(self, outputs, targets):
        with torch.no_grad():
            outputs = outputs.argmax(dim=1, keepdim=True)
            outputs, targets = outputs.cpu(), targets.cpu()
            for output, target in zip(outputs, targets):
                output, target = output.flatten(),  target.flatten()
                mask = (target >= 0) * (target < self.n_classes)
                self.confusion_matrix += torch.bincount(
                    self.n_classes * target[mask] + output[mask],
                    minlength=self.n_classes**2
                ).reshape(self.n_classes, self.n_classes)
    
    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes))

    def result(self):
        with torch.no_grad():
            matrix = self.confusion_matrix
            IoU = matrix.diag() / (matrix.sum(dim=0) + matrix.sum(dim=1) - matrix.diag())
            return IoU.mean().item()


class Checkpointer():
    def __init__(self, args):
        self.working_dir = Path(args.working_dir)
        self.ckpt_dir = self.working_dir / "checkpoint"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        with open(self.working_dir / 'config.txt', 'w') as f:
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            f.write(now + '\n\n')
            for a in vars(args):
                f.write('{}: {}\n'.format(a, getattr(args, a)))
            f.write('\n')

    def save_model(self, state_dict, epoch, is_best):
        save_path = f'{self.ckpt_dir}/model_checkpoint.pt'
        torch.save(state_dict, save_path)
        if is_best:
            shutil.copyfile(save_path, f'{self.ckpt_dir}/model_best.pt')


def get_logger(working_dir):
    file_path = os.path.join(working_dir, "logger.log")
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger
        
