


class AvgMetric(object):
    def __init__(self):
        self.reset():
    
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

class MeanIoU(AvgMetric):
    def __init__(self):
        pass

    def __call__(pred, gt):
        pred = pred.argmax(dim=1, keepdim=True)
        
