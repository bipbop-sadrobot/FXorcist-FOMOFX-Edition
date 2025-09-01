import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

class Accuracy(BinaryAccuracy):
    def __init__(self):
        super().__init__()

class Precision(BinaryPrecision):
    def __init__(self):
        super().__init__()

class Recall(BinaryRecall):
    def __init__(self):
        super().__init__()

class F1Score(BinaryF1Score):
    def __init__(self):
        super().__init__()
